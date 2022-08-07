#!/usr/bin/env python3
import collections
import copy
import os
import numpy as np
import rospy

import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import ros_numpy

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TrajectoryPoint
from quadrotor_msgs.msg import Trajectory
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation as R
from agile_autonomy_msgs.msg import MultiTrajectory

from .models.plan_learner import PlanLearner


class PlanLearningAirsim(object):
    def __init__(self, config):
        self.config = config
        self.maneuver_complete = False
        self.use_network = False
        self.net_initialized = False
        self.reference_initialized = False
        self.odometry_used_for_inference = None
        self.time_prediction = None
        self.reference_progress = 0
        self.reference_len = 1
        self.end_ref_percentage = 0.8

        # Input Stuff
        self.bridge = CvBridge()
        self.quad_name = config.quad_name  # hummingbird
        self.depth_topic = config.depth_topic  # agile_autonomy/sgm_depth
        self.odometry_topic = config.odometry_topic  # ground_truth/odometry
        self.odometry = Odometry()
        self.odom_rot_input = None
        self.odom_rot = None
        self.depth = np.zeros(
            (self.config.img_height, self.config.img_width, 3))
        # Queue Stuff
        self.depth_queue = collections.deque([], maxlen=self.config.input_update_freq)  # len = 15
        self.state_queue = collections.deque([], maxlen=self.config.input_update_freq)  # len = 15
        self.reset_queue()

        # Init Network
        self.learner = PlanLearner(settings=config)
        self._prepare_net_inputs()
        _ = self.learner.inference(self.net_inputs)
        self.net_initialized = True
        print("Net initialized")

        # ROS Stuff
        # Subscriber
        self.ground_truth_odom = rospy.Subscriber("/" + self.quad_name + "/" + self.odometry_topic,
                                                  Odometry,
                                                  self.callback_gt_odometry,
                                                  queue_size=1)
        if self.config.use_depth:
            print('depth topic subscribe')
            self.depth_sub = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", Image, 
                                                self.callback_depth,queue_size=1)

        self.fly_sub = rospy.Subscriber("/" + self.quad_name + "/agile_autonomy/start_flying", Bool,
                                        self.callback_fly, queue_size=1)  # Receive and fly, python<->agile_autonomy
        # self.reset_sub = rospy.Subscriber("success_reset", Empty,
        #                                      self.callback_success_reset,
        #                                      queue_size=1)  # python to python
        # Publisher
        self.fly_pub = rospy.Publisher("/hummingbird/agile_autonomy/start_flying", Bool,
                                       queue_size=1)  # Stop upon some condition/ publish to quit the nw prediction
        self.traj_pub = rospy.Publisher("/{}/trajectory_predicted".format(self.quad_name), MultiTrajectory,
                                        queue_size=1)  # Stop upon some condition

        self.infer_pose_pub = rospy.Publisher("infer_pose_pub", Odometry, queue_size=10)

        # Timer update input and pub traj
        self.last_depth_received = rospy.Time.now()
        self.timer_input = rospy.Timer(rospy.Duration(1. / self.config.input_update_freq),
                                       self.update_input_queues)
        self.timer_net = rospy.Timer(rospy.Duration(1. / self.config.network_frequency),
                                     self._generate_plan)
        self.timer_check = rospy.Timer(
            rospy.Duration(1. / 20.),
            self.check_task_progress)

    def reset_queue(self):
        self.depth_queue.clear()
        self.state_queue.clear()
        self.odom_rot = [0 for _ in range(9)]
        n_init_states = 21
        # 3 position(not used)
        # 9 attitude R matrix, 3 linear vel, 3 angular vel, 3 desired direction ??
        for _ in range(self.config.input_update_freq):
            self.depth_queue.append(np.zeros_like(self.depth))
            self.state_queue.append(np.zeros((n_init_states,)))

    ###########################
    #    Callback functions   #
    ###########################
    def callback_gt_odometry(self, data):
        odometry = data
        rot_body = R.from_quat([odometry.pose.pose.orientation.x,
                                odometry.pose.pose.orientation.y,
                                odometry.pose.pose.orientation.z,
                                odometry.pose.pose.orientation.w])
        self.odom_rot_input = rot_body.as_dcm().reshape((9,)).tolist()
        self.odom_rot = rot_body.as_dcm().reshape((9,)).tolist()
        self.odometry = odometry

    def callback_depth(self, data):
        '''
        Reads a depth image and saves it.
        '''
        if self.quad_name == 'hummingbird':
            depth = self.bridge.imgmsg_to_cv2(data, '8UC1')
            # depth = np.frombuffer(data.data, dtype=np.uint16)
            print("============================================================")
            print("Min Depth {}. Max Depth {}. with Nans {}".format(np.min(depth),
                                                                    np.max(depth),
                                                                    np.any(np.isnan(depth))))
            print("Min Depth {}. Max Depth {}. with Nans {}".format(np.min(depth),
                                                                    np.max(depth),
                                                                    np.any(np.isnan(depth))))
        else:
            print("Invalid quad_name!")
            raise NotImplementedError
        if (np.sum(depth) != 0) and (not np.any(np.isnan(depth))):
            # depth = np.minimum(depth, 20000)
            dim = (self.config.img_width, self.config.img_height)
            depth = cv2.resize(depth, dim)
            depth = np.array(depth, dtype=np.float32)
            # depth = depth / (80)  # normalization factor to put depth in (0,255)
            depth = np.expand_dims(depth, axis=-1)
            depth = np.tile(depth, (1, 1, 3))
            # cv2.imshow("depth_input", depth)
            # cv2.waitKey(1)
            self.depth = depth
            self.last_depth_received = rospy.Time.now()

    # def callback_success_reset(self, data):
    #     # at this point agile autonomy plan reference trajectory
    #     # publish True "start flying" topic -> fly_callback
    #     print("Received call to Clear Buffer and Restart Experiment")
    #     self.use_network = False
    #     self.reference_initialized = False
    #     self.maneuver_complete = False
    #     # Learning phase to test
    #     tf.keras.backend.set_learning_phase(0)
    #     # self.n_times_expert = 0.000
    #     # self.n_times_net = 0.001
    #     # self.crashed = False
    #     # self.exp_failed = False
    #     # self.planner_succed = True
    #     self.reset_queue()
    #     # print("Resetting experiment")
    #     # os.system("rosservice call /gazebo/unpause_physics")
    #     print('Done Reset')

    # def callback_land(self, data):
    #     self.config.execute_nw_predictions = False

    def callback_fly(self, data):
        tf.keras.backend.set_learning_phase(0)
        self.reset_queue()
        # If self.use_network is true, then trajectory is already loaded
        if data.data and (not self.use_network):
            # Load pointcloud and make kdtree out of it
            rollout_dir = os.path.join(self.config.expert_folder,
                                       sorted(os.listdir(self.config.expert_folder))[-1])
            # Load the reference trajectory
            traj_fname = os.path.join(rollout_dir, "reference_trajectory.csv")
            print("Reading Trajectory from %s" % traj_fname)
            self.load_trajectory(traj_fname)
            # self.reference_initialized = True
            # only enable network when KDTree and trajectory are ready

        # Might be here if you crash in less than a second.
        if self.maneuver_complete:
            return
        # If true, network should fly.
        # If false, maneuver is finished and network is off.
        self.use_network = data.data and self.config.execute_nw_predictions
        if (not data.data):
            self.maneuver_complete = True
            self.use_network = False

    def load_trajectory(self, traj_fname):
        self.reference_initialized = False
        traj_df = pd.read_csv(traj_fname, delimiter=',')
        self.reference_len = traj_df.shape[0]
        self.full_reference = Trajectory()
        time = ['time_from_start']
        time_values = traj_df[time].values
        pos = ["pos_x", "pos_y", "pos_z"]
        pos_values = traj_df[pos].values
        vel = ["vel_x", "vel_y", "vel_z"]
        vel_values = traj_df[vel].values
        for i in range(self.reference_len):
            point = TrajectoryPoint()
            point.time_from_start = rospy.Duration(time_values[i])
            point.pose.position.x = pos_values[i][0]
            point.pose.position.y = pos_values[i][1]
            point.pose.position.z = pos_values[i][2]
            point.velocity.linear.x = vel_values[i][0]
            point.velocity.linear.y = vel_values[i][1]
            point.velocity.linear.z = vel_values[i][2]
            self.full_reference.points.append(point)
        # Change type for easier use
        self.full_reference = self.full_reference.points
        self.reference_progress = 0
        self.reference_initialized = True
        assert len(self.full_reference) == self.reference_len
        print("Loaded traj {} with {} elems".format(
            traj_fname, self.reference_len))
        return

    def check_task_progress(self, _timer):
        # go here if there are problems with the generation of traj
        # No need to check anymore
        if self.maneuver_complete:
            return
        # check if reference is ready
        if not self.reference_initialized:
            return
        # check if it is near to end of trajectory
        if (self.reference_progress / (self.reference_len)) > self.end_ref_percentage:
            print("It worked well. (Arrived at %d / %d)" % (self.reference_progress, self.reference_len))
            self.maneuver_complete = True
            self.use_network = False
            print("Giving a stop from python")
            msg = Bool()
            msg.data = False
            self.fly_pub.publish(msg)

    def update_input_queues(self, data):
        # Positions are ignored in the new network
        if ((self.odometry is None) or
                (self.odom_rot_input is None) or
                (self.odom_rot is None)):
            return

        imu_states = [self.odometry.pose.pose.position.x,
                      self.odometry.pose.pose.position.y,
                      self.odometry.pose.pose.position.z] + \
                     self.odom_rot_input

        vel = np.array([self.odometry.twist.twist.linear.x,
                        self.odometry.twist.twist.linear.y,
                        self.odometry.twist.twist.linear.z])

        # vel = vel / np.linalg.norm(vel) * 7.
        vel = vel.squeeze()
        imu_states = imu_states + vel.tolist()

        if self.config.use_bodyrates:
            imu_states.extend([self.odometry.twist.twist.angular.x,
                               self.odometry.twist.twist.angular.y,
                               self.odometry.twist.twist.angular.z])

        if self.reference_initialized:
            quad_position = np.array([self.odometry.pose.pose.position.x,
                                      self.odometry.pose.pose.position.y,
                                      self.odometry.pose.pose.position.z]).reshape((3, 1))
            self.update_reference_progress(quad_position)
            ref_idx = np.minimum(self.reference_progress +
                                 int(self.config.future_time * 50), self.reference_len - 1)
        else:
            ref_idx = 0
        if self.reference_initialized:
            reference_point = self.full_reference[ref_idx]
            reference_position_wf = np.array([reference_point.pose.position.x,
                                              reference_point.pose.position.y,
                                              reference_point.pose.position.z]).reshape((3, 1))
            current_position_wf = np.array([self.odometry.pose.pose.position.x,
                                            self.odometry.pose.pose.position.y,
                                            self.odometry.pose.pose.position.z]).reshape((3, 1))
            difference = reference_position_wf - current_position_wf
            difference = difference / np.linalg.norm(difference)
            goal_dir = self.adapt_reference(difference)
        else:
            # Reference is not loaded at init, but we want to keep updating the list anyway
            goal_dir = np.zeros((3, 1))
        goal_dir = np.squeeze(goal_dir).tolist()

        state_inputs = imu_states + goal_dir
        self.state_queue.append(state_inputs)
        # Prepare Depth
        if self.config.use_depth:
            self.depth_queue.append(self.depth)

    def update_reference_progress(self, quad_position):
        reference_point = self.full_reference[self.reference_progress]
        reference_position_wf = np.array([reference_point.pose.position.x,
                                          reference_point.pose.position.y,
                                          reference_point.pose.position.z]).reshape((3, 1))
        distance = np.linalg.norm(reference_position_wf - quad_position)
        for k in range(self.reference_progress + 1, self.reference_len):
            reference_point = self.full_reference[k]
            reference_position_wf = np.array([reference_point.pose.position.x,
                                              reference_point.pose.position.y,
                                              reference_point.pose.position.z]).reshape((3, 1))
            next_point_distance = np.linalg.norm(reference_position_wf - quad_position)
            if next_point_distance > distance:
                break
            else:
                self.reference_progress = k
                distance = next_point_distance

    def adapt_reference(self, goal_dir):
        if self.config.velocity_frame == 'wf':
            return goal_dir
        elif self.config.velocity_frame == 'bf':
            R_W_C = np.array(self.odom_rot).reshape((3, 3))
            v_C = R_W_C.T @ goal_dir
            return v_C
        else:
            raise IOError("Reference frame not recognized")

    def select_inputs_in_freq(self, input_list):
        new_list = []
        for i in self.required_elements:
            new_list.append(input_list[i])
        return new_list

    def _prepare_net_inputs(self):
        if not self.net_initialized:
            # prepare the elements that need to be fetched in the list
            # required_elements = np.arange(start=0, stop=self.config.input_update_freq,
            #                               step=int(np.ceil(self.config.input_update_freq / self.config.seq_len)),
            #                               dtype=np.int64)
            # required_elements = -1 * (required_elements + 1)  # we need to take things at the end :)
            # self.required_elements is always [-1] if seq_len is 1
            # to make network input with
            # self.required_elements = [i for i in reversed(required_elements.tolist())]
            self.required_elements = [-1]
            # return fake input for init
            if self.config.use_bodyrates:  # True
                n_init_states = 21
            else:
                n_init_states = 18
            inputs = {'depth': np.zeros((1, self.config.seq_len, self.config.img_height, self.config.img_width, 3),
                                        dtype=np.float32),  # (1,1,224,224,3)
                      'imu': np.zeros((1, self.config.seq_len, n_init_states), dtype=np.float32)}  # (1,1,21)
            self.net_inputs = inputs
            return
        # state_inputs = np.stack(self.select_inputs_in_freq(self.state_queue), axis=0)
        # state_inputs = np.array(state_inputs, dtype=np.float32)
        state_inputs = np.array([self.state_queue[-1]], dtype=np.float32)
        new_dict = {'imu': np.expand_dims(state_inputs, axis=0)}
        if self.config.use_depth:
            # depth_inputs = np.stack(self.select_inputs_in_freq(self.depth_queue), axis=0)
            # depth_inputs = np.array(depth_inputs, dtype=np.float32)
            depth_inputs = np.array([self.depth_queue[-1]], dtype=np.float32)
            new_dict['depth'] = np.expand_dims(depth_inputs, axis=0)
        self.odometry_used_for_inference = copy.deepcopy(self.odometry)
        self.time_prediction = rospy.Time.now()
        self.net_inputs = new_dict

    # def evaluate_dagger_condition(self):
    #     # Real world dagger condition is only based on time
    #     if self.reference_progress < 50:
    #         # At the beginning always use expert (otherwise gives problems)
    #         print("Expert warm up!")
    #         return False
    #     else:
    #         print("Network in action")
    #         return True

    def _generate_plan(self, _timer):
        if (not self.net_initialized) or \
                (not self.reference_initialized) or \
                (not self.config.perform_inference):
            return
        t_start = rospy.Time.now()
        if (t_start - self.last_depth_received).to_sec() > 10.0:
            print("Stopping because no depth received")
            self.maneuver_complete = True
            # self.callback_land(Empty())
            msg = Bool()
            msg.data = False
            self.fly_pub.publish(msg)
        self._prepare_net_inputs()

        imu_data = self.net_inputs['imu']
        imu_data = imu_data.squeeze()
        xyz_pose = imu_data[:3]

        rot_mat = imu_data[3:12]

        rot = R.from_dcm(rot_mat.reshape(3, 3))
        rot_qaut = rot.as_quat()
        infer_odometry = Odometry()
        infer_odometry.pose.pose.position.x = xyz_pose[0]
        infer_odometry.pose.pose.position.y = xyz_pose[1]
        infer_odometry.pose.pose.position.z = xyz_pose[2]
        infer_odometry.pose.pose.orientation.x = rot_qaut[0]
        infer_odometry.pose.pose.orientation.y = rot_qaut[1]
        infer_odometry.pose.pose.orientation.z = rot_qaut[2]
        infer_odometry.pose.pose.orientation.w = rot_qaut[3]
        infer_odometry.header.stamp = rospy.Time.now()
        infer_odometry.header.frame_id = "world"
        self.infer_pose_pub.publish(infer_odometry)

        results = self.learner.inference(self.net_inputs)
        self.trajectory_decision(results)

    def trajectory_decision(self, net_predictions):
        # print("Network in action")
        net_in_control = True
        # select best traj
        multi_traj = MultiTrajectory()
        multi_traj.execute = net_in_control
        multi_traj.header.stamp = self.time_prediction
        multi_traj.ref_pose = self.odometry_used_for_inference.pose.pose
        best_alpha = net_predictions[0][0]
        for i in range(self.config.modes):
            traj_pred = net_predictions[1][i]
            alpha = net_predictions[0][i]
            # convert in a traj
            if i == 0 or (best_alpha / alpha > self.config.accept_thresh):
                traj_pred = self._convert_to_traj(traj_pred)
                multi_traj.trajectories.append(traj_pred)
        self.traj_pub.publish(multi_traj)

    def _convert_to_traj(self, net_prediction):
        net_prediction = np.reshape(net_prediction, ((self.config.state_dim, self.config.out_seq_len)))
        pred_traj = Trajectory()
        sample_time = np.arange(start=1, stop=self.config.out_seq_len + 1) / 10.0
        for k, t in enumerate(sample_time):
            point = TrajectoryPoint()
            point.heading = 0.0
            point.time_from_start = rospy.Duration(t)
            point.pose.position.x = net_prediction[0, k]
            point.pose.position.y = net_prediction[1, k]
            point.pose.position.z = net_prediction[2, k]
            pred_traj.points.append(point)
        return pred_traj
