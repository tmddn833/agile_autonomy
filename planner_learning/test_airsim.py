#!/usr/bin/env python3

import argparse
import time
import numpy as np
import rospy
from PlannerLearning import PlannerAirsim
from common import  MessageHandler


from config.settings import create_settings

MAX_TIME_EXP = 500  # in second, if it takes more the process needs to be killed


class Trainer:
    def __init__(self, settings):
        rospy.init_node('iterative_learning_node', anonymous=False)
        self.settings = settings
        np.random.seed(self.settings.random_seed)
        self.msg_handler = MessageHandler()

    # def start_experiment(self):
    #     self.msg_handler.publish_reset()  # /success_reset
    #     # place_quad_at_start(self.msg_handler)
    #     # # Save point_cloud
    #     # if self.settings.execute_nw_predictions:
    #     #     self.msg_handler.publish_save_pc() /save_pc
    #     # self.msg_handler.publish_autopilot_off()  # /hummingbird/autopilot/off
    #     # reset quad to initial position
    #     self.msg_handler.publish_arm_bridge()  # /hummingbird/bridge/arm
    #     self.msg_handler.publish_autopilot_start()  # /hummingbird/autopilot/start

    def perform_testing(self):
        # Perform rollout just one time!
        self.learner = PlannerAirsim.PlanLearningAirsim(self.settings)
        # Start Experiment
        self.learner.maneuver_complete = False  # Just to be sure
        # unity_start_pos = setup_sim(self.msg_handler, config=self.settings)
        # self.msg_handler.publish_autopilot_off()
        # self.start_experiment()
        # self.msg_handler.publish_arm_bridge()  # /hummingbird/bridge/arm
        self.msg_handler.publish_reset()  # /success_reset
        # at this point agile autonomy plan reference trajectory
        # publish True "start flying" topic
        # self.msg_handler.publish_autopilot_start()  # /hummingbird/autopilot/start
        start = time.time()
        exp_failed = False
        self.expert_done = False  # Re-init to be sure
        while not self.learner.maneuver_complete:
            # maneuver_complete turned off when the drone is close to the end of reference.
            # change in callback_fly function in learner
            time.sleep(0.1)
            duration = time.time() - start
            if duration > MAX_TIME_EXP:
                print("Current experiment failed. Finish the rollout")
                exp_failed = True
                break
        # if not exp_failed:
        #     # Wait one second to stop recording
        #     time.sleep(1)
        # else:
        #     # Wait one second to stop recording
        #     time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trajectory tracker.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=False,
                        default='config/test_settings.yaml')


    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='test')
    trainer = Trainer(settings)
    trainer.perform_testing()


if __name__ == "__main__":
    main()