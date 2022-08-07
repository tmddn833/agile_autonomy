#!/usr/bin/env python3

import argparse
import time
import numpy as np
import rospy
import tensorflow as tf
from PlannerLearning import PlannerAirsim
from common import  MessageHandler
import logging

from config.settings import create_settings



MAX_TIME_EXP = 500  # in second, if it takes more the process needs to be killed
tf.get_logger().setLevel(logging.ERROR)

class Trainer:
    def __init__(self, settings):
        rospy.init_node('iterative_learning_node', anonymous=False)
        self.settings = settings
        np.random.seed(self.settings.random_seed)
        self.msg_handler = MessageHandler()

    def perform_testing(self):
        # Perform rollout just one time!
        self.learner = PlannerAirsim.PlanLearningAirsim(self.settings)
        # Start Experiment
        self.learner.maneuver_complete = False  # Just to be sure
        self.msg_handler.publish_reset()  # /success_reset
        # at this point agile autonomy plan reference trajectory
        # publish True "start flying" topic
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