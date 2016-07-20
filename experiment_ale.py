# -*- coding: utf-8 -*-
"""
Simple RL glue experiment setup
"""

import numpy as np
import rlglue.RLGlue as RLGlue
import csv

max_learningEpisode = 3010

whichEpisode = 0
learningEpisode = 0


def runEpisode(is_learning_episode):
    global whichEpisode, learningEpisode

    RLGlue.RL_episode(0)
    totalSteps = RLGlue.RL_num_steps()
    totalReward = RLGlue.RL_return()

    whichEpisode += 1

    if is_learning_episode:
        learningEpisode += 1
        print "Episode " + str(learningEpisode) + "\t " + str(totalSteps) + " steps \t" + str(totalReward) + " total reward\t "
    else:
        print "Evaluation ::\t " + str(totalSteps) + " steps \t" + str(totalReward) + " total reward\t "

        # write reward in csv file
        list_csv = [str(learningEpisode), str(totalReward)]
        f_csv = open('reward.csv', 'a')
        writer_r = csv.writer(f_csv, lineterminator = '\n')
        writer_r.writerow(list_csv)
        f_csv.close()

# Main Program starts here
print "\n\nDN-ALE Experiment starting up!"
RLGlue.RL_init()

while learningEpisode < max_learningEpisode:
    # Evaluate model every 10 episodes
    if np.mod(whichEpisode, 10) == 0:
        print "Freeze learning for Evaluation"
        RLGlue.RL_agent_message("freeze learning")
        runEpisode(is_learning_episode=False)
    else:
        print "DN is Learning"
        RLGlue.RL_agent_message("unfreeze learning")
        runEpisode(is_learning_episode=True)

    # Save model every 100 learning episodes
    if np.mod(learningEpisode, 500) == 0 and learningEpisode != 0:
        print "SAVE CURRENT MODEL"
        RLGlue.RL_agent_message("save model")

RLGlue.RL_cleanup()

print "Experiment COMPLETED @ Episode ", whichEpisode
