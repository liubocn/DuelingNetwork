# -*- coding: utf-8 -*-
"""
Dueling network implementation with chainer and rlglue
Original program is written by Naoto Yoshida,
modified to double DQN by Takamitsu Omasa
"""

import copy
import argparse

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers, serializers, function, utils
from chainer.utils import type_check
import chainer.functions as F
#import relu_mean
#import bilinear_dn_link
import DN_out

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action


parser = argparse.ArgumentParser(description='DN')
parser.add_argument('--resumemodel', '-r', default=None, help='file name of resuming model')
parser.add_argument('--resumeD1', '-f', default=None, help='file name of resuming Ddata 1')
parser.add_argument('--resumeD2', '-s', default=None, help='file name of resuming Ddata 2')
parser.add_argument('--resumeTimeStep', '-t', default=None, help='Time step at resuming')
args = parser.parse_args()

class DN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 100#10**4  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6

    def __init__(self, enable_controller=[0, 1, 3, 4]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller  # Default setting : "Breakout"

        print "Initializing DN..."
#	Initialization of Chainer 1.1.0 or older.
#        print "CUDA init"
#        cuda.init()

        print "Model Building"
        self.model = FunctionSet(
            l1=F.Convolution2D(4, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 256, wscale=np.sqrt(2)),
            l5=F.Linear(3136, 256, wscale=np.sqrt(2)),
            l6=F.Linear(256, 1, initialW=np.zeros((1, 256), dtype=np.float32)),
            l7=F.Linear(256, self.num_of_actions, initialW=np.zeros((self.num_of_actions, 256),
                                               dtype=np.float32)),
            q_value=DN_out.DN_out(1, self.num_of_actions, self.num_of_actions, nobias = True)
        ).to_gpu()
        
        if args.resumemodel:
            # load saved model
            serializers.load_npz(args.resumemodel, self.model)
            print "load model from resume.model"
        

        self.model_target = copy.deepcopy(self.model)

        print "Initizlizing Optimizer"
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        if args.resumeD1 and args.resumeD2:
            # load saved D1 and D2
            npz_tmp1 = np.load(args.resumeD1)
            print "finished loading half of D data"
            npz_tmp2 = np.load(args.resumeD2)
            self.D = [npz_tmp1['D0'],
                      npz_tmp1['D1'],
                      npz_tmp1['D2'],
                      npz_tmp2['D3'],
                      npz_tmp2['D4']]
            npz_tmp1.close()
            npz_tmp2.close()
            print "loaded stored all D data"
        else:
            self.D = [np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                      np.zeros(self.data_size, dtype=np.uint8),
                      np.zeros((self.data_size, 1), dtype=np.int8),
                      np.zeros((self.data_size, 4, 84, 84), dtype=np.uint8),
                      np.zeros((self.data_size, 1), dtype=np.bool)]
            print "initialized D data"

    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.Q_func(s)  # Get Q-value
        # Generate Target Signals
        tmp2 = self.Q_func(s_dash)
        tmp2 = list(map(np.argmax, tmp2.data.get()))  # argmaxQ(s',a)
        tmp = self.Q_func_target(s_dash)  # Q'(s',*)
        tmp = list(tmp.data.get())
        # select Q'(s',*) due to argmaxQ(s',a)
        res1 = []
        for i in range(num_of_batch):
            res1.append(tmp[i][tmp2[i]])

        #max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        max_Q_dash = np.asanyarray(res1, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)
        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = np.sign(Reward[i]) + self.gamma * max_Q_dash[i]
            else:
                tmp_ = np.sign(Reward[i])

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_
        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, Q

    def stockExperience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
        else:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
            self.D[3][data_index] = state_dash
        self.D[4][data_index] = episode_end_flag

    def experienceReplay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, 4, 84, 84), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.D[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.D[1][replay_index[i]]
                r_replay[i] = self.D[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.D[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.D[4][replay_index[i]]

            s_replay = cuda.to_gpu(s_replay)
            s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def Q_func(self, state):
        print 'now Q_func is implemented'
        h1 = F.relu(self.model.l1(state / 254.0))  # scale inputs in [0.0 1.0]
        h2 = F.relu(self.model.l2(h1))
        h3 = F.relu(self.model.l3(h2))
        h4 = F.relu(self.model.l4(h3)) # left side connected with s value
        h5 = F.relu(self.model.l5(h3)) # right side connected with A value
        h6 = self.model.l6(h4) # s value
        h7 = self.model.l7(h5) # A value
        Q = self.model.q_value(h6, h7) # Q value
        return Q

    def Q_func_target(self, state):
        print 'now Q_func_target is implemented'
        h1 = F.relu(self.model_target.l1(state / 254.0))  # scale inputs in [0.0 1.0]
        h2 = F.relu(self.model_target.l2(h1))
        h3 = F.relu(self.model_target.l3(h2))
        h4 = F.relu(self.model_target.l4(h3)) # left side connected with s value
        h5 = F.relu(self.model_target.l5(h3)) # right side connected with A value
        h6 = self.model_target.l6(h4) # s value
        h7 = self.model_target.l7(h5) # A value
        Q = self.model_target.q_value(h6, h7) # Q value
        return Q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        Q = self.Q_func(s)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print "RANDOM"
        else:
            index_action = np.argmax(Q.get())
            print "GREEDY"
        return self.index_to_action(index_action), Q

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)


class dn_agent(Agent):  # RL-glue Process
    lastAction = Action()
    policyFrozen = False

    def agent_init(self, taskSpec):
        # Some initializations for rlglue
        self.lastAction = Action()

        self.time = 0
        if args.resumeTimeStep:
            self.epsilon = 1.0 - float(args.resumeTimeStep) / 10**6 # Resuming exploration rate
            if self.epsilon < 0.05:
                self.epsilon = 0.05
            str_tmp = "resuming exploration rate is " + str(self.epsilon)
            print str_tmp
        else:
            self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DN from DN_class
        self.DN = DN_class()  # default is for "Pong".

    def agent_start(self, observation):

        # Preprocess
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling

        # Initialize State
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))

        # Generate an Action e-greedy
        returnAction = Action()
        action, Q_now = self.DN.e_greedy(state_, self.epsilon)
        returnAction.intArray = [action]

        # Update for next step
        self.lastAction = copy.deepcopy(returnAction)
        self.last_state = self.state.copy()
        self.last_observation = obs_array

        return returnAction

    def agent_step(self, reward, observation):

        # Preproces
        tmp = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)  # Get Intensity from the observation
        obs_array = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8, :]  # Scaling
        obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_processed], dtype=np.uint8)
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32))

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.DN.initial_exploration < self.time:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d/%d steps" % (self.time, self.DN.initial_exploration)
                eps = 1.0
        else:  # Evaluation
                print "Policy is Frozen"
                eps = 0.05

        # Generate an Action by e-greedy action selection
        returnAction = Action()
        action, Q_now = self.DN.e_greedy(state_, eps)
        returnAction.intArray = [action]

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.state, False)
            self.DN.experienceReplay(self.time)

        # Target model update
        if self.DN.initial_exploration < self.time and np.mod(self.time, self.DN.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.DN.target_model_update()

        # Simple text based visualization
        print ' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.time, self.DN.action_to_index(action), np.sign(reward), eps, np.max(Q_now.get()))
        # Updates for next step
        self.last_observation = obs_array

        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(returnAction)
            self.last_state = self.state.copy()
            self.time += 1

        return returnAction

    def agent_end(self, reward):  # Episode Terminated

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DN.stockExperience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.last_state, True)
            self.DN.experienceReplay(self.time)

        # Target model update
        if self.DN.initial_exploration < self.time and np.mod(self.time, self.DN.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.DN.target_model_update()

        # Simple text based visualization
        print '  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon)

        # Time count
        if self.policyFrozen is False:
            self.time += 1

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        if inMessage.startswith("save model"):
            serializers.save_npz('resume.model', self.DN.model) # save current model
            np.savez('stored_D012.npz', D0=self.DN.D[0], D1=self.DN.D[1], D2=self.DN.D[2])
            np.savez('stored_D34.npz', D3=self.DN.D[3], D4=self.DN.D[4])
            return "message understood, model saved"

if __name__ == "__main__":
    AgentLoader.loadAgent(dn_agent())
