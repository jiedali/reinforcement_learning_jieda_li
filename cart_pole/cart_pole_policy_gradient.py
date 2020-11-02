# imports
import os
import time
import numpy as np
import tensorflow as tf
print(tf.__version__)
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
# choose a GPU card
os.environ['CUDA_VISIBLE_DEVICES']="2"
#
from PolicyNetwork import PolicyNetwork


## load environment
env = gym.make('CartPole-v0')

# environment parameters
state_size = 4
action_size = env.action_space.n
possible_actions = np.identity(action_size,dtype=int).tolist()

# training hyperparameters
learning_rate = 0.002
num_epochs = 500
batch_size = 1000 # each 1 is a timestep (not an episode)


## Helper function
### calculates the discounted total rewards from current step onward

def discount_rewards(r, gamma=0.95, normalization=False):
	"""
	computes the discounted rewards from time t onward for a given episode
	length of r corresponds to the number of steps in an episode
	discounted_r has the same length --> discounted rewards at each time step
	"""
	discounted_r = np.zeros_like(r)
	running_add = 0
	for i in reversed(range(0, len(r))):
		running_add = running_add * gamma + r[i]
		discounted_r[i] = running_add

	if normalization:
		mean = np.mean(discounted_r)
		std = np.std(discounted_r)
		discounted_r = (discounted_r - mean) / (std)

	return discounted_r

#==============================
#  Initialize network and session
#==============================
# Instantiate the PolicyNetwork
PolicyNetwork = PolicyNetwork(state_size, action_size, learning_rate)
# Initialize session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#============================
# run policy
#============================
def make_batch(batch_size):

	"""
	We will run the policy and generate a bunch of episodes
	We need to keep track of (st,at,rt,st+1) for each of the episode, those we will use for training
	:param batch_size: number of episodes in a batch
	:return: 3 list: states, actions, rewards of batch (each value is accumulated discounted total rewards)
	"""
	# initialize lists:

	states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
	# number of episode in batch
	episode_num = 1

	# Get a new state
	state = env.reset()

	while True:
		# run state through policy and calculate action
		action_probability_distribution = sess.run(PolicyNetwork.action_distribution, feed_dict={PolicyNetwork.inputs_: state.reshape(1,state_size)})

		# choose action
		action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())

		# perform action
		next_state, reward, done, info = env.step(action)

		# store results
		states.append(state)
		actions.append(action)
		rewards_of_episode.append(reward)

		if done:
			# if the entire episode is done

			# if you have more than 1 episode in a batch, we track rewards of each episode in the batch
			# this is undiscounted rewards, we WILL use this to do the plotting
			rewards_of_batch.append(rewards_of_episode)

			# disc_rwds_per_episode is a list with each element being the discounted reward from a time stamp t onward
			disc_rwds_per_episode = discount_rewards(rewards_of_episode, gamma=0.99, normalization=True)
			# discounted_rewards is a list of lenth (number of episode)
			discounted_rewards.append(disc_rwds_per_episode)
			#
			if len(np.concatenate(rewards_of_batch)) > batch_size:
				break
			# reset the transition stores
			rewards_of_episode = []
			# add episode
			episode_num += 1
			# reset the state
			state = env.reset()

		else:
			# if not done, the next_state become the current state
			state = next_state

	# Essentially, we will keep records of (state, action, rewards), need those for training!
	return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch)


#===================
# Training and Printing some stats
#===================

allRewards =[]

total_rewards =0
maximumRewardRecorded =0
mean_reward_total =[]
epoch =1
average_reward=[]
training = True

# saver
saver = tf.train.Saver()

if training:
	while epoch < num_epochs +1:
		states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(100)
		# total rewards of the batch
		total_reward_of_that_batch = np.sum(rewards_of_batch)
		allRewards.append(total_reward_of_that_batch)

		# calculate the mean reward of the batch
		mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
		mean_reward_total.append(mean_reward_of_that_batch)

		# calculate the average reward of all training
		average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

		# maximum reward recorded
		max_reward_recorded = np.amax(allRewards)

		print("===============================")
		print("Epoch: ", epoch, "/", num_epochs)
		print("Number of training episodes: {}".format(nb_episodes_mb))
		print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
		print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
		print("Average Reward of all training: {}".format(average_reward_of_all_training))
		print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

		# feedforward, gradient and backprop
		loss_,_ = sess.run([PolicyNetwork.loss,PolicyNetwork.train_opt], \
		        feed_dict={PolicyNetwork.inputs_: states_mb, PolicyNetwork.actions: actions_mb, PolicyNetwork.discounted_episode_rewards: discounted_rewards_mb})

