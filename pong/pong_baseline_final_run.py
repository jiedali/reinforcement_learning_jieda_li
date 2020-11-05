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
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# ===================
## load environment and define some parameters
env = gym.make('Pong-v0')

# environment parameters
state_size = [80, 80, 1]
# Jieda: here we only need to choose between action [RIGHT, LEFT]
# original action space has 6 actions
# action_size = env.action_space.n
action_size = 2
# action_size =6
possible_actions = np.identity(action_size, dtype=int).tolist()

# training hyperparameters
learning_rate = 0.0001
num_epochs = 6000
batch_size = 1000  # each 1 is a timestep (not an episode)


# =================


class PolicyNetwork(object):

	def __init__(self, state_size, action_size, learning_rate, name='PolicyNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			with tf.name_scope("inputs"):
				# Jieda note: state is image, size [210, 160, 3]
				self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
				# Jieda note: we are using sparse_softmax_cross_entropy_with_logits, so action is now a scaler (
				# instead of a vector)
				self.actions = tf.placeholder(tf.int32, [None, ], name="actions")
				self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ],
				                                                 name="discounted_episode_rewards_")
			# Jieda note: place holder for the ValueNetwork esitimated value
			# Jieda note: following line is commented out, because feed the results from dis_sample_total_rewards -
			# value_estimates
			# directly to discounted_episode_rewards
			# self.value_estimate = tf.placeholder(tf.float32, [None,], name="value_estimate_")

			with tf.name_scope("conv1"):
				self.conv1 = tf.layers.conv2d(inputs=self.inputs, filters=32, kernel_size=[8, 8], strides=[4, 4],
				                              padding="VALID",
				                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv1")

				self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm1')

				self.conv1_out = tf.nn.relu(self.conv1_batchnorm, name='conv1_out')

			with tf.name_scope("conv2"):
				self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64,
				                              kernel_size=[4, 4], strides=[2, 2], padding="VALID",
				                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv2")

				self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm2')

				self.conv2_out = tf.nn.relu(self.conv2_batchnorm, name='conv2_out')

			with tf.name_scope("conv3"):
				self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64,
				                              kernel_size=[3, 3], strides=[1, 1], padding="VALID",
				                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv3")

				self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm3')

				self.conv3_out = tf.nn.relu(self.conv3_batchnorm, name='conv3_out')

			with tf.name_scope("flatten"):
				self.flatten = tf.contrib.layers.flatten(self.conv3_out)

			with tf.name_scope("fc1"):
				self.fc1 = tf.layers.dense(inputs=self.flatten,
				                           units=512, activation=tf.nn.relu,
				                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

			with tf.name_scope("logits"):
				self.logits = tf.layers.dense(inputs=self.fc1,
				                              units=action_size,
				                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
				                              activation=None)
			with tf.name_scope("softmax"):
				self.action_distribution = tf.nn.softmax(self.logits)

			with tf.name_scope("loss"):
				self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
				                                                                    labels=self.actions)
				# Jieda noted: for baseline, we subtract the value_estimate_ from discounted_episode_rewards
				self.weighted_negative_likelihoods = tf.multiply(self.cross_entropy, self.discounted_episode_rewards)
				self.loss = tf.reduce_mean(self.weighted_negative_likelihoods)

			with tf.name_scope("train"):
				self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
				self.train_opt = self.optimizer.minimize(self.loss)


class ValueNetwork(object):

	def __init__(self, state_size, learning_rate, name='ValueNetwork'):
		self.state_size = state_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			with tf.name_scope("inputs"):
				# Jieda note: state is image, size [210, 160, 3]
				self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
				self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ],
				                                                 name="discounted_episode_rewards_")
				self.target = tf.placeholder(tf.float32, [None, ], name="target")
			# Jieda note: place holder for the ValueNetwork esitimated value
			# Jieda note: following line is commented out, because feed the results from dis_sample_total_rewards -
			# value_estimates
			# directly to discounted_episode_rewards
			# self.value_estimate = tf.placeholder(tf.float32, [None,], name="value_estimate_")

			with tf.name_scope("conv1"):
				self.conv1 = tf.layers.conv2d(inputs=self.inputs, filters=32, kernel_size=[8, 8], strides=[4, 4],
				                              padding="VALID",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv1")

				self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm1')

				self.conv1_out = tf.nn.relu(self.conv1_batchnorm, name='conv1_out')

			with tf.name_scope("conv2"):
				self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64,
				                              kernel_size=[4, 4], strides=[2, 2], padding="VALID",
				                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv2")

				self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm2')

				self.conv2_out = tf.nn.relu(self.conv2_batchnorm, name='conv2_out')

			with tf.name_scope("conv3"):
				self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64,
				                              kernel_size=[3, 3], strides=[1, 1], padding="VALID",
				                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				                              name="conv3")

				self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
				                                                     training=True,
				                                                     epsilon=1e-5,
				                                                     name='batch_norm3')

				self.conv3_out = tf.nn.relu(self.conv3_batchnorm, name='conv3_out')

			with tf.name_scope("flatten"):
				self.flatten = tf.contrib.layers.flatten(self.conv3_out)

			with tf.name_scope("fc1"):
				self.fc1 = tf.layers.dense(inputs=self.flatten,
				                           units=512, activation=tf.nn.relu,
				                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

			with tf.name_scope("output"):
				self.output_layer = tf.layers.dense(
					inputs=self.fc1,
					units=1,
					activation=None,
					kernel_initializer=tf.contrib.layers.xavier_initializer(), name="output")

			with tf.name_scope("loss"):
				self.value_estimate = tf.squeeze(self.output_layer)
				self.loss = tf.squared_difference(self.value_estimate, self.target)

			with tf.name_scope("train"):
				self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
				self.train_opt = self.optimizer.minimize(self.loss)

	# method to update the Value neuron network parameters
	def update(self, state, target, sess=None):
		sess = sess or tf.get_default_session()
		feed_dict = {self.inputs: state, self.target: target}
		_, loss = sess.run([self.train_opt, self.loss], feed_dict)

		return loss


# ===================
## Helper function
### calculates the discounted total rewards from current step onward

def discount_rewards(r, gamma=0.99, normalization=False):
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


# the is the pre-processing function for pre-processing the image (which is our state) from 210x160x3 into 6400 (
# 80*80) 2D float array
def preprocess(image):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
	image = image[35:195]  # crop
	image = image[::2, ::2, 0]  # downsample by factor of 2
	image[image == 144] = 0  # erase background (background type 1)
	image[image == 109] = 0  # erase background (background type 2)
	image[image != 0] = 1  # everything else (paddles, ball) just set to 1

	return np.reshape(image.astype(np.float).ravel(), [80, 80])


# ============================
# run policy
# ============================
def make_batch(seed, batch_size):
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
	env.seed(seed)
	state = env.reset()
	state = preprocess(state)
	state = state.reshape(80, 80, 1)

	while True:
		# run state through policy and calculate action
		action_probability_distribution = sess.run(PolicyNetwork.action_distribution,
		                                           feed_dict={PolicyNetwork.inputs: state.reshape(1, 80, 80, 1)})
		action = np.random.choice([0, 1], p=action_probability_distribution.ravel())
		#         print("debug, action probability distribution")
		#         print(action_probability_distribution)

		#         print("debug:action_probability_distribution.ravel()[2:4]")
		#         print(action_probability_distribution.ravel()[2:4])
		# normalize the probability to sum to 1
		#         p_normalized = action_probability_distribution.ravel()[2:4]/sum(action_probability_distribution.ravel()[2:4])
		#         print("debug: normalized probability")
		#         print(p_normalized)
		# Note: if we don't normalize, probability for action 2 and 3 don't sum to 1 (because there are 6 possibility)
		# choose action
		# only choose between action 2 and 3 ('RIGHT' or 'LEFT')

		# map the action 0 to 2, 1 to 3 (2 in env definition is RIGHT, 3 is LEFT)
		# this mapping is ONLY needed to perform the action in environment
		# when we feed the action back to the PolicyNetwork, we will keep the action as 0 or 1
		if action == 0:
			action_env_compatible = 2
		elif action == 1:
			action_env_compatible = 3

		# perform action
		next_state, reward, done, info = env.step(action_env_compatible)
		# process next_state to shape (80,80)
		next_state = preprocess(next_state)
		next_state = next_state.reshape(80, 80, 1)

		# the 3 lists here tracks the quantity for each episode
		# the other 2 lists (rewards_of_batch and discounted_rewards, they are both list of list)
		# they contain num_episode list and each list is the rewards of that episode of each time stamp
		# rewards_of_batch is UNdiscounted, discounted_rewards is DISCOUNTED and counts from time t onwards
		# in the calculation of loss function, we need the discounted_rewards
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
			state = preprocess(state)
			state = state.reshape(80, 80, 1)

		else:
			# if not done, the next_state become the current state
			state = next_state

	# Essentially, we will keep records of (state, action, rewards), need those for training!
	return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(
		discounted_rewards), episode_num


if __name__ == "__main__":
	# reset the graph
	tf.reset_default_graph()
	# ==============================
	#  Initialize network and session
	# ==============================
	# Instantiate the PolicyNetwork
	PolicyNetwork = PolicyNetwork(state_size, action_size, learning_rate)
	ValueNetwork = ValueNetwork(state_size, learning_rate)
	# Initialize session
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	# ===================
	# Training and Printing some stats
	# ===================
	allRewards = []
	total_rewards = 0
	maximumRewardRecorded = 0
	mean_reward_total = []
	num_epochs = 10
	average_reward = []
	training = True
	epoch = 1
	seed = 666

	# saver
	saver = tf.train.Saver()

	if training:
		while epoch < num_epochs + 1:
			states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(seed, 1000)
			# total rewards of the batch
			total_reward_of_that_batch = np.sum(rewards_of_batch)
			allRewards.append(total_reward_of_that_batch)

			# calculate the mean reward of the batch
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
			mean_reward_total.append(mean_reward_of_that_batch)

			# calculate the average reward of all training
			average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

			# maximum reward recorded
			maximumRewardRecorded = np.amax(allRewards)

			print("===============================")
			print("Epoch: ", epoch, "/", num_epochs)
			print("Number of training episodes: {}".format(nb_episodes_mb))
			print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
			print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
			print("Average Reward of all training: {}".format(average_reward_of_all_training))
			print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

			# Jieda note: update the ValueNetwork parameter, and make an estimate of Value function for each given St
			feed_dict_value = {ValueNetwork.inputs: states_mb, ValueNetwork.target: discounted_rewards_mb}
			_, loss = sess.run([ValueNetwork.train_opt, ValueNetwork.loss], feed_dict_value)

			# Now make a prediction of value function using the updated ValueNetwork parameters
			value_prediction = sess.run(ValueNetwork.value_estimate, {ValueNetwork.inputs: states_mb})

			# Jieda note: compute the discounted total rewards minus baseline
			# shape of discounted_rewards_mb should be an array of length (number of 4 tuples in that batch)
			# shape of value_prediction should also be an array of length (number of 4 tuples in that batch)
			# Note: 4 tuples are essentially samples (st,at,rt,st+1)
			discounted_rewards_mb_minus_baseline = discounted_rewards_mb - value_prediction

			# feedforward, gradient and backprop
			loss_, _ = sess.run([PolicyNetwork.loss, PolicyNetwork.train_opt],
			                    feed_dict={PolicyNetwork.inputs: states_mb, PolicyNetwork.actions: actions_mb,
			                               PolicyNetwork.discounted_episode_rewards: discounted_rewards_mb_minus_baseline})

			# update epoch
			epoch += 1

			# Every 100 epoch, we write the mean_reward_total to a text file (over-write)
			if (epoch % 2) == 0:
				with open('mean_batch_reward_for_each_epoch.txt', 'w') as file:
					file.write('%s\n' % mean_reward_total)

			if (epoch % 5) == 0:
				plt.plot(range(0, len(mean_reward_total)), mean_reward_total)
				plt.savefig('./pong_basline_best_params_epoch' + str(epoch) + '.png')

	# plot the average episode reward vs epoch
	# episode reward is the total undiscounted reward for an episode
	plt.plot(range(0, len(mean_reward_total)), mean_reward_total)
	plt.savefig('./pong_basline_best_params_final.png')
