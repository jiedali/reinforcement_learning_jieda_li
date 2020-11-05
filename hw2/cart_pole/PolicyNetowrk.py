import tensorflow as tf

class PolicyNetwork(object):

	def __init__(self, state_size, action_size, learning_rate, name='PolicyNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			with tf.name_scope("inputs"):
				self.inputs = tf.placeholder(tf.float32, [None,state_size], name = "inputs_")
				self.actions = tf.placeholder(tf.int32, [None,action_size], name ="actions")
				self.discounted_episode_rewards = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards_")

			with tf.name_scope("fc1"):
				self.fc1 = tf.layers.dense(inputs=self.inputs,
				                          units = 256, activation = tf.nn.relu,
				                          kernel_initializer = tf.contrib.layers.xavier_initializer(), name = "fc1")

			with tf.name_scope("fc2"):
				self.fc2 = tf.layers.dense(inputs = self.fc1,
				                           units = 256, activation = tf.nn.relu,
				                           kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'fc2')

			with tf.name_scope("logits"):
				self.logits = tf.layers.dense(inputs = self.fc2,
				                              units = action_size,
				                              kernel_initializer = tf.contrib.layers.xavier_initializer(),
				                              activation = None)
			with tf.name_scope("softmax"):
				self.action_distribution = tf.nn.softmax(self.logits)

			with tf.name_scope("loss"):
				self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, lables =self.actions)
				self.weighted_negative_likelihoods = tf.multiply(self.cross_entropy, self.discounted_episode_rewards)
				self.loss = tf.reduce_mean(self.weighted_negative_likelihoods)

			with tf.name_scope("train"):
				self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
				self.train_opt = self.optimizer.minimize(self.loss)



