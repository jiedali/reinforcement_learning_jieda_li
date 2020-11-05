import numpy as np
import matplotlib.pyplot as plt


def reward_function(s, a, ch, cf):
	"""
	Multi-class:

	If action is 0 (not dispatching a shuttle), then reward is Summation (element-wise multiplication of s,ch)
	If action is 1 (dispatching a shuttle), the reward is dependent on current state:
		if current state all 5 class is less than 6, reward is just the cost of dispatching a shuttle
		otherwiise, reward is Sum (cost of dispatching a shuttle + cost of remaining cust)

	:param s: current state represented by vector of length 5
	:param a: current action: 0 or 1
	:param ch: cost per remaining customer, vector of length 5
	:param cf: cost of dispatching a shuttle, a scalar
	:return: reward
	"""
	# if we take action 0 (not dispatching a shuttle)
	if a == 0:
		reward = -np.sum(np.multiply(s, ch))
	# if we take action 1 (dispatching a shuttle)
	else:
		# current state all 5 class has less than 6 customer waiting, reward is just the cost of dispatching a shuttle
		if np.max(s) <= 6:
			reward = -cf
		# current state there is at least one class of customer that has more than 6 people waiting
		# the reward is then sum(cost of dispatching a shuttle + cost of remaining customer)
		else:
			# first get next state
			next_state = s - np.array([6, 6, 6, 6, 6])
			next_state[next_state < 0] = 0
			# then get rewards
			reward = -cf - np.sum(np.multiply(next_state, ch))

	return reward


def get_next_state(s, a, arriving_customer,max_cust_waiting):
	"""
	At each time step,
		if a=0 (not dispatching), next state = previous_state + arrived customers
		if a=1 (dispatching), next state = previous_state + arrived customers - number removed by shuttle
			if previous_state + arrived customers >=15, number removed = 15
			elif previous_state + arrived customers<15, number removed < 15

	:param s: current state, vector of length 6
	:param a: action taken
	:return: next_state, vector of length 6
	"""
	# if we don't dispatch a shuttle
	if a == 0:
		next_state = s + arriving_customer
	else:
		# if we dispatch a shuttle
		next_state = s + arriving_customer - np.array([6, 6, 6, 6, 6])

	# if there is any element in next_state that is negative (meaning the total customer waiting is less than 6),
	# we will just make it 0
	next_state[next_state < 0] = 0
	# if next state is more than maximum customer limit, set it to the maximum
	next_state[next_state > max_cust_waiting] = max_cust_waiting

	# print("Show next state: %d" % next_state)
	return next_state


def expected_future_rewards_per_action(iter, s, a, Vtplus1, cf, ch, gamma, max_cust_waiting):
	"""
	This function computes the expected total rewards per specified action
	(at each time point, arrived customer number follow a distribution, uniform [1,5])

	In the multi-class case, we replace expectation over all next states, by random sample next state

	:param s: current state; In the multi-class case, this is now a vector of length 5
	:param Vtplus1: expected total rewards from future, matrix: 101*5
	:return: a scalar, expected future rewards per specified action
	"""
	fr_per_action = 0
	# Here we no longer loop through all possible next states
	# instead, we draw a random sample of next state and use that to approximate the expectation over next states

	# draw a random sample of arriving customer, represented by a vector of length 5
	arriving_customer = np.random.randint(1, 6, (5,))

	# "next_state" is a vector of length 5
	next_state = get_next_state(s, a, arriving_customer,max_cust_waiting)
	# print("next state is")
	# print(next_state)

	# in the multi-class case, total_future_reward is still a scalar,

	# This is just to locate next_state in current vector representation of value function
	# The index is done following how state is looped, start from S_1 in range(0,101).. most inner loop s_5
	# index_for_next_state_in_value_vector = next_state[0] * (max_cust_waiting ** 4) + \
	#                                        next_state[1] * (max_cust_waiting ** 3) + \
	#                                        next_state[2] * (max_cust_waiting ** 2) + \
	#                                        next_state[3] * (max_cust_waiting ** 1) + \
	#                                        next_state[4]

	fr_per_action = Vtplus1[iter]
	# discount the expected future reward
	# add the current reward, now we get the total expected future reward, per THIS action
	fr_per_action = reward_function(s, a, ch, cf) + gamma * fr_per_action

	return fr_per_action


def enumeration(T, cf, ch, gamma, max_cust_wating):
	"""

	:param T: maximum time period
	:param cf:the cost of dispatching a shuttle, a scalar
	:param ch:the cost per customer left wating per time period, a vector of length 5
	:param gamma:discount factor of future rewards
	:param max_cust_wating: the maximum number of customer waiting per each class (default to 100)
	:return: V_t: value functions, a vector with length (max_cust_wating + 1) **5
	"""
	# total number of state is now (max_cust_wating + 1) **5 = 101**5 = 10,510,100,501
	V_t = np.zeros((max_cust_wating + 1) ** 5)
	V_tplus1 = np.zeros((max_cust_wating + 1) ** 5)
	for t in range(T, -1, -1):
		# for all possible states; Now we need to loop through possible state for each type of customer;
		# Given max_cust_wating=100, this is a loop of 10,510,100,501 iteration
		# s_1 refers to the state of customer type 1;
		# use a counter to count the iteration/ also to index the value function vector
		iter = 0
		for s_1 in range(0, max_cust_wating + 1):
			# s_2 refers to the state of customer type 2;
			for s_2 in range(0, max_cust_wating + 1):
				for s_3 in range(0, max_cust_wating + 1):
					for s_4 in range(0, max_cust_wating + 1):
						for s_5 in range(0, max_cust_wating + 1):
							# Now each unique state will be represented by a vector of length 5
							s = np.array([s_1, s_2, s_3, s_4, s_5])
							max_future_rewards = float('-inf')
							# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
							for a in [0, 1]:
								# compute the expected future rewards per action (expectation over next states is
								# replaced by sampling)
								fr_per_action = expected_future_rewards_per_action(iter, s, a, V_tplus1, cf, ch, gamma, max_cust_wating)
								# print("action: %d" % a)
								# print("future reward per action %d" % fr_per_action)
								# get max future rewards
								max_future_rewards = max(max_future_rewards, fr_per_action)
							V_t[iter] = max_future_rewards
							# counter plus 1 iteration
							iter += 1

		V_tplus1 = V_t.copy()

		return V_t


def value_iteration(cf, ch, gamma, max_cust_wating, theta):
	# total number states is max_cust_wating+1 = 200+1=201 (plus one state of no customer waiting)
	V_t = np.zeros((max_cust_wating + 1) ** 5)
	V_tplus1 = np.zeros((max_cust_wating + 1) ** 5)
	# keep track of how many iterations
	T=0
	while True:
		# for all possible states (possible number of customers at station is 0 ~ 200)
		delta = 0
		iter = 0
		for s_1 in range(0, max_cust_wating + 1):
			# s_2 refers to the state of customer type 2;
			for s_2 in range(0, max_cust_wating + 1):
				for s_3 in range(0, max_cust_wating + 1):
					for s_4 in range(0, max_cust_wating + 1):
						for s_5 in range(0, max_cust_wating + 1):
							# Now each unique state will be represented by a vector of length 5
							s = np.array([s_1, s_2, s_3, s_4, s_5])
							max_future_rewards = float('-inf')
							# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
							for a in [0, 1]:
								# compute the expected future rewards per action (expectation over next states)
								fr_per_action = expected_future_rewards_per_action(iter, s, a, V_tplus1, cf, ch, gamma, max_cust_wating)
								# print("action: %d" % a)
								# print("future reward per action %d" % fr_per_action)
								# get max future rewards
								max_future_rewards = max(max_future_rewards, fr_per_action)
							V_t[iter] = max_future_rewards
							# compute delta across all states, take the maximum delta across all states
							delta = max(delta, np.abs(max_future_rewards - V_tplus1[iter]))
							# print("value of delta %f" % delta)
							iter += 1
						# copy V_t to V_tplus1, to be used for next iteration
		V_tplus1 = V_t.copy()
		T+=1
		print("current value of delta: %f" % delta)
		# print("current iter value: %d" % iter)
		# print("Moved to next iteration: T %d" % T)
		if delta < theta:
			break

	return V_t, T

def policy_eval_function(policy, cf, ch, gamma, max_cust_wating, theta):
	"""
	This function evaluates a policy value iteratively

	:param policy: n_states * n_action matrix, the policy to be evaluated
	:param cf:the cost of dispatching a shuttle
	:param ch:the cost per customer left wating per time period
	:param gamma:discount factor of future rewards
	:param max_cust_wating: maximum number of customer waiting (total states = max_cust_wating+1)
	:param theta: threshold for convergence 0.00001
	:return: V: value of the policy
	"""
	# total number states is (max_cust_wating + 1) ** 5
	V =np.zeros((max_cust_wating + 1) ** 5)
	# initialize a policy (shape of the policy is num_states * num_actions)
	# keep track of how many iterations
	while True:
		iter = 0
		# for all possible states
		delta = 0
		for s_1 in range(0, max_cust_wating + 1):
			# s_2 refers to the state of customer type 2;
			for s_2 in range(0, max_cust_wating + 1):
				for s_3 in range(0, max_cust_wating + 1):
					for s_4 in range(0, max_cust_wating + 1):
						for s_5 in range(0, max_cust_wating + 1):
							# expected_future_rewards represents expected total rewards (over action/next state) under given state
							expected_future_rewards = 0
							s = np.array([s_1, s_2, s_3, s_4, s_5])
							# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
							for a in [0, 1]:
								# compute the expected future rewards per action (expectation over next states)
								fr_per_action = expected_future_rewards_per_action(iter, s, a, V, cf, ch, gamma)
								# get expectation over actions (probability per action given by policy)
								expected_future_rewards += policy[s][a] * fr_per_action
							# Note: this gives the expected future rewards over all possibble actions under given policy and all
							# posible next states
							# compute delta across all states
							delta = max(delta, np.abs(expected_future_rewards - V[iter]))
							#
							V[iter] = expected_future_rewards
						# print("value of delta %f" % delta)
		iter += 1
		if delta < theta:
			break
	return V


def policy_improvement(cf, ch, gamma, max_cust_wating, theta):

	def expected_future_rewards_per_state(s, V, policy):

		# fr_per_state is a vector of length(number_of_possible_actions)
		fr_per_state = np.zeros(policy[s].shape)
		# loop through all possible future states, get expected future rewards over possible next states
		# for all possible actions [0,1]
		for a in [0, 1]:
			fr_per_action = 0
			for arriving_customer in range(1, 6):
				# First get next state
				next_state = get_next_state(s, a, arriving_customer)
				# compute total expected future rewards (expectation over next states, over actions)
				# policy[s][a] is the probability of choosing action a in state s
				# here V is based on incumbent policy
				fr_per_action += (1 / 5) * (reward_function(s, a, ch, cf) + gamma * V[next_state])
			# fr_per_action += policy[s][a] * (1 / 5) * V[next_state]
			# # discount the expected future reward
			# # add the current reward, now we get the total expected future reward, per THIS action
			# fr_per_action = reward_function(s, a, ch, cf) + gamma * fr_per_action
			# update the value for a particular action
			fr_per_state[a] = fr_per_action
		# returns a vector of values per state (each element being the value per action, per that state)
		return fr_per_state

	# ============Policy improvement main loop========
	# Multi-class situation, the policy matrix shape is 10,510,100,501 * 2
	policy = np.ones(((max_cust_wating + 1)**5, 2))
	policy[:,0]=0
	#
	iter = 0
	while True:
		print("Iteration: %d" % iter)
		# evaluate current policy, V is vector with length - number of possible states
		V = policy_eval_function(policy, cf, ch, gamma, max_cust_wating, theta)

		# used to indicate if policy is converged
		policy_stable = True

		# for each state
		for s in range(0, (max_cust_wating + 1)**5):
			# current best action
			chosen_action = np.argmax(policy[s])

			# find the best action with respect to V_pi
			action_values = expected_future_rewards_per_state(s, V, policy)
			best_action = np.argmax(action_values)
			# print("action_values, best_action, chosen action")
			# print(action_values)
			# print(best_action)
			# print(chosen_action)
			# greedily update the policy
			if chosen_action != best_action:
				policy_stable = False
				policy[s] = np.eye(2)[best_action]
		iter += 1

		if policy_stable == True:
			print("The algorithm has converged.")
			return policy, V

if __name__ == "__main__":
	# ============2 (a) Enumeration ==============
	max_cust_waiting = 10
	V_t= enumeration(T=100, cf=100, ch=np.array([1, 1.5, 2, 1.5, 3]), gamma=0.95, max_cust_wating=max_cust_waiting)

	# compute step size, in order to locate the values for each class of customer in value vector
	step_size_ch1 = (max_cust_waiting+1)**4
	step_size_ch1p5 = (max_cust_waiting + 1) ** 3
	step_size_ch2 = (max_cust_waiting + 1) ** 2
	step_size_ch2p5 = (max_cust_waiting + 1) ** 1
	#
	V_t_ch1 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch1, step_size_ch1)]
	V_t_ch1p5 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch1p5, step_size_ch1p5)]
	V_t_ch2 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch2, step_size_ch2)]
	V_t_ch2p5 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch2p5, step_size_ch2p5)]
	plt.plot(range(0, max_cust_waiting+1),V_t_ch1, 'b', label='T=100, max_cust_waiting=20, for ch=1')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch1p5, 'r',label='T=100, max_cust_waiting=20, for ch=1p5')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch2, 'g',label='T=100, max_cust_waiting=20, for ch=2')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch2p5, 'y',label='T=100, max_cust_waiting=20, for ch=2p5')
	plt.plot(range(0, max_cust_waiting + 1), V_t[0:max_cust_waiting + 1], 'm',label='T=100, max_cust_waiting=20, for ch=3')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/2a_value_function_debugrun.png')

	# ============2(b) Value iteration ==============

	max_cust_waiting = 10
	V_t, T = value_iteration(cf=100, ch=np.array([1, 1.5, 2, 1.5, 3]), gamma=0.95, max_cust_wating=max_cust_waiting,theta=0.1)

	# compute step size, in order to locate the values for each class of customer in value vector
	step_size_ch1 = (max_cust_waiting+1)**4
	step_size_ch1p5 = (max_cust_waiting + 1) ** 3
	step_size_ch2 = (max_cust_waiting + 1) ** 2
	step_size_ch2p5 = (max_cust_waiting + 1) ** 1
	#
	V_t_ch1 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch1, step_size_ch1)]
	V_t_ch1p5 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch1p5, step_size_ch1p5)]
	V_t_ch2 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch2, step_size_ch2)]
	V_t_ch2p5 = [V_t[i] for i in range(0, (max_cust_waiting+1)*step_size_ch2p5, step_size_ch2p5)]
	plt.plot(range(0, max_cust_waiting+1),V_t_ch1, 'b', label='max_cust_waiting=20, for ch=1')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch1p5, 'r',label='max_cust_waiting=20, for ch=1p5')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch2, 'g',label='max_cust_waiting=20, for ch=2')
	plt.plot(range(0, max_cust_waiting + 1), V_t_ch2p5, 'y',label='max_cust_waiting=20, for ch=2p5')
	plt.plot(range(0, max_cust_waiting + 1), V_t[0:max_cust_waiting + 1], 'm',label='max_cust_waiting=20, for ch=3')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/2b_value_iteration.png')

	# =========== Question 2(c): plot optimum policy from policy iteration
	policy_improved, V = policy_improvement(cf=100, ch=np.array([1, 1.5, 2, 1.5, 3]), gamma=0.95, max_cust_wating=10, theta=0.1)
	# ======================
	# transform policy
	transformed_policy = policy_improved.argmax(axis=1)
	plt.plot(range(0,max_cust_waiting + 1),transformed_policy,label='Greedy Policy for Ch = 3')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/2c_greedy_policy.png')



