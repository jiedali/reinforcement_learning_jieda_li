import numpy as np
import matplotlib.pyplot as plt


def reward_function(s, a, ch, cf):
	"""
	If action is 0 (not dispatching a shuttle), then reward is the cost per customer * num_cust remained
	If action is 1 (dispatching a shuttle), then reward is dependent on current state (number of customers),
	if current number of customer greater than 15, reward = cost of dispatching a shuttle +cost of remaining customer
	if current number of customer smaller than 15, reward = cost of dispatching a shuttle only
	:param s: current state
	:param a: current action
	:param ch: cost per remaining customer
	:param cf: cost of dispatching a shuttle
	:return: reward
	"""
	if a == 0:
		reward = -s * ch
	elif a == 1:
		if s >= 15:
			reward = -cf - (s - 15) * ch
		elif s < 15:
			reward = -cf
	return reward


def get_next_state(s, a, arriving_customer):
	"""
	At each time step,
		if a=0 (not dispatching), next state = previous_state + arrived customers
		if a=1 (dispatching), next state = previous_state + arrived customers - number removed by shuttle
			if previous_state + arrived customers >=15, number removed = 15
			elif previous_state + arrived customers<15, number removed < 15

	:param s: current state
	:param a: action taken
	:return: next_state value
	"""
	if a == 0:
		next_state = s + arriving_customer
	else:
		if s + arriving_customer >= 15:
			next_state = s + arriving_customer - 15
		else:
			next_state = 0

	# if number of customers exceed 200, bring it back to 200
	if next_state > 200:
		next_state = 200
	else:
		next_state = next_state

	# print("Show next state: %d" % next_state)

	return next_state


def expected_future_rewards_per_action(s, a, Vtplus1, cf, ch, gamma):
	"""
	This function computes the expected total rewards per specific action
	Expectation over all **possible future states**
	The stochasticity arises from next state being random;
	(at each time point, arrived customer number follow a distribution, uniform [1,5])
	:param s: current state
	:param Vtplus1: expected total rewards from future, vector of length (max number of states)
	:return: a scalar, expecred future rewards per this action,
	"""

	fr_per_action = 0
	# loop through all possible future states, get expected future rewards over possible next states
	for arriving_customer in range(1, 6):
		# In each time step,
		next_state = get_next_state(s, a, arriving_customer)
		fr_per_action += (1 / 5) * Vtplus1[next_state]

	# discount the expected future reward
	# add the current reward, now we get the total expected future reward, per THIS action
	fr_per_action = reward_function(s, a, ch, cf) + gamma * fr_per_action
	return fr_per_action


def enumeration(T, cf, ch, gamma, max_cust_wating):
	"""

	:param T: maximum time period
	:param cf:the cost of dispatching a shuttle
	:param ch:the cost per customer left wating per time period
	:param gamma:discount factor of future rewards
	:param max_cust_wating: the maximum number of customer wating at the station (total number of states)
	:return: V_t: value functions, a vector of length (number of states)
	"""

	# total number states is max_cust_wating+1 = 200+1=201 (plus one state of no customer waiting)
	V_t = np.zeros(max_cust_wating + 1)
	V_tplus1 = np.zeros(max_cust_wating + 1)
	for t in range(T, -1, -1):
		# for all possible states (possible number of customers at station is 0 ~ 200)
		for s in range(0, max_cust_wating + 1):
			max_future_rewards = float('-inf')
			# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
			for a in [0, 1]:
				# compute the expected future rewards per action (expectation over next states)
				fr_per_action = expected_future_rewards_per_action(s, a, V_tplus1, cf, ch, gamma)
				print("action: %d" % a)
				print("future reward per action %d" % fr_per_action)
				# get max future rewards
				max_future_rewards = max(max_future_rewards, fr_per_action)
			V_t[s] = max_future_rewards
		V_tplus1 = V_t.copy()

	return V_t


def value_iteration(cf, ch, gamma, max_cust_wating, theta):
	# total number states is max_cust_wating+1 = 200+1=201 (plus one state of no customer waiting)
	V_t = np.zeros(max_cust_wating + 1)
	V_tplus1 = np.zeros(max_cust_wating + 1)
	# keep track of how many iterations
	iter = 0
	while True:
		# for all possible states (possible number of customers at station is 0 ~ 200)
		delta = 0
		for s in range(0, max_cust_wating + 1):
			max_future_rewards = float('-inf')
			# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
			for a in [0, 1]:
				# compute the expected future rewards per action (expectation over next states)
				fr_per_action = expected_future_rewards_per_action(s, a, V_tplus1, cf, ch, gamma)
				# print("action: %d" % a)
				# print("future reward per action %d" %fr_per_action)
				# get max future rewards
				max_future_rewards = max(max_future_rewards, fr_per_action)
			V_t[s] = max_future_rewards
			# compute delta across all states, take the maximum delta across all states
			delta = max(delta, np.abs(max_future_rewards - V_tplus1[s]))
			print("value of delta %f" % delta)

		iter += 1
		# copy V_t to V_tplus1, to be used for next iteration
		V_tplus1 = V_t.copy()
		if delta < theta:
			break

	return V_t, iter


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
	# total number states is max_cust_wating+1 = 200+1=201 (plus one state of no customer waiting)
	V = np.zeros(max_cust_wating + 1)
	# initialize a policy (shape of the policy is num_states * num_actions)
	# keep track of how many iterations
	iter = 0
	while True:
		# for all possible states (possible number of customers at station is 0 ~ 200)
		delta = 0
		for s in range(0, max_cust_wating + 1):
			# expected_future_rewards represents expected total rewards (over action/next state) under given state
			expected_future_rewards = 0
			# for all possible actions (only two, 1, dispatch a shuttle; 0, not dispatch a shuttle)
			for a in [0, 1]:
				# compute the expected future rewards per action (expectation over next states)
				fr_per_action = expected_future_rewards_per_action(s, a, V, cf, ch, gamma)
				# get expectation over actions (probability per action given by policy)
				expected_future_rewards += policy[s][a] * fr_per_action
			# Note: this gives the expected future rewards over all possibble actions under given policy and all
			# posible next states
			# compute delta across all states
			delta = max(delta, np.abs(expected_future_rewards - V[s]))
			#
			V[s] = expected_future_rewards

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
			# 	fr_per_action += policy[s][a] * (1 / 5) * V[next_state]
			# # discount the expected future reward
			# # add the current reward, now we get the total expected future reward, per THIS action
			# fr_per_action = reward_function(s, a, ch, cf) + gamma * fr_per_action
			# update the value for a particular action
			fr_per_state[a] = fr_per_action
		# returns a vector of values per state (each element being the value per action, per that state)
		return fr_per_state

	# ============Policy improvement main loop========
	# start with a policy that always choose action 1
	# policy = np.ones([max_cust_wating + 1, 2]) / 2
	policy = np.ones((max_cust_wating + 1, 2))
	policy[:,0]=0
	#
	iter = 0
	while True:
		print("Iteration: %d" % iter)
		# evaluate current policy, V is vector with length - number of possible states
		V = policy_eval_function(policy, cf, ch, gamma, max_cust_wating, theta)

		# used to indicate if policy is converged
		policy_stable = True

		print("I am here!")
		# for each state
		for s in range(0, max_cust_wating + 1):
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
			print("Second I am here!")
			return policy, V


if __name__ == "__main__":

	# =================
	# Question 1a: plot value function from enumeration and value iteration
	# V_t_enum_10 = enumeration(T=10, cf=100, ch=2, gamma=0.95, max_cust_wating=200)
	# V_t_enum_50 = enumeration(T=50, cf=100, ch=2, gamma=0.95, max_cust_wating=200)
	V_t_enum_500 = enumeration(T=500, cf=100, ch=2, gamma=0.95, max_cust_wating=200)
	plt.plot(range(0,201), V_t_enum_500, 'g', label='Value function at Time 0 from enumeration (T=500)')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/1a_valuefunction_enumeraion_time0.png')

	## ============ 1(b)
	# plot optimum value function from value iteration
	V_t, iter = value_iteration(cf=100, ch=2, gamma=0.95, max_cust_wating=200, theta=0.00001)
	plt.plot(range(0, 201), V_t, 'b', label='optimum value function from value iteration')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/1b_valuefunction_value_iteration.png')

	## ==================
	# Question 1(c): plot optimum policy from policy iteration
	policy_improved, V = policy_improvement(cf=100, ch=2, gamma=0.95, max_cust_wating=200, theta=0.00001)
	# V_greedy = policy_eval_function(policy=policy_improved, cf=100, ch=2, gamma=0.95, max_cust_wating=200, theta=0.00001)
	# plt.plot(range(0,201),V_greedy,'g', label='greedy policy')
	# plt.legend()
	# plt.savefig('./shuttle_dispatch/plots/valua_functions_greedy_vs_valueiter.png')
	# ======================
	# transform policy
	transformed_policy = policy_improved.argmax(axis=1)
	plt.plot(range(0,201),transformed_policy,label='Greedy Policy')
	plt.legend()
	plt.savefig('./shuttle_dispatch/plots/1c_greedy_policy.png')
