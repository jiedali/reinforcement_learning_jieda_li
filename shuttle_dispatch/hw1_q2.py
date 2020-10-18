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
	V_t = np.zeros((max_cust_wating + 1)**5)
	V_tplus1 = np.zeros((max_cust_wating + 1)**5)
	for t in range(T, -1, -1):
		# for all possible states (possible number of customers at station is 0 ~ 200)
		for s in range(0, (max_cust_wating + 1)**5):
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
