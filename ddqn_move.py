# Contains code from from: https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py
import random
import numpy as np
import pandas as pd
import os
from absl import app, logging
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
import pickle
from pysc2.env import sc2_env, run_loop, environment

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from networks import DQN, Transition, ReplayMemory
from main_agent import Hyperparams
	
class DDQNMoveAgent(base_agent.BaseAgent):
	
	def __init__(self, H:Hyperparams=None, *args, **kwargs):
		super(DDQNMoveAgent, self).__init__()
		self.H = H or Hyperparams(tau=0.5)

		# TODO: Do not hardcode following variables
		self.name = kwargs.get('name', "Dual_DQN_Move")
		self.description = kwargs.get('description', "")
		self.execution_id = datetime.strftime(
			datetime.now(),
			"%y%m%d%H%M%S"
		)
		self.execution_folder = os.path.join(
			"./saves",
			self.name,
			self.execution_id
		)

		logging.info(f"Saving folder: {self.execution_folder}")

		n_observations = 2 
		n_actions = 64 

		self.n_observations = 2 
		self.n_actions = 64 

		self.steps_done = 0
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logging.info(f"Chosen device is {self.device}")
		self.policy_net_x = DQN(n_observations, n_actions).to(self.device)
		self.target_net_x = DQN(n_observations, n_actions).to(self.device)
		self.target_net_x.load_state_dict(self.policy_net_x.state_dict())
		self.optimizer_x = optim.AdamW(self.policy_net_x.parameters(), lr=self.H.lr, amsgrad=True)
		self.memory_x = ReplayMemory(self.H.memory_len)
		

		self.policy_net_y = DQN(n_observations, n_actions).to(self.device)
		self.target_net_y = DQN(n_observations, n_actions).to(self.device)
		self.target_net_y.load_state_dict(self.policy_net_y.state_dict())
		self.optimizer_y = optim.AdamW(self.policy_net_y.parameters(), lr=self.H.lr, amsgrad=True)
		self.memory_y = ReplayMemory(self.H.memory_len)
		
		self.rewards = []
		self.n_epis_to_mean = 10 # Number of episodes to calculate mean and save
		self.n_epis_force_save = 100
		self.max_mean_n_rewards = -math.inf
		self.n_eps = 0

		self.marine = None
		self.beacon = None
		self.new_game()

	def reset(self):
		super(DDQNMoveAgent, self).reset()
		self.new_game()

	def new_game(self):
		self.base_top_left = None
		self.previous_state = None
		self.previous_action = None
		self.n_steps = 0
		self.prev_score = 0
		self.marine = None
		self.beacon = None
		self.dist = 0
	
	def get_state(self):
		x = (self.beacon.x, self.marine.x)
		y = (self.beacon.y, self.marine.y)
		s_x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
		s_y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
		return s_x, s_y

	def get_dist(self):
		return abs(self.marine.x - self.beacon.x), abs(self.marine.y - self.beacon.y)

	def get_reward(self, obs):
		dist = self.get_dist()
		res = self.dist[0] - dist[0], self.dist[1] - dist[1]
		print(f"Prev, Curr, rew: {self.dist, dist, res}")
		self.dist = dist
		if obs.reward == 1:
			res = 1000, 1000
		return res
	
	def get_episode_reward(self,obs):
		return obs.observation['score_cumulative']['score']

	def select_action(self, state):
		sample = random.random()
		eps_threshold = self.H.eps_end + (self.H.eps_start - self.H.eps_end) * \
				math.exp(-1. * self.steps_done / self.H.eps_decay)
		self.steps_done += 1
		if sample > eps_threshold:
				with torch.no_grad():
						# t.max(1) will return the largest column value of each row.
						# second column on max result is index of where max element was
						# found, so we pick action with the larger expected reward.
						return (
							self.policy_net_x(state[0]).max(1).indices.view(1, 1), 
							self.policy_net_y(state[1]).max(1).indices.view(1, 1))
		else:
				action_x = random.randint(0,self.n_actions-1)
				action_y = random.randint(0,self.n_actions-1)
				return (
					torch.tensor([[action_x]], device=self.device, dtype=torch.long), 
					torch.tensor([[action_y]], device=self.device, dtype=torch.long))

	def optimize_model_x(self):
		if len(self.memory_x) < self.H.batch_size:
				return
		transitions = self.memory_x.sample(self.H.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
																					batch.next_state)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
																								if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = self.policy_net_x(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1).values
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.H.batch_size, device=self.device)
		with torch.no_grad():
				next_state_values[non_final_mask] = self.target_net_x(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.H.gamma) + reward_batch

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer_x.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.policy_net_x.parameters(), 100)
		self.optimizer_x.step()

	def optimize_model_y(self):
		if len(self.memory_y) < self.H.batch_size:
				return
		transitions = self.memory_y.sample(self.H.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
																					batch.next_state)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
																								if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = self.policy_net_y(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1).values
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.H.batch_size, device=self.device)
		with torch.no_grad():
				next_state_values[non_final_mask] = self.target_net_y(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.H.gamma) + reward_batch

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer_y.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.policy_net_y.parameters(), 100)
		self.optimizer_y.step()

	def update_networks(self):
		# Update target_network form policy net current params
		target_net_state_dict = self.target_net_x.state_dict()
		policy_net_state_dict = self.policy_net_x.state_dict()
		for k in policy_net_state_dict:
			target_net_state_dict[k] = policy_net_state_dict[k] * self.H.tau + target_net_state_dict[k] * (1-self.H.tau)
		self.target_net_x.load_state_dict(target_net_state_dict)

		target_net_state_dict = self.target_net_y.state_dict()
		policy_net_state_dict = self.policy_net_y.state_dict()
		for k in policy_net_state_dict:
			target_net_state_dict[k] = policy_net_state_dict[k] * self.H.tau + target_net_state_dict[k] * (1-self.H.tau)
		self.target_net_y.load_state_dict(target_net_state_dict)

	def save(self, folder):
		path = os.path.join(self.execution_folder, folder)
		if not os.path.isdir(path):
			os.makedirs(path)
		policy = os.path.join(path, "policy_x.pth")
		target = os.path.join(path, "target_x.pth")
		torch.save(self.policy_net_x.state_dict(), policy)
		torch.save(self.target_net_x.state_dict(), target)
		policy = os.path.join(path, "policy_y.pth")
		target = os.path.join(path, "target_y.pth")
		torch.save(self.policy_net_y.state_dict(), policy)
		torch.save(self.target_net_y.state_dict(), target)

	def episode_done(self, obs):
		self.n_eps += 1
		r = self.get_episode_reward(obs)
		self.rewards.append(r)
		to_mean = self.rewards[-self.n_epis_to_mean:]
		mean = sum(to_mean) / self.n_epis_to_mean
		logging.info(f"Eps {self.n_eps} finished with reward {r}, new mean = {mean} in {self.n_steps} steps")
		if (len(to_mean) >= self.n_epis_to_mean and (mean > self.max_mean_n_rewards)) or (self.n_eps%self.n_epis_force_save == 0):
			logging.info("saving")
			self.max_mean_n_rewards = mean
			self.save(f"{self.n_eps}_{mean}")

	def memory_push_and_rotation(self, obs, state, action, reward):

		# reward = torch.tensor([reward], device=self.device)
		if not obs.first():
			# print("\n\n")
			# print(self.previous_state[0])
			# print(self.previous_action[0])
			# print(state[0])
			# print(reward[0])
			# print("\n\n")
			# print(self.previous_state[1])
			# print(self.previous_action[1])
			# print(state[1])
			# print(reward[1])
			# As in first step previous_state and previous action will be None
			self.memory_x.push(self.previous_state[0], self.previous_action[0], state[0], torch.tensor([reward[0]], device=self.device))
			self.memory_y.push(self.previous_state[1], self.previous_action[1], state[1], torch.tensor([reward[1]], device=self.device))
		self.previous_state = state
		self.previous_action = action

	@staticmethod
	def get_marine(obs):
		return [unit for unit in obs.observation.feature_units
				if unit.unit_type == units.Terran.Marine
				and unit.alliance == features.PlayerRelative.SELF][0]
	
	@staticmethod
	def get_beacon(obs):
		units = [u for u in obs.observation.feature_units
				 if u.alliance == features.PlayerRelative.NEUTRAL][0]
		return units

	def step(self, obs):
		super(DDQNMoveAgent, self).step(obs)
		self.n_steps += 1
		self.marine = self.get_marine(obs)
		self.beacon = self.get_beacon(obs)
		if obs.first():
			self.dist = self.get_dist()

		done = obs.last()

		state = self.get_state()
		action = self.select_action(state)
		reward = self.get_reward(obs)
		print(f"Posiciones: {(self.beacon.x, self.beacon.y), (self.marine.x, self.marine.y)}")
		
	
		self.memory_push_and_rotation(obs, state, action, reward)

		self.optimize_model_x()
		self.optimize_model_y()
		self.update_networks()

		if done:
			self.episode_done(obs) # Saves if rewards mean is higher than before

		
		if obs.first():
			return actions.FUNCTIONS.select_army("select")
		move_cords = [action[0].item(), action[1].item()]
		return actions.FUNCTIONS.Move_screen("now", move_cords)
		# return actions.FUNCTIONS.Move_screen("now", (69,14))
		# return actions.FUNCTIONS.no_op()
