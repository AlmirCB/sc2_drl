# Contains code from https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py
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

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

	def __init__(self, n_observations, n_actions):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observations, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, n_actions)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)


class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class Hyperparams():
		def __init__(self, 
							 batch_size=512,
							 gamma=0.999,
							 eps_start=0.9,
							 eps_end=0.05,
							 eps_decay=100000,
							 tau=0.7,
							 lr=1e-5,
							 memory_len=10000):
		
				self.batch_size = batch_size
				self.gamma=gamma
				self.eps_start = eps_start
				self.eps_end = eps_end
				self.eps_decay = eps_decay
				self.tau = tau
				self.lr = lr
				self.memory_len = memory_len

		def __str__(self):
			return(f"Hyperparams: BATCH_SIZE={self.batch_size}, GAMMA={self.gamma}, "
						 f"EPS_START={self.eps_start}, EPS_END={self.eps_end}, EPS_DECAY={self.eps_decay}, "
						 f"TAU={self.tau}, LR={self.lr}, MEMORY_LEN={self.memory_len}")


class DQNAgent():
	def __init__(self, H:Hyperparams, n_inputs:int, n_outputs:int):
		self.H = H or Hyperparams()
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		# Networks parameters
		self.steps_done = 0
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.policy_net = DQN(n_inputs, n_outputs).to(self.device)
		self.target_net = DQN(n_inputs, n_outputs).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.H.lr, amsgrad=True)
		self.memory = ReplayMemory(self.H.memory_len)

		# Stats Tracking
		self.q_values = []
		self.losses = []

	def reset_eps_stats(self):
		self.q_values = []
		self.losses = []

	def select_action(self, state, playing=False):
		sample = random.random()
		state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		eps_threshold = self.H.eps_end + (self.H.eps_start - self.H.eps_end) * \
				math.exp(-1. * self.steps_done / self.H.eps_decay)
		self.steps_done += 1
		# print(f"Select Action: {self.steps_done}")
		if sample > eps_threshold or playing:
			with torch.no_grad():
				# t.max(1) will return the largest column value of each row.
				# second column on max result is index of where max element was
				# found, so we pick action with the larger expected reward.
				q_values = self.policy_net(state)
				self.q_values.append(q_values)
				return q_values.max(1).indices.view(1, 1)
		else:
			# print(f"RANDOM {eps_threshold}")
			action = random.randint(0,self.n_outputs-1)
			return torch.tensor([[action]], device=self.device, dtype=torch.long)
		
	def optimize_model(self):
		if len(self.memory) < self.H.batch_size:
				return
		transitions = self.memory.sample(self.H.batch_size)
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
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1).values
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.H.batch_size, device=self.device)
		with torch.no_grad():
				next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.H.gamma) + reward_batch

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
		self.losses.append(loss.item())
		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()

	def update_networks(self):
		# Update target_network form policy net current params
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for k in policy_net_state_dict:
			target_net_state_dict[k] = policy_net_state_dict[k] * self.H.tau + target_net_state_dict[k] * (1-self.H.tau)
		self.target_net.load_state_dict(target_net_state_dict)

	def memory_push(self, prev_state, prev_action, state, reward):
		reward = torch.tensor([reward], device=self.device)
		state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		prev_state = torch.tensor(prev_state, dtype=torch.float32, device=self.device).unsqueeze(0)
		self.memory.push(prev_state, prev_action, state, reward)

	def get_mean_q(self):
		if not self.q_values:
			print("No q_values")
			# self.q_values = torch.tensor(np.zeros(self.n_outputs).astype(int)),
			return np.zeros(self.n_outputs).astype(int)
		return torch.cat(self.q_values).mean(dim=0).tolist()

	def save(self, path):
		if not os.path.isdir(path):
			os.makedirs(path)
		policy = os.path.join(path, "policy.pth")
		target = os.path.join(path, "target.pth")
		optimizer = os.path.join(path, "optim.pth")
		torch.save(self.policy_net.state_dict(), policy)
		torch.save(self.target_net.state_dict(), target)
		torch.save(self.optimizer.state_dict(), optimizer)

	def load(self, path, n_eps=0):
		self.steps_done = n_eps * 601
		policy = os.path.join(path[0], "policy.pth")
		target = os.path.join(path[0], "target.pth")
		optimizer = os.path.join(path[0], "optim.pth")
		if not os.path.isfile(policy):
			logging.warning("NO POLICY FILE FOUND, CAN'T LOAD AGENT")
			return
		self.policy_net.load_state_dict(torch.load(policy))
		logging.info(f"Loaded file: {policy}")

		if not os.path.isfile(target):
			logging.warning("NO TARGET NET FILE FOUD, TRAINING WONT BE PROPERLY CONTINUED")
		else:
			self.target_net.load_state_dict(torch.load(target))
			logging.info(f"Loaded file: {target}")
		
		if not os.path.isfile(optimizer):
			logging.warning("NO OPTIMIZER FILE FOUD, TRAINING WONT BE PROPERLY CONTINUED")
		else:
			self.optimizer.load_state_dict(torch.load(optimizer))
			logging.info(f"Loaded file: {optimizer}")
		