import random
import numpy as np
import os
import glob
import math

# TORCH IMPORTs: TODO: Move all the functionality involving torch to networks.py file.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

# PYSC2 imports
from absl import logging
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

# Project imports
from networks import DQN, Transition, ReplayMemory, Hyperparams
from utils import flush

# Training conf

class Agent(base_agent.BaseAgent):
	actions = ("do_nothing",
						 "harvest_minerals", 
						 "build_supply_depot", 
						 "build_barracks", 
						 "train_marine", 
						 "attack")
	
	def get_my_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type 
						and unit.alliance == features.PlayerRelative.SELF]
	
	def get_enemy_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type 
						and unit.alliance == features.PlayerRelative.ENEMY]
	
	def get_my_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type 
						and unit.build_progress == 100
						and unit.alliance == features.PlayerRelative.SELF]
		
	def get_enemy_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type 
						and unit.build_progress == 100
						and unit.alliance == features.PlayerRelative.ENEMY]
		
	def get_distances(self, obs, units, xy):
		units_xy = [(unit.x, unit.y) for unit in units]
		return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

	def step(self, obs):
		super(Agent, self).step(obs)
		if obs.first():
			command_center = self.get_my_units_by_type(
					obs, units.Terran.CommandCenter)[0]
			self.base_top_left = (command_center.x < 32)
			
	def do_nothing(self, obs):
		return actions.RAW_FUNCTIONS.no_op()
	
	def harvest_minerals(self, obs):
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		if len(idle_scvs) > 0:
			mineral_patches = [unit for unit in obs.observation.raw_units
												 if unit.unit_type in [
													 units.Neutral.BattleStationMineralField,
													 units.Neutral.BattleStationMineralField750,
													 units.Neutral.LabMineralField,
													 units.Neutral.LabMineralField750,
													 units.Neutral.MineralField,
													 units.Neutral.MineralField750,
													 units.Neutral.PurifierMineralField,
													 units.Neutral.PurifierMineralField750,
													 units.Neutral.PurifierRichMineralField,
													 units.Neutral.PurifierRichMineralField750,
													 units.Neutral.RichMineralField,
													 units.Neutral.RichMineralField750
												 ]]
			scv = random.choice(idle_scvs)
			distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
			mineral_patch = mineral_patches[np.argmin(distances)] 
			return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
					"now", scv.tag, mineral_patch.tag)
		return actions.RAW_FUNCTIONS.no_op()
	
	def build_supply_depot(self, obs):
		supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
				len(scvs) > 0):
			supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
			distances = self.get_distances(obs, scvs, supply_depot_xy)
			scv = scvs[np.argmin(distances)]
			return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
					"now", scv.tag, supply_depot_xy)
		return actions.RAW_FUNCTIONS.no_op()
		
	def build_barracks(self, obs):
		completed_supply_depots = self.get_my_completed_units_by_type(
				obs, units.Terran.SupplyDepot)
		barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and 
				obs.observation.player.minerals >= 150 and len(scvs) > 0):
			barracks_xy = (22, 21) if self.base_top_left else (35, 45)
			distances = self.get_distances(obs, scvs, barracks_xy)
			scv = scvs[np.argmin(distances)]
			return actions.RAW_FUNCTIONS.Build_Barracks_pt(
					"now", scv.tag, barracks_xy)
		return actions.RAW_FUNCTIONS.no_op()
		
	def train_marine(self, obs):
		completed_barrackses = self.get_my_completed_units_by_type(
				obs, units.Terran.Barracks)
		free_supply = (obs.observation.player.food_cap - 
									 obs.observation.player.food_used)
		if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
				and free_supply > 0):
			barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
			if barracks.order_length < 5:
				return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
		return actions.RAW_FUNCTIONS.no_op()
	
	def attack(self, obs):
		marines = self.get_my_units_by_type(obs, units.Terran.Marine)
		if len(marines) > 0:
			attack_xy = (38, 44) if self.base_top_left else (19, 23)
			distances = self.get_distances(obs, marines, attack_xy)
			marine = marines[np.argmax(distances)]
			x_offset = random.randint(-4, 4)
			y_offset = random.randint(-4, 4)
			return actions.RAW_FUNCTIONS.Attack_pt(
					"now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
		return actions.RAW_FUNCTIONS.no_op()

class RandomAgent(Agent):
	def step(self, obs):
		super(RandomAgent, self).step(obs)
		action = random.choice(self.actions)
		return getattr(self, action)(obs)

class BasicDRLAgent(Agent):
	
	def __init__(self, H:Hyperparams=None, *args, **kwargs):
		super(BasicDRLAgent, self).__init__()
		self.H = H or Hyperparams()

		# TODO: Do not hardcode following variables
		self.playing = kwargs.get('playing', False)
		self.name =  kwargs.get('name', "BASIC_DRL_SPARSED")
		self.description = kwargs.get('description', "DRL Agent with main DQN and" 
								"Target DQN, Replay memory and a the end of -1 | 0 | 1")
		self.execution_id = datetime.strftime(
			datetime.now(),
			"%y%m%d%H%M%S"
		)
		self.execution_folder = os.path.join(
			"./saves",
			self.name,
			self.execution_id + ("playing" if self.playing else "")
		)

		logging.info(f"Saving folder: {self.execution_folder}")

		n_observations = 21 # form counting self.get_state return
		n_actions = 6 # do_nothing, harvest_minerals, build_supply_depot, build_barracks, train_marine, attack

		self.steps_done = 0
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logging.info(f"Chosen device is {self.device}")
		self.policy_net = DQN(n_observations, n_actions).to(self.device)
		self.target_net = DQN(n_observations, n_actions).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())

		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.H.lr, amsgrad=True)
		self.memory = ReplayMemory(self.H.memory_len)
		
		self.rewards = []
		self.n_epis_to_mean = 10 # Number of episodes to calculate mean and save
		self.n_epis_force_save = 100
		self.max_mean_n_rewards = -math.inf
		self.n_eps = 0

		self.tracker = None # Codecarbon tracker
		
		self.new_game()

	def load(self, load_path, episode):
		p = load_path + "/" + str(episode) + "*"
		if path := glob.glob(p):
			policy = os.path.join(path[0], "policy.pth")
			# target = os.path.join(path[0], "target.pth")
			if not os.path.isfile(policy):
				logging.warning("NO POLICY FILE FOUND, CAN'T LOAD AGENT")
				return
				
			self.policy_net.load_state_dict(torch.load(policy))
			logging.info(f"Loaded file: {policy}")
			return
		
		logging.warning(f"NO FOLDER FOUND, CAN'T LOAD AGENT\n{p}")

	def set_tracker(self, tracker):
		self.tracker = tracker

	def reset(self):
		super(BasicDRLAgent, self).reset()
		self.new_game()

	def new_game(self):
		if self.playing:
			logging.warning("EXECUTING IN PLAY MODE, NOT LEARNING")
		self.base_top_left = None
		self.previous_state = None
		self.previous_action = None
		self.n_steps = 0
		self.prev_score = 0
	
	def get_state(self, obs):
		#SCVS
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		
		# CC
		command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
		
		#Supply Depots
		supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
		completed_supply_depots = self.get_my_completed_units_by_type(
				obs, units.Terran.SupplyDepot)
		can_afford_supply_depot = obs.observation.player.minerals >= 100

		# Barrackses
		barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
		completed_barrackses = self.get_my_completed_units_by_type(
				obs, units.Terran.Barracks)
		can_afford_barracks = obs.observation.player.minerals >= 150

		# Marines
		marines = self.get_my_units_by_type(obs, units.Terran.Marine)
		queued_marines = (completed_barrackses[0].order_length 
											if len(completed_barrackses) > 0 else 0)
		can_afford_marine = obs.observation.player.minerals >= 100

		# Free supply
		free_supply = (obs.observation.player.food_cap - 
									 obs.observation.player.food_used)
		
		# Enemy SCVS
		enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
		enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]

		# Enemy CC
		enemy_command_centers = self.get_enemy_units_by_type(
				obs, units.Terran.CommandCenter)
		
		# Enemy Supply Depots
		enemy_supply_depots = self.get_enemy_units_by_type(
				obs, units.Terran.SupplyDepot)
		enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
				obs, units.Terran.SupplyDepot)
		
		# Enemy Barrackses
		enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
		enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
				obs, units.Terran.Barracks)
		
		# Enemy Marines
		enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
		
		return (len(command_centers),
						len(scvs),
						len(idle_scvs),
						len(supply_depots),
						len(completed_supply_depots),
						len(barrackses),
						len(completed_barrackses),
						len(marines),
						queued_marines,
						free_supply,
						can_afford_supply_depot,
						can_afford_barracks,
						can_afford_marine,
						len(enemy_command_centers),
						len(enemy_scvs),
						len(enemy_idle_scvs),
						len(enemy_supply_depots),
						len(enemy_completed_supply_depots),
						len(enemy_barrackses),
						len(enemy_completed_barrackses),
						len(enemy_marines))

	def get_reward(self, obs):
		return obs.reward
	
	def get_episode_reward(self,obs):
		return obs.reward

	def select_action(self, state):
		sample = random.random()
		eps_threshold = self.H.eps_end + (self.H.eps_start - self.H.eps_end) * \
				math.exp(-1. * self.steps_done / self.H.eps_decay)
		self.steps_done += 1
		if sample > eps_threshold or self.playing:
				with torch.no_grad():
						# t.max(1) will return the largest column value of each row.
						# second column on max result is index of where max element was
						# found, so we pick action with the larger expected reward.
						return self.policy_net(state).max(1).indices.view(1, 1)
		else:
				action = random.randint(0,len(self.actions)-1)
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

	def save(self, folder):
		path = os.path.join(self.execution_folder, folder)
		if not os.path.isdir(path):
			os.makedirs(path)
		policy = os.path.join(path, "policy.pth")
		target = os.path.join(path, "target.pth")
		torch.save(self.policy_net.state_dict(), policy)
		torch.save(self.target_net.state_dict(), target)

	def flush_carbon_stats(self):
		if self.tracker:
			carbon_stats = flush(self.tracker)
			logging.info(carbon_stats)

	def episode_done(self, obs):
		self.n_eps += 1
		r = self.get_episode_reward(obs)
		self.rewards.append(r)
		to_mean = self.rewards[-self.n_epis_to_mean:]
		mean = sum(to_mean) / self.n_epis_to_mean
		logging.info(f"Eps {self.n_eps} finished with reward {r}, new mean = {mean} in {self.n_steps} steps")	
		if self.playing:
			return
		
		self.flush_carbon_stats()

		if (len(to_mean) >= self.n_epis_to_mean and (mean >= self.max_mean_n_rewards)) or (self.n_eps%self.n_epis_force_save == 0):
			logging.info("saving")
			self.max_mean_n_rewards = max(self.max_mean_n_rewards, mean)
			self.save(f"{self.n_eps}_{mean}")

	def memory_push_and_rotation(self, obs, state, action, reward):
		reward = torch.tensor([reward], device=self.device)
		if not obs.first() and not self.playing:
			# As in first step previous_state and previous action will be None
			self.memory.push(self.previous_state, self.previous_action, state, reward)
			
		self.previous_state = state
		self.previous_action = action

	def step(self, obs):
		super(BasicDRLAgent, self).step(obs)
		self.n_steps += 1

		done = obs.last()

		observation = self.get_state(obs)
		state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
		action = self.select_action(state)
		reward = self.get_reward(obs)
	
		self.memory_push_and_rotation(obs, state, action, reward)
		if not self.playing:
			self.optimize_model()
			self.update_networks()

		if done:
			self.episode_done(obs) # Saves if rewards mean is higher than before

		action_str = self.actions[action]
		return getattr(self, action_str)(obs)

class DRLReshapeAgent(BasicDRLAgent):
	def __init__(self, H:Hyperparams=None):
		name = "DRL_reward_shaping"
		description = "DRL Agent with main DQN and Target DQN, Replay memory and a reward each step corresponding to\
				the score increased +- 10000 if it win/loses the game"
		super(DRLReshapeAgent, self).__init__(H, name=name, description=description)
		
	def get_reward(self, obs):
		#'score_cumulative', 'score_by_category', 'score_by_vital'

		score = obs.observation['score_cumulative']['score']
		prev_score = self.prev_score
		self.prev_score = score
		return (obs.reward * 10000) + score - prev_score
	
	def get_episode_reward(self,obs):
		score = obs.observation['score_cumulative']['score']
		return (obs.reward * 10000) + score

class SparseAgent(BasicDRLAgent):
	def __init__(self, STEP_MUL, H:Hyperparams=None, *args, **kwargs):
		# As network update is going to be done just once each episode we need a big TAU
		H = H or Hyperparams(tau=0.7)
		name = "DRL_compensation_sparse"
		description = "DRL Agent with sparsed reward, at the end of the episode it will give a reward to each given step"
		super(SparseAgent, self).__init__(H, name=name, description=description, **kwargs)

		# As I counted the max number of steps is 28800 (21:25 min) and the game is given as a TIE
		# TODO: GET MAX NUMBER OF STEPS AS PARAMETER
		steps_per_episode = math.ceil(28800 / STEP_MUL) + 1
		self.episode_memory = ReplayMemory(steps_per_episode)
		
	def learn_at_the_end(self, reward, add_penalty=False):
		# This method is used to, once we have the last reward, propagate it to all transitions
		reward = torch.tensor([reward], device=self.device)
		adapted_reward = reward
		while self.episode_memory:
			if add_penalty:
				adapted_reward = reward / len(self.episode_memory)
			t = self.episode_memory.memory.popleft()
			self.memory.push(t.state, t.action, t.next_state, adapted_reward)
			self.optimize_model()
		
		self.update_networks()

	def memory_push_and_rotation(self, obs, state, action, reward):
		reward = torch.tensor([reward], device=self.device)
		if not obs.first() and not self.playing:
			# As in first step previous_state and previous action will be None
			self.episode_memory.push(self.previous_state, self.previous_action, state, reward)
			
		self.previous_state = state
		self.previous_action = action

	def step(self, obs):
		super(BasicDRLAgent, self).step(obs)
		self.n_steps += 1

		done = obs.last()

		observation = self.get_state(obs)
		state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
		action = self.select_action(state)
		reward = self.get_reward(obs)
	
		self.memory_push_and_rotation(obs, state, action, reward)

		if done:
			self.episode_done(obs) # Saves if rewards mean is higher than before and output logs
			if not self.playing:
				self.learn_at_the_end(reward, add_penalty=True)

		action_str = self.actions[action]
		return getattr(self, action_str)(obs)
