# WITHOUT MASK
import random
import numpy as np
import pandas as pd
import os
import glob
import math
import pickle
import time

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
from networks import DQN, Transition, ReplayMemory
from utils import flush
from main_agent import Hyperparams

MAP_NAME = "CollectMineralsAndGas"

_SCREEN_DIMENSION = 84

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
MINERAL_UNITS = [
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
	units.Neutral.RichMineralField750]


class FarmingAgent(base_agent.BaseAgent):

	actions = ("do_nothing",
			"harvest_minerals",
			"build_supply_depot",
			"build_command_center",
			"train_scv",
			"train_scv2",
			"balance_workers")
	def __init__(self):
		super(FarmingAgent, self).__init__()
		self.reset()

	def reset(self):
		super(FarmingAgent, self).reset()
		# self.available_command_centers = self.POSITIONS[units.Terran.CommandCenter].copy()
		# self.available_supply_depots = self.POSITIONS[units.Terran.SupplyDepot].copy()
		self.command_centers = []
		self.supply_depots = []
		self.units = {}
		self.top_positions = {
			units.Terran.SupplyDepot: [(17, 27), (18, 28), (17, 28), (16, 28), (17, 30)],
			units.Terran.CommandCenter: [(41, 21)]
		}
		self.bot_positions = {
			units.Terran.SupplyDepot: [(40, 40), (41, 40), (40, 39), (38, 39), (41, 39)],
			units.Terran.CommandCenter: [(17, 48)]
		}
		self.positions={}
		
		# (19, 23)
		# 	Supply depots: [(17, 27), (18, 28), (17, 28), (16, 28), (17, 30)]
		# 	Barracks: [(25, 23), (25, 21), (25, 19)]
		# (17, 48)
		# 	Supply depots: [(10, 48), (10, 49), (10, 46), (10, 50), (10, 52)]
		# 	Barracks: [(18, 44), (16, 44), (20, 44)]
		# (39, 45)
		# 	Supply depots: [(40, 40), (41, 40), (40, 39), (38, 39), (41, 39)]
		# 	Barracks: [(34, 45), (34, 43), (34, 41)]
		# (41, 21)
		# 	Supply depots: [(38, 20), (38, 19), (39, 18), (38, 18), (36, 19)]
		# 	Barracks: [(42, 23), (40, 23), (44, 23)]
	
	def select_initial_position(self, obs):
		if not obs.first():
			logging.warning("Called set initial position out of first observation")
			return
		command_center = self.get_my_units_by_type(
				obs, units.Terran.CommandCenter)[0]
		self.base_top = (command_center.x < 32)
		if self.base_top:
			self.positions = self.top_positions
		else:
			self.positions = self.bot_positions

	def get_my_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type
						and unit.alliance == features.PlayerRelative.SELF]

	def get_enemy_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type
						and unit.alliance == features.PlayerRelative.ENEMY]

	def get_units_by_tag(self, obs, tag):
		return [unit for unit in obs.observation.raw_units
						if unit.tag == tag]

	def get_my_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type
						and unit.build_progress == 100
						and unit.alliance == features.PlayerRelative.SELF]

	def get_my_building_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type
						and unit.build_progress < 100
						and unit.alliance == features.PlayerRelative.SELF]

	def get_enemy_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
						if unit.unit_type == unit_type
						and unit.build_progress == 100
						and unit.alliance == features.PlayerRelative.ENEMY]

	def get_distances(self, units, xy):
		units_xy = [(unit.x, unit.y) for unit in units]
		return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

	def save_units_position(self, obs):
		all_units = [unit for unit in obs.observation.raw_units]
		info = [(unit.unit_type, unit.alliance,unit.x, unit.y) for unit in all_units]
		df = pd.DataFrame.from_records(info, columns=['Type', "ALLIANCE", "X", "Y"])

		with open("units.csv", "w") as f:
			f.write(df.to_csv())

	def delete_unit(self, unit):
		#Unused?
		current_tags = self.units[unit.unit_type]
		if unit.tag in current_tags:
			del(self.units[unit.unit_type][current_tags.index(unit.tag)])

	def update_units(self, obs, unit_type):
		"""_summary_

		Args:
			obs (_type_): _description_
			unit_type (_type_): _description_
		"""
		new = self.get_my_units_by_type(obs, unit_type)
		current = self.units.get(unit_type, [])
		new_tags = [u.tag for u in new]
		current_tags = [u.tag for u in current]
		res = []

		count = 0
		for u in current:
			if u.tag not in new_tags:
				self.positions[unit_type].append((u.x, u.y))
				del(current[count])
			else:
				res.append(new[new_tags.index(u.tag)])
		for u in new:
			if u.tag not in current_tags:
				res.append(u)

		self.units[unit_type] = res

	def check_for_new_buildings(self, obs):
		self.update_units(obs, units.Terran.CommandCenter)
		self.update_units(obs, units.Terran.SupplyDepot)		

	def step(self, obs):
		super(FarmingAgent, self).step(obs)
		self.check_for_new_buildings(obs)
		if obs.first():
			self.select_initial_position(obs)
			return actions.RAW_FUNCTIONS.no_op()

	def do_nothing(self, obs=None):
		return actions.RAW_FUNCTIONS.no_op()

	def harvest_minerals(self, obs):
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		if len(idle_scvs) > 0:
			mineral_patches = [unit for unit in obs.observation.raw_units
												 if unit.unit_type in MINERAL_UNITS]
			if len(mineral_patches) > 0:
				com_centers = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)

				scv = random.choice(idle_scvs)

				distances_scv_cc = self.get_distances(com_centers, (scv.x, scv.y))
				com_cent = com_centers[np.argmin(distances_scv_cc)]
				distances = self.get_distances(mineral_patches, (com_cent.x, com_cent.y))
				mineral_patch = mineral_patches[np.argmin(distances)]
				return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
					"now", scv.tag, mineral_patch.tag)
		return actions.RAW_FUNCTIONS.no_op()

	def build_supply_depot(self, obs):
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		if not idle_scvs:
			idle_scvs = [scv for scv in scvs if scv.order_id_0 in [359, 362]] #harvesting
		current = self.units.get(units.Terran.SupplyDepot)
		occuped_places = [(u.x,u.y) for u in current]
		# print("__________________________")
		# print(occuped_places)
		# print(free_places)
		# print("__________________________")
		free_places = [p for p in self.positions[units.Terran.SupplyDepot] if p not in occuped_places]
		can_afford = obs.observation.player.minerals >= 100
		

		if free_places and can_afford and idle_scvs:
			to_build_xy = free_places[0]
			self.positions[units.Terran.SupplyDepot].remove(to_build_xy)
			distances = self.get_distances(idle_scvs, to_build_xy)
			scv = idle_scvs[np.argmin(distances)]
			to_build_xy = (to_build_xy[0]-1, to_build_xy[1]) # little compensation
			return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
					"now", scv.tag, to_build_xy)
		return actions.RAW_FUNCTIONS.no_op()

	def build_command_center(self, obs):
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		if not idle_scvs:
			idle_scvs = [scv for scv in scvs if scv.order_id_0 in [359, 362]] #harvesting
		current = self.units.get(units.Terran.CommandCenter)
		occuped_places = [(u.x,u.y) for u in current]
		free_places = [p for p in self.positions[units.Terran.CommandCenter] if p not in occuped_places]
		can_afford = obs.observation.player.minerals >= 400
		
		if free_places and can_afford and idle_scvs:
			to_build_xy = free_places[0]
			distances = self.get_distances(idle_scvs, to_build_xy)
			scv = idle_scvs[np.argmin(distances)]
			return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
					"now", scv.tag, to_build_xy)
		return actions.RAW_FUNCTIONS.no_op()

	def train_scv(self, obs):
		completed_cc = self.get_my_completed_units_by_type(
				obs, units.Terran.CommandCenter)
		free_supply = (obs.observation.player.food_cap -
									 obs.observation.player.food_used)
		if (len(completed_cc) > 0 and obs.observation.player.minerals >= 50
				and free_supply > 0):
			cc = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
			if cc.order_length < 5:
				return actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)
		return actions.RAW_FUNCTIONS.no_op()

	def train_scv2(self, obs):
		completed_cc = self.get_my_completed_units_by_type(
				obs, units.Terran.CommandCenter)
		free_supply = (obs.observation.player.food_cap -
									 obs.observation.player.food_used)
		if (len(completed_cc) > 2 and obs.observation.player.minerals >= 50
				and free_supply > 0):
			cc = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
			if cc.order_length < 5:
				return actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)
		return actions.RAW_FUNCTIONS.no_op()

	def balance_workers(self,obs):
		"""Action to pass an scv from the Command Center with higher svc number to the
		one with lower number"""
		completed_command_centers = self.get_my_completed_units_by_type(
				obs, units.Terran.CommandCenter)
		if len(completed_command_centers) <=1:
			return self.do_nothing(obs)
		cc_pos, n_workers = list(
			zip(*[((u.x, u.y), u.assigned_harvesters) for u in completed_command_centers]))

		# # If all command centers have same number of harvesters we do nothing
		# if np.argmin(n_workers) == np.argmax(n_workers):
		# 	return self.do_nothing(obs)

		# If max difference of workers == 1 we do nothing:
		if max(n_workers) - min(n_workers) <= 1:
			return self.do_nothing(obs)

		# Choose nearest scv from max command Center with max assigned_harvesters
		to_give_pos = cc_pos[np.argmax(n_workers)]
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		scvs = [s for s in scvs if (s.order_id_0 in [359, 362])] # If is harvesting or returning to CC
		distances_scvs = self.get_distances(scvs, to_give_pos)
		chosen_scv = scvs[np.argmin(distances_scvs)]


		# Choose nearest mineral from command center with min assigned_harvesters
		to_recive_pos = cc_pos[np.argmin(n_workers)]
		minerals = [unit for unit in obs.observation.raw_units
												 if unit.unit_type in MINERAL_UNITS]
		distances_minerals = self.get_distances(minerals, to_recive_pos)
		chosen_mineral = minerals[np.argmin(distances_minerals)]

		# Sends SCV to mineral
		return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
					"now", chosen_scv.tag, chosen_mineral.tag)

	def invalid_action_masking(self, obs):
		# SCV
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		can_afford_scv = obs.observation.player.minerals >= 50
		# Supply
		free_supply = (obs.observation.player.food_cap -
									 obs.observation.player.food_used)

		# Supply Depots
		supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
		completed_supply_depots = self.get_my_completed_units_by_type(obs, units.Terran.SupplyDepot)
		can_afford_supply_depot = obs.observation.player.minerals >= 100
		max_supply_depots_reached = len(self.positions.get(units.Terran.SupplyDepot)) <= 0

		# Command Centers
		command_centers = self.units.get(units.Terran.CommandCenter, [])
		can_afford_command_center = obs.observation.player.minerals >= 400
		completed_command_centers = self.get_my_completed_units_by_type(obs, units.Terran.CommandCenter)
		max_command_centers_reached = len(self.positions.get(units.Terran.CommandCenter)) <= 0
		queued_scv_1 = (command_centers[0].order_length
				  if len(completed_command_centers) > 0 else 0)
		queued_scv_2 = (command_centers[1].order_length
				  if len(completed_command_centers) > 1 else 0)
		workers_cc_1 = (command_centers[0].assigned_harvesters
				  if len(completed_command_centers) > 0 else 0)
		workers_cc_2 = (command_centers[1].assigned_harvesters
				  if len(completed_command_centers) > 1 else 0)
		workers_balance = [workers_cc_1, workers_cc_2]
		workers_are_balenced = (max(workers_balance) - min(workers_balance)) <= 1

		action_mask = np.zeros(len(self.actions)).astype(int)
		# print(queued_scv_1, queued_scv_2)
		# INVALID HARVEST MINERALS
		if len(idle_scvs) <= 0:
			action_mask[self.actions.index("harvest_minerals")] = 1
		# INVALID BUILD SUPPLY DEPOT
		if not can_afford_supply_depot or max_supply_depots_reached:
			action_mask[self.actions.index("build_supply_depot")] = 1
		# INVALID BUILD Command Center
		if not can_afford_command_center or max_command_centers_reached:
			action_mask[self.actions.index("build_command_center")] = 1
		# Invalid TRAIN SCV 1
		if not can_afford_scv or queued_scv_1 >= 5 or free_supply < 1:
			action_mask[self.actions.index("train_scv")] = 1
		# Invalid TRAIN SCV 2
		if (not can_afford_scv or queued_scv_2 >= 5 or free_supply < 1 or len(completed_command_centers)<=1):
			action_mask[self.actions.index("train_scv2")] = 1
		# Invalid Balance Workers
		if len(completed_command_centers) <= 1 or workers_are_balenced:
			action_mask[self.actions.index("balance_workers")] = 1

		return action_mask


class MiningRandomAgent64(FarmingAgent):
	def __init__(self, H:Hyperparams=None, *args, **kwargs):
		super(FarmingAgent, self).__init__()
		self.H = H or Hyperparams()

		# TODO: Do not hardcode following variables
		n_observations = 14 # form counting self.get_state return
		n_actions = len(self.actions)
		self.playing = kwargs.get('playing', False)
		self.name =  kwargs.get('name', "Mining_random_agent")
		self.description = kwargs.get('description', "DRL Agent with main DQN and"
								"Target DQN, Replay memory and a the end of -1 | 0 | 1")
		self.execution_id = datetime.strftime(
			datetime.now(),
			"%y%m%d%H%M%S")
		self.execution_folder = os.path.join(
			"./saves",
			self.name,
			self.execution_id + ("playing" if self.playing else ""))
		self.rewards = []
		logging.info(f"Saving folder: {self.execution_folder}")

	def step(self, obs):
		super(MiningRandomAgent64, self).step(obs)
		action = random.choice(self.actions)
		# self.rewards.append(obs.reward)
		# if obs.last():
		# 	with open("rewards.pkl", "wb") as wf:
		# 		pickle.dump(self.rewards, wf) # serialize the list
		# print(f"Executing {action}")
		return getattr(self, action)(obs)


class MiningDRLAgent64(FarmingAgent):

	def __init__(self, H:Hyperparams=None, *args, **kwargs):
		super(FarmingAgent, self).__init__()
		self.H = H or Hyperparams()
		# TODO: Do not hardcode following variables
		n_observations = 14 # form counting self.get_state return
		n_actions = len(self.actions)
		self.playing = kwargs.pop('playing', False)
		self.name =  kwargs.pop('name', "MINING_DRL")
		self.description = kwargs.pop('description', "DRL Agent with main DQN and"
								"Target DQN, Replay memory")
		self.execution_id = datetime.strftime(
			datetime.now(),
			"%y%m%d%H%M%S")
		self.execution_folder = os.path.join(
			"./saves",
			self.name,
			self.execution_id + ("playing" if self.playing else ""))
		self.debug = kwargs.pop("debug", False)
		self.save_backups = kwargs.pop("save", True)
		
		#Training parameters
		self.steps_done = 0
		self.reward_inv_act_mask = kwargs.pop("reward_inv_act_mask", 0) # Invalid Action Masking
		self.reward_normalize = kwargs.pop("reward_normalize", False)
		self.reward_cap_zero = kwargs.pop("reward_cap_zero", False)
		self.reward_use_compensation = kwargs.pop("reward_use_compensation", False)
		self.reward_compensation_method = kwargs.pop("reward_compensation", self.reward_compensation) # method to pass to all rewards
		self.reward_sparse = kwargs.pop("reward_sparse", False) # Keeps episode steps in memory and optimize all them with final episode reward

		# Networks parameters
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.policy_net = DQN(n_observations, n_actions).to(self.device)
		self.target_net = DQN(n_observations, n_actions).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.H.lr, amsgrad=True)
		self.memory = ReplayMemory(self.H.memory_len)

		# Data collection variables
		self.rewards = [] # Store rewards of the whole episode
		self.episode_rewards = [] # Store rewards of every step, restarts in new game
		self.n_epis_to_mean = 10 # Number of episodes to calculate mean and save
		self.n_epis_force_save = 100
		self.max_mean_n_rewards = -math.inf
		self.n_eps = 0
		self.q_values = []

		# Codecarbon tracker
		self.tracker = None

		# REWARD COMPENSATION
		self.global_mean_reward = 10.635635961344484
		self.global_std_reward = 7.869382086382208

		if self.reward_inv_act_mask > 0:
			logging.warning("Invalid Action Reward is positive, This WONT WORK")
		logging.info(f"Saving folder: {self.execution_folder}")
		logging.info(f"Chosen device is {self.device}")

		if kwargs:
			logging.warning(f"NOT ALL KWARGS USED: {kwargs}")

		self.new_game()

	def log_config(self):
		logging.info(f"Training Description: {self.description}")
		logging.info(f"Save folder: {self.execution_folder}")
		logging.info(f"Started training with following parameters \n{self.H}")
		reward_conf_dict = {
			   'invalid_act_mask': self.reward_inv_act_mask,
			   'normalize': self.reward_normalize,
			   'cap_zero': self.reward_cap_zero,
			   'compensation': self.reward_use_compensation,
			   'compensation_method': self.reward_compensation_method.__name__,
			   'sparse': self.reward_sparse}
		logging.info(f"Reward Shaping:\n {reward_conf_dict}")

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

		logging.warning(f"NO FOLDER FOUND, CAN'T LOAD AGENT\n\t\t{p}")

	def set_tracker(self, tracker):
		self.tracker = tracker

	def reset(self):
		super(MiningDRLAgent64, self).reset()
		self.new_game()

	def new_game(self):
		if self.playing:
			logging.warning("EXECUTING IN PLAY MODE, NOT LEARNING")
		self.base_top = None
		self.previous_state = None
		self.previous_action = None
		self.n_steps = 0
		self.prev_score = 0
		self.episode_rewards = []
		self.losses = []
		self.q_values = []
		self.prev_action_mask = []

	@staticmethod
	def get_mean_pos(unit_grp):
		positions = ((u.x, u.y) for u in unit_grp)
		mean_pos = [sum(v)/len(v) for v in list(zip(*positions))]
		return(mean_pos)

	def get_state(self, obs):
		
		# SCVs
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		if obs.first():
			initial_pos = self.get_mean_pos(scvs)
			self.initial_pos_center = initial_pos[0]>30

		
		# Command Centers
		completed_command_centers = self.get_my_completed_units_by_type(
				obs, units.Terran.CommandCenter)

		queued_scv_1 = (completed_command_centers[0].order_length
											if len(completed_command_centers) > 0 else 0)
		queued_scv_2 = (completed_command_centers[1].order_length
											if len(completed_command_centers) > 1 else 0)
		workers_cc_1 = (completed_command_centers[0].assigned_harvesters
											if len(completed_command_centers) > 0 else 0)

		workers_cc_2 = (completed_command_centers[1].assigned_harvesters
											if len(completed_command_centers) > 1 else 0)
		free_supply = (obs.observation.player.food_cap -
									 obs.observation.player.food_used)
		can_afford_supply_depot = obs.observation.player.minerals >= 100
		can_afford_command_center = obs.observation.player.minerals >= 400
		can_afford_scv = obs.observation.player.minerals >= 50

		supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
		completed_supply_depots = self.get_my_completed_units_by_type(
				obs, units.Terran.SupplyDepot)

		return (
			self.initial_pos_center,
			len(completed_command_centers),
			len(scvs),
			len(idle_scvs),
			len(supply_depots),
			len(completed_supply_depots),
			queued_scv_1,
			workers_cc_1,
			queued_scv_2,
			workers_cc_2,
			free_supply,
			can_afford_supply_depot,
			can_afford_command_center,
			can_afford_scv,
			)

	def reward_normalization(self, reward):
		return (reward - self.global_mean_reward)/ self.global_std_reward

	def reward_compensation(self, reward):
		pass

	def invalid_prev_action(self, obs):
		if obs.first():
			self.prev_action_mask = self.invalid_action_masking(obs)
			return int(0)
		res = self.prev_action_mask[self.previous_action]
		self.prev_action_mask = self.invalid_action_masking(obs)
		return res

	def get_reward(self, obs):
		rew = obs.reward
		if self.reward_normalize:
			rew = self.reward_normalization(rew)
		if self.reward_cap_zero:
			rew = max(rew, 0)
		if self.reward_inv_act_mask:
			rew = (rew, self.reward_inv_act_mask)[self.invalid_prev_action(obs)]
		if self.reward_sparse:
			rew = 0
		return rew

	def get_episode_reward(self,obs):
		# self.reward_use_compensation = kwargs.get("reward_use_compensation", False)
		# self.reward_compensation_method = kwargs.get("reward_compensation", self.reward_compensation) # method to pass to all rewards
		episode_rewards = self.episode_rewards
		if self.reward_use_compensation:
			episode_rewards = [self.reward_compensation_method(r) for r in episode_rewards]
		return sum(episode_rewards)

	def get_mean_q(self):
		return torch.cat(self.q_values).mean(dim=0).tolist()

	def get_used_actions(self):
		# TODO: LESS LOOPS AND MORE MATHS
		action_list = np.zeros(len(self.actions)).astype(int)
		for q_val in self.q_values:
			index = q_val.max(1).indices.view(1, 1)
			action_list[index] += 1
		return action_list.tolist()

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
				q_values = self.policy_net(state)
				self.q_values.append(q_values)
				return q_values.max(1).indices.view(1, 1)
		else:
			# print("RANDOM")
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

	def save(self, folder):
		if not self.save_backups:
			return
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
		mean_episode_losses = sum(self.losses) / len(self.losses) if self.losses else 0
		# logging.info(f"Eps {self.n_eps} finished with reward {r}, new mean = {mean} in {self.n_steps} steps")
		log_info = {
			"eps": self.n_eps, 
			"r": r, 
			"mean": mean, 
			"steps": self.n_steps, 
			"ep_mean_loss": mean_episode_losses, 
			"mean_q": self.get_mean_q(), 
			"used_actions": self.get_used_actions()}
		logging.info(log_info)

		self.flush_carbon_stats()

		if self.playing:
			return

		if (len(to_mean) >= self.n_epis_to_mean and (mean >= self.max_mean_n_rewards)) or (self.n_eps%self.n_epis_force_save == 0):
			logging.info("saving")
			self.max_mean_n_rewards = max(self.max_mean_n_rewards, mean)
			self.save(f"{self.n_eps}_{mean}")

	def memory_push(self, obs, state, reward):
		reward = torch.tensor([reward], device=self.device)
		if not obs.first() and not self.playing:
			self.memory.push(self.previous_state, self.previous_action, state, reward)
	
	def step(self, obs):
		super(MiningDRLAgent64, self).step(obs)
		self.n_steps += 1

		done = obs.last()

		observation = self.get_state(obs)
		state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
		action = self.select_action(state)
		reward = self.get_reward(obs)

		self.episode_rewards.append(reward)
		self.memory_push(obs, state, reward)

		self.previous_state = state
		self.previous_action = action
		if not self.playing:
			self.optimize_model()
		if done:
			self.episode_done(obs) # Saves if rewards mean is higher than before
			self.update_networks()
		action_str = self.actions[action]
		if self.debug:
			print(action_str, self.invalid_prev_action(obs), reward)
		return getattr(self, action_str)(obs)


class MiningScriptedAgent(FarmingAgent):
	def __init__(self, H:Hyperparams=None, *args, **kwargs):
		super(FarmingAgent, self).__init__()
		self.H = H or Hyperparams()
		# TODO: Do not hardcode following variables
		self.playing = kwargs.get('playing', False)
		self.name =  kwargs.get('name', "Mining Scripted Agent")
		self.description = kwargs.get('description', "Scripted agent to beat minigame")
		self.execution_id = datetime.strftime(
			datetime.now(),
			"%y%m%d%H%M%S")
		self.execution_folder = os.path.join(
			"./saves",
			self.name,
			self.execution_id + ("playing" if self.playing else ""))
		self.rewards = []
		logging.info(f"Saving folder: {self.execution_folder}")

	def step(self, obs):
		super(MiningScriptedAgent, self).step(obs)
		# return getattr(self, "harvest_minerals")(obs)
		# return getattr(self, "do_nothing")(obs)
		scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
		idle_scvs = [scv for scv in scvs if scv.order_length == 0]
		free_supply = (obs.observation.player.food_cap -
									 obs.observation.player.food_used)
		virtual_supply = 8 * len(self.get_my_building_units_by_type(obs, units.Terran.SupplyDepot))
		virtual_supply += 15 * len(self.get_my_building_units_by_type(obs, units.Terran.CommandCenter))
		can_afford_supply_depot = obs.observation.player.minerals >= 100
		can_afford_command_center = obs.observation.player.minerals >= 400
		can_afford_scv = obs.observation.player.minerals >= 50
		completed_command_centers = self.get_my_completed_units_by_type(
				obs, units.Terran.CommandCenter)
		queued_scv_1 = (completed_command_centers[0].order_length
											if len(completed_command_centers) > 0 else 0)
		queued_scv_2 = (completed_command_centers[1].order_length
											if len(completed_command_centers) > 1 else 0)

		workers_cc_1 = (completed_command_centers[0].assigned_harvesters
											if len(completed_command_centers) > 0 else 0)

		workers_cc_2 = (completed_command_centers[1].assigned_harvesters
											if len(completed_command_centers) > 1 else 0)

		max_workers_cc_1 = (completed_command_centers[0].ideal_harvesters
											if len(completed_command_centers) > 0 else 0)

		max_workers_cc_2 = (completed_command_centers[1].ideal_harvesters
											if len(completed_command_centers) > 1 else 0)

		max_workers_reached_1 = (workers_cc_1 + queued_scv_1) >= max_workers_cc_1
		max_workers_reached_2 = (workers_cc_2 + queued_scv_2) >= max_workers_cc_2

		# Idles to work
		if len(idle_scvs) + virtual_supply > 0:
			# print("IDLS TO WORK")
			return self.harvest_minerals(obs)

		# Low free supply -> Build supply deppot or do nothing (to save resources)
		if free_supply <= queued_scv_1 + queued_scv_2:
			# print("Build SVC")
			# If it can't afford it it will return no_op so it will keep trying
			# until enougth resources have been saved.
			return self.build_supply_depot(obs)

		if can_afford_command_center and len(completed_command_centers) < 2:
			# print("BUILD CC")
			return self.build_command_center(obs)

		# TODO: Train workers on the one with less workers
		# Train workers on CC 1 if possible
		if (len(completed_command_centers) > 0 and
	  		queued_scv_1 < 5 and
			not max_workers_reached_1 and
			can_afford_scv):
			# print("TRAIN 1")
			return self.train_scv(obs)

				# Train workers on CC 2 if possible
		if (len(completed_command_centers) > 1 and
	  		queued_scv_2 < 5 and
	  		not max_workers_reached_2 and
			can_afford_scv):
			# print("TRAIN 2")
			return self.train_scv2(obs)

		# print("Balance Workers")
		return self.balance_workers(obs)

		print("DO NOTHING")
		return self.do_nothing()

