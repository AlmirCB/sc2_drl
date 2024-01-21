######################################################################
##### THIS IS JUST A TEST TO CHECK MARINE POSITION IN EVERY STEP #####
################ NEEDS TO MOVE MARINE WITH THE MOUSE #################
######################################################################



import random
import numpy as np
import pandas as pd
import os
import sys
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop, environment
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import pickle
EXECUTION_STARTS = datetime.strftime(datetime.now(), "%y%m%d-%H%M")


MAP_NAME = "MoveToBeacon"
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_SCREEN_DIMENSION = 84

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

STIMATIONS = [{}] # EPISODE, STEP, STATE (WITH ACTION), STIMATED_REWARD, NEXT_STIMATED_REWARD, REAL_REWARD, MU_X, SIGMA_X, MU_Y, SIGMA_Y
EPISODE_COUNTS = 0



def save_stimations():
    print("Saving stimations")
    dir = f"stimations-{EXECUTION_STARTS}"
    if not os.path.isdir(dir):
      os.mkdir(dir)
    file = os.path.join(dir, f'st_{EPISODE_COUNTS}.pkl')
    with open(file, 'wb') as f:  # open a text file
      pickle.dump(STIMATIONS, f) # serialize the list

def clip_grad_norm_(module, max_grad_norm = 0.5):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)
   

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ConvNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ConvNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4), # TODO: No se a qué se refiere con la"activación ReLU" que menciona el enunciado.
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=fc1_dims, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=fc1_dims, out_channels=fc2_dims, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.linear_net = nn.Linear(
           
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


class RL_Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=4,
                 layer1_size=64, layer2_size=64, n_outputs=1):
        self.alpha = alpha # Actor lr
        self.beta = beta # Critic lr
        self.input_dims = input_dims 
        self.gamma = gamma # Discount factor (for future rewards)
        self.n_actions = n_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.n_outputs = n_outputs

        self.log_probs = None # Log probability of selecting an action
        self.actor = GenericNetwork(alpha, 4, layer1_size, 
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, 6, layer1_size, 
                                    layer2_size, n_actions=1)
        
    def choose_actions(self, observation):
        global STIMATIONS
        # print(self.actor.forward(observation))
        mu_x, sigma_x, mu_y, sigma_y = self.actor.forward(observation)
        
        STIMATIONS[-1].update({
            "mu_x": mu_x.item(), 
            "sigma_x": sigma_x.item(), 
            "mu_y": mu_y.item(), 
            "sigma_y": sigma_y.item()})
        # print(mu_x.item(), sigma_x.item(), mu_y.item(), sigma_y.item())
        sigma_x = T.exp(sigma_x) # As our network could return a negative sigma
        sigma_y = T.exp(sigma_y)
        action_probs_x = T.distributions.Normal(mu_x, sigma_x)
        action_probs_y = T.distributions.Normal(mu_y, sigma_y)
        probs_x = action_probs_x.sample(sample_shape=T.Size([self.n_outputs]))
        probs_y = action_probs_y.sample(sample_shape=T.Size([self.n_outputs]))
        log_probs_x = action_probs_x.log_prob(probs_x).to(self.actor.device)
        log_probs_y = action_probs_y.log_prob(probs_y).to(self.actor.device)
        p_x = T.exp(log_probs_x)
        p_y = T.exp(log_probs_y)
        # self.log_probs = log_probs_x * log_probs_y
        self.log_probs = T.log((p_x + p_y) / 2)
        STIMATIONS[-1].update({
           "log_probs_x": log_probs_x.item(),
           "log_probs_y": log_probs_y.item(),
           "log_probs": self.log_probs.item()
        })
        action_x = T.sigmoid(probs_x) * _SCREEN_DIMENSION
        action_y = T.sigmoid(probs_y) * _SCREEN_DIMENSION

        # Gym does not accept tensors so we pass the item()
        return int(action_x.item()), int(action_y.item())
    

    def learn(self, state, reward, new_state, done):
        global STIMATIONS
        # With torch we should zero_grad the optimizer at the top because
        # we don't want the gradient of previous samples to affect the 
        # current one


        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        # print(f"Critic_next = {critic_value_}, Critic = {critic_value}")

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        # When done 1-1 = 0 -> discard future reward
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2
        STIMATIONS[-1].update({
            "old_critic": critic_value.item(), 
            "new_critic": critic_value_.item(), 
            "delta": delta.item(), 
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()})
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        clip_grad_norm_(self.actor.optimizer)
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.optimizer)        
        self.critic.optimizer.step()



class BeaconAgent(base_agent.BaseAgent):
  
    rl_ag = RL_Agent(alpha=0.000005, beta=10e-5, input_dims=4, gamma=0.99,
                  layer1_size=256, layer2_size=256)
    
    n_steps = 0
    total_reward = 0
    all_rewards = []
    last_state = None
    dist = None

    @staticmethod
    def get_my_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]
    
    @staticmethod
    def get_beacon(obs):
        units = [u for u in obs.observation.feature_units
                 if u.alliance == features.PlayerRelative.NEUTRAL]
        return units


    def step(self, obs):
        
        super(BeaconAgent, self).step(obs)
        
        marine = self.get_my_units_by_type(obs, units.Terran.Marine)[0]
        beacon = self.get_beacon(obs)[0]
        marine_pos = (marine.x, marine.y)
        beacon_pos = (beacon.x, beacon.y)
        print(f"Marine: {marine_pos}, Beacon: {beacon_pos}")
        # if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
           
        #     return FUNCTIONS.Move_screen("now", beacon_pos)
        # else:
        #     return FUNCTIONS.select_army("select")
        return FUNCTIONS.no_op()

        


def main(unused_argv):
#   agent = MoveToBeacon()
  agent = BeaconAgent()
  try:
    with sc2_env.SC2Env(
        map_name=MAP_NAME,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            # action_space=actions.ActionSpace.RAW,
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_raw_units=True,
            use_feature_units=True,
            use_unit_counts=True,
            # raw_resolution=64,
            allow_cheating_layers = True            
        ),
        step_mul=160,
        disable_fog=True,
        realtime=True,
        # visualize=True,
    ) as env:
      run_loop.run_loop([agent], env, max_episodes=1000000)
  except KeyboardInterrupt:
    pass



if __name__ == "__main__":
  app.run(main)

