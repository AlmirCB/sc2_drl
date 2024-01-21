from absl import app, logging
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env, run_loop, environment
from main_agent import RandomAgent, DRLReshapeAgent, SparseAgent, Hyperparams
from ddqn_move import DDQNMoveAgent
import os

def kwargs_to_dict(**kwargs):
    return(kwargs)




main_agent = kwargs_to_dict(
    map_name="Simple64",
    players=[sc2_env.Agent(sc2_env.Race.terran), 
             sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64),
    step_mul=48,
    disable_fog=True)

farming_agent = kwargs_to_dict(
    map_name="Simple64",
    players=[sc2_env.Agent(sc2_env.Race.terran), 
             sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64),
    step_mul=480,
    disable_fog=True)

main_agent_vs_bots = kwargs_to_dict(
    map_name="Simple64",
    players=[sc2_env.Agent(sc2_env.Race.terran), 
             sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64),
    step_mul=48,
    disable_fog=True)

ddqn_move_beacon = kwargs_to_dict(
    map_name="MoveToBeacon",
    players=[sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84, minimap=64),
        use_raw_units=True,
        use_feature_units=True,
        use_unit_counts=True,
        
        allow_cheating_layers = True    
    ),
    step_mul=16,
    # realtime=True,
    disable_fog=True,)

mining_minigame = kwargs_to_dict(
    map_name="CollectMineralsAndGas",
    players=[sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        feature_dimensions=features.Dimensions(screen=84, minimap=64),
        use_raw_units=True,
        use_feature_units=True,
        use_unit_counts=True,
        # raw_resolution=64,
        allow_cheating_layers = True            
    ),
    step_mul=32,
    disable_fog=True,
    # realtime=True,
    # visualize=True,
)