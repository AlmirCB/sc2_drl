from absl import app, logging, flags
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop
from pysc2.env import sc2_env, run_loop, environment
import os

from utils import get_codecarbon_tracker
from game_configs import main_agent, mining_minigame, farming_agent
from main_agent import RandomAgent, DRLReshapeAgent, SparseAgent
from networks import Hyperparams
from agent_resources import MiningAgent, MiningRandomAgent, MiningDRLAgent, MiningJustFarmAgent, DebuggingAgent, MiningScriptedAgent
from ddqn_move import DDQNMoveAgent
from agent_farming_64 import MiningDRLAgent64, MiningRandomAgent64
# from full_agent import FullRandomAgent#, FullDRLAgent
# from full_agent_sep import MainDRLAgent, MainDRLAgent, FarmingDRLAgent, IdleAgent, FullRandomAgent, FarmingRandomAgent, ReducedFarmingDRLAgent
from full_agent_sep import MultiDRLAgent, ReducedRandomAgent, FullReducedDRLAgent

FLAGS = flags.FLAGS
flags.DEFINE_bool("playing", True, "Loads the agent in playing mode")
flags.DEFINE_bool("save", True, "Create logs and save network states")
flags.DEFINE_bool("debug", False, "Display debugging messages")
flags.DEFINE_string("load_path", None, "Path to saved agent execution folders")
flags.DEFINE_integer("load_eps", 0, "Episode to be loaded")
flags.DEFINE_bool("realtime", False, "Play in real time")
flags.DEFINE_string("description", None, "Descripción del entrenamiento que se va a lanzar")

ENV_CONF = main_agent


def main(unused_argv):
	description = None
	kwargs = {
		"playing": FLAGS.playing,
		"description": FLAGS.description or description,
		"save": FLAGS.save,
		"debug": FLAGS.debug,
	}
	if rt := FLAGS.realtime:
		ENV_CONF.setdefault("realtime", rt)

	agent1 = FullReducedDRLAgent(Hyperparams(eps_decay=100000, tau=0.7, lr=1e-5, eps_end=0.001), **kwargs)
	agent2 = ReducedRandomAgent()
	
	tracker = None
	if FLAGS.save:
		log_folder = agent1.execution_folder
		if not os.path.isdir(log_folder):
			os.makedirs(log_folder)
		
		if hasattr(agent1, "set_tracker"):
			tracker = get_codecarbon_tracker(log_folder)
			tracker.start()
			agent1.set_tracker(tracker)
	
		logging.get_absl_handler().python_handler.use_absl_log_file(log_dir=log_folder, program_name=agent1.name)

	agent1.log_config()
	logging.info(description)

	if (path:= FLAGS.load_path) and (eps:=FLAGS.load_eps):
		# TODO: This method should load trainning params in case we want to go on trainning
		agent1.load(path, eps)
	
	
	try:
		with sc2_env.SC2Env(**ENV_CONF) as env:
			run_loop.run_loop([agent1, agent2], env, max_episodes=1200)
	except KeyboardInterrupt:
		pass
	finally:
		emissions = tracker.stop() if tracker else None
		print(f"Emissions: {emissions}")



if __name__ == "__main__":
	app.run(main)
