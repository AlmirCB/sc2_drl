import pandas as pd
import pickle
import os
import numpy as np
from pysc2.lib.named_array import  NamedDict, NamedNumpyArray
from pysc2.lib import units
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


pd.set_option('display.max_columns', None) # Defaults to 20


observation_path = "../saves/observations/obs_1270.pkl"
with open(observation_path, 'rb') as f:
    obs = pickle.load(f)
def get_keys(named_obj):
    "Get keys from NamedDict or NamedNumpyArray from pysc2.lib.named_array"
    if type(named_obj) == NamedNumpyArray:
        return [k for k in named_obj._index_names[1].keys()]
    else:
        return [k for k in named_obj.keys()]
    
def get_df(named_array):
    keys = get_keys(named_array)
    df = pd.DataFrame.from_records(named_array, columns=keys)
    return df

def get_units_by_type(df, unit_type):
    return df[df['unit_type'] == unit_type]

def get_units_by_side(df, playerrelative):
    return df[df['alliance'] == playerrelative]

def get_position_list(df):
    return [v for v in zip(df['x'], df['y'])]

def get_distances(pos_list, point, max_dist):
    dist = np.linalg.norm(np.array(pos_list) - np.array(point), axis=1)
    res = [x for x in zip(pos_list,dist)]
    res.sort(key=lambda x: x[1])
    res = [x for x in filter(lambda x: x[1]<max_dist, res)]
    return res