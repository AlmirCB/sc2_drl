import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

def get_file_txt(file):
    txt = ""
    with open(file, "r") as rf:
        txt = rf.read()

    return txt

def parse_regex_groups(regex_grps, txt):
    ret = {}
    for r,grp in regex_grps:
        try:
            if res:=re.search(r, txt):
                ret = {n:float(v) for n,v in zip(grp,res.groups())}
        except Exception as e:
            pass
        finally:
            pass
    return ret

def eval_regex_line(reg, line):
    params = {}
    if res:=re.search(reg, line):
        params = eval(res.group())
    return params

def parse_reward_shaping(txt):
    r = r"Reward Shaping:\n ({.*})"
    params = {}
    if res:=re.search(r, txt):
        params = res.groups()[0]
    return params

def parse_hyperparams(txt):
    hyperparams_regex = (
        (r"Hyperparams: BATCH_SIZE=(.*), GAMMA=(.*), EPS_START=(.*), EPS_END=(.*), EPS_DECAY=(.*), TAU=(.*), LR=(.*), MEMORY_LEN=(.*)",
        ("batch_size", "gamma", "eps_start", "eps_end", "eps_decay", "tau", "lr", "memory_len")),
        (r"Hyperparams: BATCH_SIZE=(.*), GAMMA=(.*),              EPS_START=(.*), EPS_END=(.*), EPS_DECAY =(.*),              TAU =(.*), LR =(.*), MEMORY_LEN =(.*)",
        ("batch_size", "gamma", "eps_start", "eps_end", "eps_decay", "tau", "lr", "memory_len"))
    )
    return parse_regex_groups(hyperparams_regex, txt)

def parse_description(txt):
    r = r"Training Description: (.*)"
    
    if res:=re.search(r, txt):
        return res.groups()[0]

def parse_training_begin(txt):
    r = r"Environment is ready\n"
    if res:=re.search(r, txt):
        return(res.end())

def get_training_df(txt):
    txt = txt[parse_training_begin(txt):]
    log_lines = txt.split("\n")
    params = {}
    eps=0
    for line in log_lines:
        # print(line)
        if res := parse_line(line):
            eps = int(res.pop("eps", eps))
            params.setdefault(eps, {}).update(res)
        # else:
        #     print(line)

    df = pd.DataFrame.from_records([{**{"eps":k}, **v} for k,v in params.items()])
    return df

def parse_line(line):
    res = parse_line_1(line) or parse_line_2(line) or parse_line_3(line) or parse_codecarbon(line)
    return res

def parse_line_1(line):
    r = r"{'eps': (.*)}"
    params = {}
    if res:=re.search(r, line):
        params = eval(res.group())
    return params

def parse_line_2(line):
    rgx_grp = (
        (r"Eps (\d*) finished with reward (.*), new mean = (.*) in (\d*) steps",
         ("eps", "agent_reward", "mean", "agent_steps")),
        (r"Eps (\d*) finished with reward (.*), new mean = (.*)",
         ("epis", "agent_reward", "mean"))
    )
    return parse_regex_groups(rgx_grp, line)

def parse_line_3(line):
    rgx_grp = (
        (r"Episode (\d*) finished after (\d*) game steps. Outcome: \[(\d*)\], reward: \[(\d*)\], score: \[(\d*)\]",
         ('eps', 'full_steps', "outcome", "last_reward", "score")),)
    return parse_regex_groups(rgx_grp, line)

def parse_codecarbon(line):
    rgx_grp = (
        (r'.*EmissionsData\(.*cpu_energy=(.*), gpu_energy=(.*), ram_energy=(.*), energy_consumed=(.*), country_name.*\)',
        ('cpu', 'gpu', 'ram', 'total')),)
    return parse_regex_groups(rgx_grp, line)

def parse_training_log(f):
    txt = get_file_txt(f)
    res = {}
    res.setdefault("description", parse_description(txt))
    res.setdefault("reward_shaping", parse_reward_shaping(txt))
    res.setdefault("hyperparams", parse_hyperparams(txt))
    
    df = get_training_df(txt)
    eps_start = res['hyperparams']['eps_start']
    eps_end = res['hyperparams']['eps_start']
    eps_decay = res['hyperparams']['eps_decay']
    try:
        def get_epsilon(steps):
            return ((eps_end + (eps_start - eps_end)) * math.exp(-1. * (steps/eps_decay)))
        df['steps_count'] = df['steps'].cumsum()
        df['epsilon'] = df['steps_count'].apply(lambda x: get_epsilon(x))
        df = df.rename(columns={"r": "reward"})
    except Exception as e:
        print("Error when operating with dataframe {e}")

    res.setdefault("df", df)
    return res

def get_execution_command(f, eps=None, save=False):
    return f"python train.py --playing=True --load_path={os.path.split(f)[0]} --load_eps={eps} --save={save} --realtime"

def normalize_episode_reward(r, mean = 10.635635961344484, std = 7.869382086382208):
    # Normalize a reward that has yet been registered without normalization
    rew = (r / std) - (421 * (mean / std)) # 421 = n_steps per episode
    return rew

def denormalize_episode_reward(r, mean = 10.635635961344484, std = 7.869382086382208):
    # Normalize a reward that has yet been registered without normalization
    rew = (r * std) + (421 * (mean / std)) # 421 = n_steps per episode
    return rew

def get_mean_n(serie:pd.Series, n:int) ->pd.Series:
    """Get the mean over n rows, it is used for getting the mean reward over n eps

    Args:
        serie (pd.Series): Contains the values
        n (int): The number of values to get the mean

    Returns:
        pd.Series: Same lenght as input serie.
    """
    return serie.rolling(n).mean()

def mean_report(df, mean_values = [1], normalized=False, mean_lims=None):
    # TODO: Use always same colors
    rew = df["reward"]
    if normalized:
        rew = df["reward"].apply(lambda x: normalize_episode_reward(x))

    tags = []
    if mean_lims:
        plt.ylim(mean_lims)
    for i in mean_values:
        tags.append(f'mean_{i}')
        plt.plot(get_mean_n(rew, i))
    
    plt.legend(tags)
    plt.show()

def plot_reward_eps_comparation(df, n=1, normalize=True, mean_lims=None):
    r = df["reward"]
    if normalize:
        r = normalize_episode_reward(r)
    t = range(len(df))
    data1 = get_mean_n(r, n)
    data2 = df["epsilon"]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of episodes')
    ax1.set_ylabel(f'Mean Reward of {n} episodes', color=color)
    if mean_lims:
        print(mean_lims)
        ax1.set_ylim(mean_lims)    
    
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('epsilon', color=color)
    ax2.set_ylim(0,1)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def code_carbon_analysis(df, title=None):
    cc_df = df[['cpu', 'gpu', 'ram', 'total']]
    print(cc_df[-1:])
    ax = cc_df.plot(lw=2, title=title)
    ax.set_xlabel('Number of episodes')
    ax.set_ylabel('KWh')