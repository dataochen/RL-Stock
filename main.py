import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

models_dir = 'models/IPO'
TIMESTEPS = 10000
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


#学习
def learn(stock_code,start_times):
    model=''
    if start_times:
        model_path=f'{models_dir}/{TIMESTEPS*start_times}'
        model = PPO2.load(model_path,get_env(stock_code))
    else:
        model=PPO2(MlpPolicy, get_env(stock_code), verbose=0, tensorboard_log='.\log')

    for i in range(start_times, start_times+3):
        model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
         #保存训练过的模型
        model.save(f'{models_dir}/{TIMESTEPS*i}')

def get_env(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    # The algorithms require a vectorized environment to run
    return DummyVecEnv([lambda: StockTradingEnv(df)])

def re_learn(stock_code):
    learn(stock_code)

#预判 todo
def show(stock_code,start_times):
    model_path=f'{models_dir}/{TIMESTEPS*start_times}'
    env=get_env(stock_code)
    obs = env.reset()
    model = PPO2.load(model_path,env)
    episodes = 2
    for ep in range(episodes):
        obs = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()



def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)





if __name__ == '__main__':
#     multi_stock_trade()
#     test_a_stock_trade('sh.600036')
#     ret = find_file('./stockdata/train', '600036')
#     print(ret)
#     re_learn('sh.600030')
    #learn('sh.600030',2)
    show('sh.600030',4)

