import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv
# from rlenv.StockTradingEnvMy import MyStockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

models_dir = 'models/IPO'
TIMESTEPS = 100000
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


#学习 可以依据预测的比例浮动来判断模型学习是否达标
def learn(stock_code,start_times):
    model=''
    if start_times:
        model_path=f'{models_dir}/{stock_code}/{TIMESTEPS*start_times}'
        model = PPO2.load(model_path,get_env(stock_code,True))
    else:
        model=PPO2(MlpPolicy, get_env(stock_code,True), verbose=0, tensorboard_log='.\log')

    for i in range(start_times, start_times+1000):
        model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
         #保存训练过的模型
        file_create(f'{models_dir}/{stock_code}')
        model.save(f'{models_dir}/{stock_code}/{TIMESTEPS*i}')

def get_env(stock_code,test):
    stock_file= find_file('./stockdata/train', str(stock_code)) if test else find_file('./stockdata/test', str(stock_code))
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    # The algorithms require a vectorized environment to run
    return DummyVecEnv([lambda: StockTradingEnv(df)])

def re_learn(stock_code):
    learn(stock_code)

#model验证
def model_test(stock_code,start_times):
    model_path=f'{models_dir}/{stock_code}/{TIMESTEPS*start_times}'
    env=get_env(stock_code,False)
    model = PPO2.load(model_path,env)
    day_profits = []
    obs = env.reset()
    stock_file=find_file('./stockdata/test', str(stock_code))
    df = pd.read_csv(stock_file)
    df_test = df.sort_values('date')

    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    fig, ax = plt.subplots()
    ax.plot(day_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('day')
    plt.ylabel('profit')
    ax.legend(prop=font)
    plt.show()
    #保存图片
    # plt.savefig(f'./img/{stock_code}.png')

#已持有的 下一步如何操作 推荐
def model_used_test(stock_code,start_times):
    day_profits = []
    stock_file = find_file('./stockdata/test', str(stock_code))
    df_test = pd.read_csv(stock_file)
    env = DummyVecEnv([lambda: MyStockTradingEnv(df_test)])
    obs = env.reset()
    model_path=f'{models_dir}/{stock_code}/{TIMESTEPS*start_times}'
    model = PPO2.load(model_path,env)

    for i in range(len(df_test) - 1):
        obs, rewards, done, info = env.step()
        profit = env.render()
        day_profits.append(profit)
        if done:
            break

    pre_print(model,obs)
    fig, ax = plt.subplots()
    ax.plot(day_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('天')
    plt.ylabel('利益')
    ax.legend(prop=font)
    plt.show()
    #保存图片
    # plt.savefig(f'./img/{stock_code}.png')

#预测
def predict(stock_code,start_times):
    model_path=f'{models_dir}/{stock_code}/{TIMESTEPS*start_times}'
    #???拉最后一天的 用于预测？
    # stock_file = find_file('./stockdata/pre', 'defulat')
    stock_file = find_file('./stockdata/pre', stock_code)
    df_pre = pd.read_csv(stock_file)
    env = DummyVecEnv([lambda: StockTradingEnv(df_pre)])

    obs = env.reset()
    model = PPO2.load(model_path,env)
    episodes = 20
    for ep in range(episodes):
        pre_print(model,obs)

    env.close()


def pre_print(model,obs):
    action, _states = model.predict(obs)
    action_type = action[0][0]
    prob = action[0][1]
    doWhat = ''
    if action_type < 1:
        doWhat='买入'
    elif action_type < 2:
        doWhat='卖出'
    else:
        doWhat='持有'
    #根据prob倒叙 top5的 prob/5=每股建议买入金额比（占余额）
    print(doWhat+f'比例： {prob}')

def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)

def file_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

def test(reward):
    cc= 1 if reward > 0 else -100
    print(cc)

if __name__ == '__main__':
#     multi_stock_trade()
#     re_learn('sh.600030')
    learn('sz.002230',0)
    # model_test('sz.002230',4)
    # predict('sh.600030',4)
