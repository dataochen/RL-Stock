import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float16)

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,
            self.df.loc[self.current_step, 'adjustflag'] / 10,
            self.df.loc[self.current_step, 'tradestatus'] / 1,
            self.df.loc[self.current_step, 'pctChg'] / 100,
            self.df.loc[self.current_step, 'peTTM'] / 1e4,
            self.df.loc[self.current_step, 'pbMRQ'] / 100,
            self.df.loc[self.current_step, 'psTTM'] / 100,
            self.df.loc[self.current_step, 'pctChg'] / 1e3,
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        return obs

    def _used_action(self, action):
        buyPrice=self.df.loc[self.current_step, "buyPrice"]
        shares_bought=self.df.loc[self.current_step, "shares_bought"]
        if shares_bought>0:
            #buy
            print('买入股数：'+f'{shares_bought}')
            prev_cost = self.cost_basis * self.shares_held
            additional_cost=buyPrice * shares_bought
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
        elif shares_bought<0:
            #sell
            shares_sold=-shares_sold
            print('卖出股数：'+f'{shares_sold}')
            self.balance += shares_sold * buyPrice
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * buyPrice
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self):
        # Execute one time step within the environment
        self._used_action()


        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # loop training
            # done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # profits
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = 1 if reward > 0 else -100

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'余额: {self.balance}')
        print(f'持有股数: {self.shares_held} (总卖出股数: {self.total_shares_sold})')
        print(f'持有股票的平均成本: {self.cost_basis} (总卖出金额: {self.total_sales_value})')
        print(f'净值: {self.net_worth} (最大净值: {self.max_net_worth})')
        print(f'利益: {profit}')
        return profit