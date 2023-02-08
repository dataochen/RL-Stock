import gym
from stable_baselines3 import PPO
import os

models_dir = 'models/PPO'
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def save():
    env = gym.make('LunarLander-v2')
    env.reset()

    TIMESTEPS = 10000
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    for i in range(1, 30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='PPO')
        model.save(f'{models_dir}/{TIMESTEPS*i}')

    env.close()

def load():
    model_path = f'{models_dir}/10000.zip'

    env = gym.make('LunarLander-v2')
    env.reset()

    model = PPO.load(model_path, env=env)

    episodes = 2
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render('human')
            #print(rewards)
            if done:
                break
    env.close()
def show():

    # Create the environment
    env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

    # required before you can step the environment
    env.reset()

    for step in range(200):
    	env.render('human')
    	# take random action
    	env.step(env.action_space.sample())

    env.close()

if __name__ == '__main__':
    #save();
    load();
    #show();