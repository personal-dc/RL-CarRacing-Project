import argparse

import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn

import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)

'''
Wrapper env for testing
Have only changed so that state representation changed
Reward function is the original one so that we can compare across different models
'''
class Env():

    def __init__(self):
        self.env = gym.make('CarRacing-v3', render_mode = 'rgb_array')
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * 4
        return np.array(self.stack)

    # don't changed reward function
    def step(self, action):
        total_reward = 0
        for i in range(8):
            img_rgb, reward, die, trunc, _ = self.env.step(action)
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        rgb = rgb[0] if len(rgb) == 2 else rgb
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

'''
Load trained model
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

'''
Testing agent
'''
class Agent():

    def __init__(self, changed_reward):
        self.net = Net().float().to(device)
        self.changed_reward = changed_reward

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        dir = 'param/self_changed_reward.pkl' if self.changed_reward else 'param/self_unchanged_reward.pkl'
        self.net.load_state_dict(torch.load(dir, map_location=torch.device('cpu') ))



def run_test(changed_reward = True):
    agent = Agent(changed_reward = changed_reward)
    agent.load_param()

    env = Env()

    data_dir = 'data/self_changed_reward_test.txt' if changed_reward else 'data/self_unchanged_reward_test.txt'
    

    testing_records = []
    state = env.reset()
    for i_ep in range(1000):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            score += reward
            state = state_
            if done or die:
                break

        # print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        print(f'Ep {i_ep} Score: {round(score, 2)} Frames: {t}')
        f = open(data_dir, 'a')
        f.write(f'Ep {i_ep} Score: {round(score, 2)} Frames: {t}')
        f.write('\n')
                
        testing_records.append((i_ep, score, t))


if __name__ == "__main__":
    run_test()