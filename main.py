import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from gym.envs.registration import register
from models import PolicyNet
from models import ValueNet
import argparse
import ast



class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 使用 SGD 优化器
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, fail):

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        try:
            probs = self.actor(state)
            # 检查概率是否为nan
            if torch.isnan(probs).any():
                raise ValueError("Probabilities returned NaN")
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

        except ValueError as e:
            print(f"Error in computing action: {e}")
            return fail

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def registration_envs():
    register(
        id='bbp-v0',  # Format should be xxx-v0, xxx-v1
        entry_point='myenv:BinPacking3DEnv',  # Expalined in envs/__init__.py
    )

def parse_tuple(arg):
    """将字符串解析为元组"""
    return tuple(ast.literal_eval(arg))

def parse_boxlist(item):
    """将字符串解析为一个元组列表"""
    return [tuple(ast.literal_eval(item))]

if __name__ == "__main__":
    # 初始化参数
    actor_lr = 1e-4  # 学习率
    critic_lr = 0.01  # 学习率
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cpu")
    bin_size = (20,20,20)

    registration_envs()
    env_name = "bbp-v0"

    parser = argparse.ArgumentParser(description='Process some parameters.')

    # 添加参数
    parser.add_argument('--container_size', type=parse_tuple, default=(20, 20, 20),
                        help='Container size as a tuple (x, y, z). Default is (20, 20, 20).')
    parser.add_argument('--max_items', type=int, default=60,
                        help='Maximum number of items. Default is 60.')
    parser.add_argument('--boxlist', type=parse_boxlist, default=None,
                        help='List of boxes as a list of tuples [(x1, y1, z1), ...]. Default is None.')

    # 解析参数
    args = parser.parse_args()

    env = gym.make(env_name, bin_size=args.container_size, max_items=args.max_items, boxlist=args.boxlist)

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    giveup_action = action_dim - 1
    fail = action_dim
    max_steps = 5*action_dim
    # 初始化智能体
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    # 开始训练
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes, max_steps, fail, giveup_action)

    # 保存模型
    torch.save(agent.actor.state_dict(), 'ppo_actor.pt')
    torch.save(agent.critic.state_dict(), 'ppo_critic.pt')

    # 绘制训练结果
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()