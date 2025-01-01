import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from gym.envs.registration import register
from MCtree import MCTree
from MCtree_node import PutNode


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = self.fc2(x)
        log_probs = F.log_softmax(probs, dim=1)
        probs = torch.exp(log_probs)
        return probs


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# 用 MCTS 来生成策略并指导 PPO 选择动作
def train_with_mcts(env, agent, num_episodes, max_steps, giveup_action, zeta=1, sim_times=50):
    return_list = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        mcts_tree = MCTree(env, state, [], nmodel=agent.actor, search_depth=10)  # 假设 nmodel 为 PPO 的 actor

        for step in range(max_steps):
            if done:
                break

            # 使用MCTS生成当前状态下的策略
            policy = mcts_tree.get_policy(sim_times, zeta)
            action = mcts_tree.sample_action(policy)  # 选择一个基于MCTS的动作

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 收集经验
            agent.update({
                'states': [state],
                'actions': [action],
                'rewards': [reward],
                'next_states': [next_state],
                'dones': [done]
            })

            state = next_state

        return_list.append(total_reward)

    return return_list


# 注册环境
def registration_envs():
    register(id='bbp-v0', entry_point='myenv:BinPacking3DEnv')


if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 0.01
    num_episodes = 5000
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cpu")

    registration_envs()
    env_name = "bbp-v0"
    env = gym.make(env_name)

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_steps = 5 * action_dim
    giveup_action = action_dim - 1

    # 初始化 PPO 智能体
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    # 使用 MCTS 结合 PPO 训练智能体
    return_list = train_with_mcts(env, agent, num_episodes, max_steps, giveup_action)

    # 保存模型
    torch.save(agent.actor.state_dict(), 'ppo_actor.pth')
    torch.save(agent.critic.state_dict(), 'ppo_critic.pth')

    # 绘制训练结果
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO with MCTS on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO with MCTS on {}'.format(env_name))
    plt.show()

