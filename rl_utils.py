from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes, max_steps, fail, giveup):
    return_list = []
    last_successful_agent = None
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                failed = False
                steps = 0
                while not done:
                    action = agent.take_action(state, fail)
                    if action < fail:
                        if steps < max_steps:
                            next_state, reward, done, _ = env.step(action)
                            transition_dict['states'].append(state)
                            transition_dict['actions'].append(action)
                            transition_dict['next_states'].append(next_state)
                            transition_dict['rewards'].append(reward)
                            transition_dict['dones'].append(done)
                            state = next_state
                            episode_return += reward
                            if reward > 0 :
                                steps = 0
                            else:
                                steps += 1

                        else:
                            action = giveup
                            next_state, reward, done, _ = env.step(action)
                            transition_dict['states'].append(state)
                            transition_dict['actions'].append(action)
                            transition_dict['next_states'].append(next_state)
                            transition_dict['rewards'].append(reward)
                            transition_dict['dones'].append(done)
                            state = next_state
                            episode_return += reward
                            steps = 0

                    else:
                        failed = True
                        break

                if failed:
                    if last_successful_agent is not None:
                        agent = deepcopy(last_successful_agent)
                        continue

                else:
                    return_list.append(episode_return)
                    agent.update(transition_dict)  # 尝试更新智能体
                    last_successful_agent = deepcopy(agent)  # 本轮训练成功，更新缓存的智能体

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('PPO on {bbp-v0}')
        plt.show()

        torch.save(agent.actor.state_dict(), 'ppo_actor.pt')
        torch.save(agent.critic.state_dict(), 'ppo_critic.pt')
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    last_successful_model = None
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        try:
                            agent.update(transition_dict)  # 尝试更新智能体
                            last_successful_agent = deepcopy(agent)  # 本轮训练成功，更新缓存的智能体
                        except ValueError as e:
                            print(f"Warning: NaN detected in episode {i_episode}, reverting to last successful model.")
                            # 恢复到上一轮成功的智能体
                            if last_successful_agent is not None:
                                agent = deepcopy(last_successful_agent)
                            break  # 跳出本轮训练，进行下一轮

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
