import gym
import copy

from gym import register

from MCtree import MCTree
import numpy as np
import time

import sys

sys.path.append("..")
from models import nnModel

def test(box_size_list, env, obser, simulation_times, search_depth, rollout_length, nmodel):
    sim_env = copy.deepcopy(env)
    size_idx = len(box_size_list)
    action_list = []
    box_size_distribution = dict()
    box_num = 0
    sum_reward = 0
    print("length: ", size_idx)
    search_depth = 50
    rollout_length = 10

    mctree = MCTree(sim_env, obser, box_size_list, nmodel=nmodel, search_depth=search_depth,
                    rollout_length=rollout_length)
    while True:
        # show some information
        print(box_size_list[:10])
        # print(sim_env.space.plain)
        # MCTS simulation
        pl = mctree.get_policy(simulation_times, zeta=1e-5)
        action = mctree.sample_action(pl)

        #assert sim_env.cur_box() == box_size_list[0]
        obser, r, done, dt = sim_env.step(action)
        print(dt['ratio'])
        sum_reward += r
        if done:
            dt['reward'] = sum_reward
            # print('---------------------')
            # print(dt)
            # print('---------------------')
            # print(action_list)
            # print('---------------------')
            # print(sim_env.space.plain)
            # print('---------------------')
            for (key, value) in box_size_distribution.items():
                box_size_distribution[key] = value / box_num
            # print(box_size_distribution)
            # print('---------------------')
            return [dt['ratio'], dt['counter'], dt['reward']]

        # fetch new box
        assert size_idx <= len(env.box_list)
        next_box = copy.deepcopy(env.box_list[size_idx])
        size_idx += 1
        # update dis
        # tribution
        box_num += 1
        new_put_box = tuple(box_size_list[0])
        if new_put_box not in box_size_distribution:
            box_size_distribution[new_put_box] = 0
        box_size_distribution[new_put_box] += 1
        # update action
        action_list.append(action)
        # to next node
        mctree.succeed(action, next_box, obser)


def compare_test(env, args_list, times=5, hidden_dim = 128):
    result = dict()
    case_num = len(args_list)
    print("Case number: %d" % times)
    #nmodel = nnModel('../pretrained_models/default_cut_2.pt', args)
    box_list = env.box_list
    container_size = env.bin_size

    nmodel = nnModel('ppo_actor.pt', 'ppo_critic.pt', container_size, hidden_dim)
    for i in range(times):
        print('Case %d' % (i + 1))
        obser = env.reset()
        next_box_size_list = env.box_list
        # print(len(next_box_size_list))
        for j in range(case_num):
            if j not in result:
                result[j] = []
            arg = args_list[j]
            print(arg)
            start = time.time()
            ratio, counter, reward = test(next_box_size_list[:4], env, obser, *arg, nmodel)
            end = time.time()
            result[j].append([ratio, counter, reward, end - start])
        print('//////////////////////////////////////////////////')
    for (key, value) in result.items():
        result[key] = np.array(value)
    return result

def registration_envs():
    register(
        id='bbp-v0',  # Format should be xxx-v0, xxx-v1
        entry_point='myenv:BinPacking3DEnv',  # Expalined in envs/__init__.py
    )

if __name__ == '__main__':
    registration_envs()
    env_name = "bbp-v0"
    env = gym.make(env_name)

    args_list = list()
    args_list.append([100, None, -1])
    hidden_dim = 128
    result = compare_test(env, args_list, 5, hidden_dim)
    for (key, value) in result.items():
        print(value[:, 0])
        print(value[:, 1])
        meanv = value.mean(axis=-2)
        print(meanv)
        print("avg_time_per_item", meanv[-1] / meanv[1])