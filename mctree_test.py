import gym
import copy
import argparse
import ast
import math
import pandas as pd

from gym import register

from MCtree import MCTree
import numpy as np
import time

import sys

sys.path.append("..")
from models import nnModel

def test(box_size_list, env, obser, nmodel, box_mask):
    sim_env = copy.deepcopy(env)
    size_idx = len(box_size_list)
    action_list = []
    box_size_distribution = dict()
    box_num = 0
    sum_reward = 0
    print("length: ", size_idx)
    search_depth = 50
    rollout_length = 10
    simulation_times = 100
    box_set = set()
    new_box_mask = box_mask
    mctree = MCTree(sim_env, obser, box_size_list, nmodel=nmodel, search_depth=search_depth,
                    rollout_length=rollout_length)
    while True:
        # show some information
        #print(box_size_list[:10])
        # print(sim_env.space.plain)
        # MCTS simulation
        pl = mctree.get_policy(simulation_times, zeta=1e-5)
        action = mctree.sample_action(pl)

        #assert sim_env.cur_box() == box_size_list[0]

        obser, r, done, dt = sim_env.step(action)
        if r > 0:
            print(dt['counter'])
            print(dt['ratio'])
        sum_reward += r

        if done:
            dt['reward'] = sum_reward
            index_array = np.array(list(sim_env.box_set))  # 将集合转换为 NumPy 数组
            box_mask[index_array] = 0
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
            return new_box_mask, dt['counter']

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

def parse_tuple(arg):
    """将字符串解析为元组"""
    return tuple(ast.literal_eval(arg))

def parse_boxlist(item):
    """将字符串解析为一个元组列表"""
    return [tuple(ast.literal_eval(item))]

def read_csv(path):
    df = pd.read_csv(path)
    data_list = []
    for row in df.values:
        for _ in range(row[5]):
            tmp = (row[2], row[3], row[4])
            data_list.append(tmp)
    return data_list

def deal_with_box(Bin, Boxes):
    new_boxes = []
    z1, z2, z3 = Bin[0] / 20, Bin[1] / 20, Bin[2] / 20
    for item in Boxes:
        item_1 = (int(math.ceil(item[0] / z1)), int(math.ceil(item[1] / z2)), int(math.ceil(item[2] / z3)))
        new_boxes.append(item_1)
    return np.array(new_boxes)

def get_ratio(bin, box):
    return box[0]*box[1]*box[2]/(bin[0]*bin[1]*bin[2])
if __name__ == '__main__':
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

    # 输出解析后的参数
    """
    print("Container Size:", args.container_size)
    print("Max Items:", args.max_items)
    print("Box List:", args.boxlist)
    """
    registration_envs()
    env_name = "bbp-v0"
    # 解析输入字符串为数组
    bins = [(35, 23, 13), (37, 26, 13), (38, 26, 13), (40, 28, 16), (42, 30, 18), (42, 30, 40), (52, 40, 17), (54, 45, 36)]
    box_size_list = read_csv('task3.csv')
    hidden_dim = 128
    box_mask = np.ones(len(box_size_list),dtype=int)
    it = 0
    for bin in bins:
        it += 1
        print("container %d" % it)
        boxes = deal_with_box(bin,box_size_list)*box_mask[:, np.newaxis]
        boxes_list = [tuple(item) for item in boxes]
        env = gym.make(env_name, bin_size=args.container_size, max_items=len(box_size_list), boxlist=boxes_list)
        obser = env.reset()
        container_size = env.bin_size
        nmodel = nnModel('ppo_actor.pt', 'ppo_critic.pt', container_size, hidden_dim)
        box_mask, box_set = test(env.box_list[:4], env, obser, nmodel, box_mask)
        index_array = np.array(list(box_set))
        ratio = 0
        for i in index_array:
            ratio += get_ratio(bin, box_size_list[i])
        print("the ratio: %" % ratio)

    """
    args_list = list()
    args_list.append([100, None, -1])
    result = compare_test(env, args_list, 5, hidden_dim)
    for (key, value) in result.items():
        print(value[:, 0])
        print(value[:, 1])
        meanv = value.mean(axis=-2)
        print(meanv)
        print("avg_time_per_item", meanv[-1] / meanv[1])
    """
#python mctree_test.py --container_size=(20,20,20) --max_items=100 --boxlist="[(15,15,15), (25,25,25)]"