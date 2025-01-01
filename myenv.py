import gym
from gym import spaces
import numpy as np
import heapq
import random
import math
from space import Space
from mycreator import RandomBoxCreator
from space import Space
from bpp_gen import bpp_generator

class BinPacking3DEnv(gym.Env):
    def __init__(self, bin_size=(20, 20, 20), max_items=60):
        super(BinPacking3DEnv, self).__init__()

        self.bin_size = np.array(bin_size)
        self.max_items = max_items
        self.space = Space(*self.bin_size)
        self.area = int(self.bin_size[0] * self.bin_size[1])
        self.act_len = self.area * 6
        self.obs_len = self.area *(1+3)
        self.action_space = gym.spaces.Discrete(self.act_len)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height, shape=(self.obs_len,))
        self.steps = 0
        self.box_list = bpp_generator(max_items, bin_size[0], bin_size[1], bin_size[2])
        self.rotated_box = self.cur_box()
        print(self.space.plain_size)

        # 初始化环境状态
        self.reset()

    def reset(self):
        self.box_list.clear()
        self.space = Space(*self.bin_size)
        self.box_list = bpp_generator(self.max_items, self.bin_size[0], self.bin_size[1], self.bin_size[2])
        self.steps = 0
        #self.cur_box = self.box_list[self.steps]
        return self.cur_observation()

    def cur_box(self):
            return self.box_list[self.steps]

    def get_box_plain(self):
        x_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.cur_box()[0]
        y_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.cur_box()[1]
        z_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.cur_box()[2]
        return (x_plain, y_plain, z_plain)

    def cur_observation(self):
        hmap = self.space.plain
        size = self.get_box_plain()
        return np.reshape(np.stack((hmap,  *size)), newshape=(-1,))

    def get_box_ratio(self):
        coming_box = self.cur_box()
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def get_possible_position(self, box, plain=None):
        x = box[0]
        y = box[1]
        z = box[2]

        if plain is None:
            plain = self.space.plain

        length = self.space.plain_size[0]
        width = self.space.plain_size[1]

        action_mask = np.zeros(shape=(length, width), dtype=np.int32)

        for i in range(length - x + 1):
            for j in range(width - y + 1):
                if self.space.check_box(plain, x, y, i, j, z) >= 0:
                    action_mask[i, j] = 1

        return action_mask

    def rotate_box(self, rotation):
        coming_box = self.cur_box()
        if rotation == 0:
            x = coming_box[0]
            y = coming_box[1]
            z = coming_box[2]
        elif rotation == 1:
            x = coming_box[0]
            y = coming_box[2]
            z = coming_box[1]
        elif rotation == 2:
            x = coming_box[1]
            y = coming_box[0]
            z = coming_box[2]
        elif rotation == 3:
            x = coming_box[1]
            y = coming_box[2]
            z = coming_box[0]
        elif rotation == 4:
            x = coming_box[2]
            y = coming_box[0]
            z = coming_box[1]
        else:
            x = coming_box[2]
            y = coming_box[1]
            z = coming_box[0]

        return (x,y,z)

    def check_end(self):
        if self.steps == self.max_items:
            return True
        return False

    def check_able(self):
        rotated_box1 = self.rotate_box(1)
        rotated_box2 = self.rotate_box(2)
        rotated_box3 = self.rotate_box(3)
        rotated_box4 = self.rotate_box(4)
        rotated_box5 = self.rotate_box(5)
        action_mask = self.get_possible_position(self.cur_box())
        action_mask1 = self.get_possible_position(rotated_box1)
        action_mask2 = self.get_possible_position(rotated_box2)
        action_mask3 = self.get_possible_position(rotated_box3)
        action_mask4 = self.get_possible_position(rotated_box4)
        action_mask5 = self.get_possible_position(rotated_box5)
        if action_mask.sum() == 0 and action_mask1.sum() == 0 and action_mask2.sum() == 0 and action_mask3.sum() == 0 and action_mask4.sum() == 0 and action_mask5.sum() == 0:
            return False
        return True


    def step(self, action):
        is_end = self.check_end()
        is_able = self.check_able()
        reward = 0

        if is_end:
            done = True
            return self.cur_observation(), reward, done, {}

        rotation = action % 6
        idx = action // 6
        self.rotated_box = self.rotate_box(rotation)
        succeeded = self.space.drop_box(self.rotated_box, idx, False)

        if not succeeded:
            reward = 0.00
            done = False
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'mask': np.ones(shape=self.act_len)}
            return self.cur_observation(), reward, done, {}

        print(self.steps)
        box_ratio = self.get_box_ratio()
        plain = self.space.plain
        reward += box_ratio * 10
        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['ratio'] = self.space.get_ratio()
        # info['mask'] = self.get_possible_position().reshape((-1,))
        self.steps += 1
        is_able = self.check_able()
        is_end = self.check_end()

        if is_end:
            done = True
            return self.cur_observation(), reward, done, {}

        while not is_able:
            self.steps += 1
            is_end = self.check_end()
            if is_end:
                done = True
                return self.cur_observation(), reward, done, {}
            is_able = self.check_able()

        return self.cur_observation(), reward, done, {}

    def render(self, mode='human'):
        # 可以实现必要的可视化功能
        pass

