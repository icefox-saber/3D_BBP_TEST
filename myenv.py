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
    def __init__(self, bin_size=(20, 20, 20), max_items=60, boxlist = None):
        super(BinPacking3DEnv, self).__init__()

        self.bin = np.zeros((bin_size[0], bin_size[1], bin_size[2]))
        self.bin_size = np.array(bin_size)
        self.max_items = max_items
        self.space = Space(*self.bin_size)
        self.area = int(self.bin_size[0] * self.bin_size[1])
        self.act_len = self.area * 6 + 1
        self.obs_len = self.area *(1+3)
        self.action_space = gym.spaces.Discrete(self.act_len)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height, shape=(self.obs_len,))
        self.steps = 0
        self.box_set = set()

        if boxlist is None:
            self.set = False
        else:
            self.set = True

        if self.set:
            self.box_list = boxlist
        else:
            self.box_list = bpp_generator(self.max_items, self.bin_size[0], self.bin_size[1], self.bin_size[2])
        self.rotated_box = self.cur_box()
        # 初始化环境状态
        self.reset()

    def reset(self):
        self.space = Space(*self.bin_size)
        self.box_set = set()
        self.steps = 0

        if self.set is False:
            self.box_list.clear()
            self.box_list = bpp_generator(self.max_items, self.bin_size[0], self.bin_size[1], self.bin_size[2])

        while np.sum(self.cur_box()) == 0:
            self.steps += 1

        return self.cur_observation()

    def cur_box(self):
        if self.check_end():
            return (2,2,2)
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

    def get_box_ratio(self, lx, ly):
        coming_box = self.cur_box()
        container = []
        rx = min(lx+coming_box[0]+1, self.bin_size[0]-1)
        lx = max(0,lx - 1)
        container.append(rx - lx)
        uy = min(ly+coming_box[1]+1, self.bin_size[1]-1)
        ly = max(0, ly - 1)
        container.append(uy - ly)
        container.append(self.cur_box()[2])
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (container[0] * container[1] * container[2])

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
        reward = 0

        if is_end:
            print(self.space.get_ratio())
            done = True
            return self.cur_observation(), reward, done, {}

        while np.sum(self.cur_box()) == 0:
            self.steps += 1

        if action == self.area * 6:
            self.steps += 1
            reward = 0.00
            done = False
            info = {'counter': self.box_set, 'ratio': self.space.get_ratio(),
                    'mask': np.ones(shape=self.act_len)}
            return self.cur_observation(), reward, done, info

        rotation = action % 6
        idx = action // 6
        lx = idx % self.bin_size[0]
        ly = idx // self.bin_size[0]
        self.rotated_box = self.rotate_box(rotation)
        succeeded = self.space.drop_box(self.rotated_box, idx, False)

        if not succeeded:
            reward = 0.00
            done = False
            info = {'counter': self.box_set, 'ratio': self.space.get_ratio(),
                    'mask': np.ones(shape=self.act_len)}
            return self.cur_observation(), reward, done, info

        plain = self.space.plain
        self.bin[lx:lx + self.cur_box()[0]+1, ly:ly + self.cur_box()[1]+1, plain[lx][ly]:plain[lx][ly] + self.cur_box()[2]+1] = 1
        coming_box = self.cur_box()
        container = []
        rx = min(lx + coming_box[0] + 1, self.bin_size[0] - 1)
        new_lx = max(0, lx - 1)
        container.append(rx - lx)
        uy = min(ly + coming_box[1] + 1, self.bin_size[1] - 1)
        new_ly = max(0, ly - 1)
        container_size = np.sum(self.bin[new_lx:rx+1, new_ly:new_ly + uy+1, 0:plain[lx][ly] + self.cur_box()[2]]+1)
        box_ratio = coming_box[0]*coming_box[1]*coming_box[2]/container_size
        reward += box_ratio * 5
        done = False
        self.box_set.add(self.steps)
        info = dict()
        info['counter'] = self.box_set
        ratio = self.space.get_ratio()
        info['ratio'] = ratio
        info['mask'] = self.get_possible_position(self.cur_box()).reshape((-1,))
        self.steps += 1
        is_able = self.check_able()
        is_end = self.check_end()


        if is_end:
            done = True
            info = dict()
            info['counter'] = self.box_set
            ratio = self.space.get_ratio()
            info['ratio'] = ratio
            info['mask'] = self.get_possible_position(self.cur_box()).reshape((-1,))
            return self.cur_observation(), reward, done, info


        while np.sum(self.cur_box()) == 0 or not is_able:
            self.steps += 1
            is_end = self.check_end()
            if is_end:
                done = True
                info = dict()
                info['counter'] = self.box_set
                ratio = self.space.get_ratio()
                info['ratio'] = ratio
                info['mask'] = self.get_possible_position(self.cur_box()).reshape((-1,))
                return self.cur_observation(), reward, done, info
            while np.sum(self.cur_box()) == 0:
                self.steps += 1
            is_able = self.check_able()

        return self.cur_observation(), reward, done, info

    def render(self, mode='human'):
        # 可以实现必要的可视化功能
        pass

