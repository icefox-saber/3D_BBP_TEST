import numpy as np
import copy
import torch
from bpp_gen import bpp_generator

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        pass

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

class RandomBoxCreator(BoxCreator):
    def __init__(self,sample_num = 20, length = 100, width = 100, height = 100):
        self.box_list = bpp_generator(sample_num, length, width, height)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])
