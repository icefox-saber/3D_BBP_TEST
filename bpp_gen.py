import heapq
import random
import math
import numpy as np

def neg_volume(item):
    return -1*item[0]*item[1]*item[2]

def rand_reverse(item):
    flag = random.randint(0, 1)
    if flag:
        return item
    else:
        a = -1
        b = -1
        cnt = 0
        while a == b:
            a = random.randint(0, 2)
            b = random.randint(0, 2)
            cnt += 1
            if cnt == 100:
                print("Generator Error.")
                item2 = (-1, -1, -1)
                return item2
        item1 = [item[0], item[1], item[2]]
        item1[a], item1[b] = item1[b], item1[a]
        res = (item1[0], item1[1], item1[2])
        return res

def bpp_generator(sample_num, length, width, height):
    max_length = length // 2
    max_width = width // 2
    max_height = height // 2
    box_set = []

    for i in range(sample_num+1):
        random_length = random.randint(0, max_length)
        random_width = random.randint(0, max_width)
        random_height = random.randint(0, max_height)
        box_set.append((2+random_length, 2+random_width, 2+random_height))

    return box_set



"""
def bpp_generator(sample_num, length, width, height):
    total_volume = -1* length * width * height;
    heap_item = [(total_volume, (length, width, height))]
    while len(heap_item) < sample_num:
        largest_item = heapq.heappop(heap_item)[1]
        axis = -1
        high_1 = -1
        axis = 0
        for i11 in range(0,3):
            if largest_item[i11]>largest_item[axis]:
                axis = i11
        high_1 = largest_item[axis]
            
        pos = random.randint(high_1//3, high_1*2//3)
        item1 = (-1, -1, -1)
        item2 = (-1, -1, -1)
        if axis == 0:
            item1 = (pos, largest_item[1], largest_item[2])
            item2 = (largest_item[0]-pos, largest_item[1], largest_item[2])
        elif axis == 1:
            item1 = (largest_item[0], pos, largest_item[2])
            item2 = (largest_item[0], largest_item[1]-pos, largest_item[2])
        elif axis == 2:
            item1 = (largest_item[0], largest_item[1], pos)
            item2 = (largest_item[0], largest_item[1], largest_item[2]-pos)
        item1 = rand_reverse(item1)
        item2 = rand_reverse(item2)
        item1 = (neg_volume(item1), item1)
        item2 = (neg_volume(item2), item2)
        heapq.heappush(heap_item, item1)
        heapq.heappush(heap_item, item2)
    res_item = [item[1] for item in heap_item]
    return res_item
"""




"""
def main():
    list1 = bpp_generator(20, 100, 100, 150)
    print(list1)
    print(len(list1))
    volumn1 = 0
    for item in list1:
        volumn1 += item[0]*item[1]*item[2]
    print(volumn1)

if __name__ == "__main__":
    main()
"""
