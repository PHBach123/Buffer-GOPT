import numpy as np
import copy
import torch


class BoxCreator(object):
    def __init__(self, buffer_size=2):
        self.box_list = []  
        self.buffer_list = []  
        self.buffer_size = buffer_size  

    def reset(self):
        self.box_list.clear()
        self.buffer_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def add_to_buffer(self, item):
        if len(self.buffer_list) >= self.buffer_size:
            del self.buffer_list[0]
        self.buffer_list.append(item)
        
    def remove_from_buffer(self, index):
        if 0 <= index < len(self.buffer_list):
            del self.buffer_list[index]

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        
        while len(self.box_list) < length:
            self.generate_box_size()
            #self.add_to_buffer(self.box_list[-1])
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self, index):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)
        self.buffer_list.pop(index)


class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k))

    def __init__(self, box_size_set=None, buffer_size=2):
        super().__init__(buffer_size)
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])
        self.add_to_buffer(copy.deepcopy(self.box_list[-1]))


# load data
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None, buffer_size=2):  # data url
        super().__init__(buffer_size)  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
