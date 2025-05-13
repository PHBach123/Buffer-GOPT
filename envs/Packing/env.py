from typing import Optional

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
# from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import copy

from render import VTKRender


class PackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        k_buffer=2,
        is_render=False,
        is_hold_on=False,
        box_creator=None,
        **kwags
    ) -> None:
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        self.k_buffer = k_buffer
        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        self.box_creator = box_creator
        self.test = load_test_data

        # for rendering
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

    def _set_space(self) -> None:
        obs_len = self.area + 3  # the state of bin + the dimension of box (l, w, h)
        obs_len += self.k_placement * 6
        self.action_space = spaces.Discrete(self.k_placement)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )),
                "mask": spaces.Discrete(self.k_placement)
            }
        )

    def get_box_ratio(self, index):
        coming_box = self.box_creator.buffer_list[index]
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])


    @property
    def cur_observation(self):
        """
            get current observation and action mask
        """
        
        hmap = self.container.heightmap
        size = list(self.next_box)
        # print("buffer list: ", self.box_creator.buffer_list)
        # print("box list: ", self.box_creator.box_list)
        # print("size: ", size)
        tmp = copy.deepcopy(self.box_creator.buffer_list)
        placements, mask = self.get_possible_position(tmp)

        self.candidates = np.zeros_like(self.candidates)
        if len(placements) != 0:
            # print("candidates:")
            # for c in placements:
            #     print(c)
            self.candidates[0:len(placements)] = placements
        size.extend([size[1], size[0], size[2]])
        # print('---------------------------------------------------')
        # print('hmap: \n', hmap)
        # print("placements: \n", placements)
        # # print("mask: \n", mask)
        # print("size: ", size)

        for i in tmp:
            i.extend([i[1], i[0], i[2]])
        temp = [item for sublist in tmp for item in sublist]

        obs = np.concatenate((hmap.reshape(-1), np.array(temp).reshape(-1), self.candidates.reshape(-1)))
        mask = mask.reshape(-1)
        return {
            "obs": obs, 
            "mask": mask
        }

    @property
    def next_box(self) -> list:
        return self.box_creator.preview(self.k_buffer)[0]

    def get_possible_position(self, next_box):
        """
            get possible actions for next box
        Args:
            scheme: the scheme how to generate candidates

        Returns:
            candidate action mask, i.e., the position where the current item should be placed
        """
        if self.action_scheme == "heightmap":
            candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
        elif self.action_scheme == "EP":
            candidates, mask = self.container.candidate_from_EP(next_box, self.k_placement)
        elif self.action_scheme == "EMS":
            candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
        elif self.action_scheme == "FC": # full coordinate space
            candidates, mask = self.container.candidate_from_FC(next_box)
        else:
            raise NotImplementedError("action scheme not implemented")

        return candidates, mask 

    def idx2pos(self, idx):

        index = idx // (2*self.k_placement)
        ind = idx % (2*self.k_placement)
        # print("buffer",self.box_creator.buffer_list)
        # print("index",index)
        # print("ind",ind)
        if ind >= self.k_placement - 1:
            ind = ind - self.k_placement
            rot = 1
        else:
            rot = 0

        pos = self.candidates[ind][:3]

        if rot == 1:
            dim = [self.box_creator.buffer_list[index][1], self.box_creator.buffer_list[index][0], self.box_creator.buffer_list[index][2]]
        else:
            dim = list(self.box_creator.buffer_list[index])
        self.render_box = [dim, pos]

        return index, pos, rot, dim

    def step(self, action):
        """

        :param action: action index
        :return: cur_observation
                 reward
                 done, Whether to end boxing (i.e., the current box cannot fit in the bin)
                 info
        """
        
        index, pos, rot, size = self.idx2pos(action)
        # print(self.box_creator.box_list)
        # print("box:", self.box_creator.buffer_list[index])
        # print("pos:", pos)
        # print("rot:", rot)      
        # print("size:", size)
        succeeded = self.container.place_box(self.box_creator.buffer_list[index], pos, rot)
        
        if not succeeded:
            if self.reward_type == "terminal":  # Terminal reward
                reward = self.container.get_volume_ratio()
            else:  # Step-wise/Immediate reward
                reward = 0.0
            done = True
            
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio(index)

        self.box_creator.drop_box(index)  # remove current box from the list
        self.box_creator.generate_box_size()  # add a new box to the list

        if self.reward_type == "terminal":
            reward = 0.01
        else:
            reward = box_ratio
        done = False
        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}

        return self.cur_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.box_creator.reset()
        self.container = Container(*self.bin_size)
        self.box_creator.generate_box_size()
        self.candidates = np.zeros_like(self.candidates)
        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self, idx: int  = 0):
        self.renderer.add_item(self.render_box[0], self.render_box[1])
        self.renderer.save_img(idx)
