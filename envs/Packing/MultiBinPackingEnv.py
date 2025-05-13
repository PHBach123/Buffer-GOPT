from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .env import PackingEnv  # Giả sử PackingEnv nằm trong file packing_env.py
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator

class MultiBinPackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),  
        item_set=None,
        data_name=None,
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type="step-wise",  
        action_scheme="EMS",  
        k_placement=80, 
        k_buffer=2, 
        num_bins=3,  
        is_render=False,
        is_hold_on=False,
        **kwargs
    ):
        self.num_bins = num_bins
        self.bin_size = container_size
        self.k_placement = k_placement
        self.k_buffer = k_buffer
        self.reward_type = reward_type

        # Initialize box_creator 
        if not load_test_data:
            assert item_set is not None
            if data_type == "random":
                print(f"using items generated randomly")
                self.box_creator = RandomBoxCreator(item_set,self.k_buffer)  
            if data_type == "cut":
                print(f"using items generated through cutting method")
                low = list(item_set[0])
                up = list(item_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            print(f"use box dataset: {data_name}")
            self.box_creator = LoadBoxCreator(data_name, self.k_buffer)        
        
        # Initialize PackingEnv instances
        self.bins = [
            PackingEnv(
                container_size=container_size,
                item_set=item_set,
                data_name=data_name,
                load_test_data=load_test_data,
                enable_rotation=enable_rotation,
                data_type=data_type,
                reward_type=reward_type,
                action_scheme=action_scheme,
                k_placement=k_placement,
                k_buffer=k_buffer,
                is_render=is_render,
                is_hold_on=is_hold_on,
                box_creator=self.box_creator,
                **kwargs
            ) for _ in range(num_bins)
        ]

        # Synchronize box_creator 
        for bin_env in self.bins:
            bin_env.box_creator = self.box_creator

        # Set observation and action spaces
        self._set_space()

    def _set_space(self):
        # Observation space: Combine observations from all bins
        single_obs_len = self.bins[0].observation_space["obs"].shape[0]
        total_obs_len = single_obs_len * self.num_bins
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(total_obs_len,)),
            "mask": spaces.Box(low=0, high=1, shape=(self.num_bins * self.k_placement,), dtype=np.int32)
        })

         # Action space: Select a bin and a placement index within the bin 
        self.action_space = spaces.Discrete(self.num_bins * 2 * self.k_placement)

    @property
    def cur_observation(self):
        """
        Combine observation and mask from all bins
        """
        np.set_printoptions(threshold=np.inf)
        obs_list = []
        mask_list = []
        for bin_env in self.bins:
            bin_obs = bin_env.cur_observation
            obs_list.append(bin_obs["obs"])
            mask_list.append(bin_obs["mask"])
            # print('obs:',bin_obs["obs"])
            # print('mask:',bin_obs["mask"])

        # Concatenate observations and masks
        obs = np.concatenate(obs_list)
        mask = np.concatenate(mask_list)
        # print(f"obs_concat: {obs}", obs)
        # print(f"mask_concat: {mask}", mask)
        return {"obs": obs, "mask": mask}

    def step(self, action):
        """
        Execute an action: select a bin and place an item
        """

        bin_idx = action // (2 * self.k_placement * self.k_buffer)  # Bin index
        placement_idx = action % (2 * self.k_placement * self.k_buffer)  # Placement index within the bin

        obs, reward, done, truncated, info = self.bins[bin_idx].step(placement_idx)

        # if done == True:
        #     print(f"Bin 0:",self.bins[0].container.get_volume_ratio())
            # for i in range(self.num_bins):
            #     print(f"Bin {i} state: {self.bins[i].container.get_volume_ratio()}")
            #     print(self.bins[i].box_creator.buffer_list)

        # if np.random.rand() < 0.1:
        #     done = True

        info["bin_idx"] = bin_idx
        info["total_ratio"] = sum(bin.container.get_volume_ratio() for bin in self.bins) / self.num_bins
        self.bin_idx = bin_idx

        return self.cur_observation, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Reset all bins
        """
        self.box_creator.reset()
        for bin_env in self.bins:
            bin_env.reset(seed=seed)
        return self.cur_observation, {}

    def seed(self, s=None):
        """
        Set seed for all bins
        """
        for bin_env in self.bins:
            bin_env.seed(s)

    def render(self):
        """
        Render all bins (if rendering is enabled)
        """
        self.bins[self.bin_idx].render(self.bin_idx)
