import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    ReplayBuffer,
    to_numpy,
    Collector
)
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

class PackCollector(Collector):
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Thu thập một số lượng bước hoặc episode xác định.

        :param int n_step: số bước cần thu thập.
        :param int n_episode: số episode cần thu thập.
        :param bool random: sử dụng chính sách ngẫu nhiên hay không.
        :param float render: thời gian nghỉ giữa các khung hình khi render.
        :param bool no_grad: có giữ gradient trong policy.forward() hay không.
        :param gym_reset_kwargs: tham số bổ sung cho hàm reset của môi trường.

        :return: Một dict chứa các khóa sau:
            - "n/ep": số episode thu thập.
            - "n/st": số bước thu thập.
            - "rews": mảng phần thưởng episode.
            - "lens": mảng độ dài episode.
            - "idxs": mảng chỉ số bắt đầu episode trong buffer.
            - "rew": trung bình phần thưởng episode.
            - "len": trung bình độ dài episode.
            - "rew_std": độ lệch chuẩn phần thưởng.
            - "len_std": độ lệch chuẩn độ dài.
            - "bin_idxs": mảng chỉ số thùng được sử dụng.
            - "total_ratios": mảng tỷ lệ sử dụng không gian trung bình.
            - "total_ratio": trung bình tỷ lệ sử dụng không gian.
            - "total_ratio_std": độ lệch chuẩn tỷ lệ sử dụng không gian.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector.collect, "
                f"got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in PackCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        episode_bin_idxs = []
        episode_total_ratios = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # Lưu trữ trạng thái ẩn nếu có
            last_state = self.data.policy.pop("hidden_state", None)

            # Lấy hành động tiếp theo
            if random:
                act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                policy = result.get("policy", Batch())
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            action_remap = self.policy.map_action(self.data.act)
            

            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,
                ready_env_ids
            )
            done = np.logical_or(terminated, truncated)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )

            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # Thêm dữ liệu vào buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # Thu thập thống kê
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                
                # Thu thập thông tin bổ sung từ multi-bin packing
                # episode_bin_idxs.append([info["bin_idx"][i] for i in env_ind_local])
                # episode_total_ratios.append([info["total_ratio"][i] for i in env_ind_local])

                # Reset môi trường đã hoàn thành
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # Loại bỏ các env dư thừa nếu dùng n_episode
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # Cập nhật thống kê tổng quát
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        # Tính toán kết quả trả về
        if episode_count > 0:
            rews, lens, idxs= list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
            # total_ratio_mean, total_ratio_std = total_ratios.mean(), total_ratios.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            bin_idxs, total_ratios = np.array([]), np.array([])
            rew_mean = rew_std = len_mean = len_std = 0
            # total_ratio_mean = total_ratio_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            # "bin_idxs": bin_idxs,
            # "total_ratios": total_ratios,
            # "total_ratio": total_ratio_mean,
            # "total_ratio_std": total_ratio_std,
        }