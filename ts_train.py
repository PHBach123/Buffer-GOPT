"""
refer:
- https://github.com/albertcity/OCARL
- https://github.com/pioneer-innovation/Real-3D-Embodied-Dataset
"""
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
import warnings
warnings.filterwarnings("ignore")

import time
import pprint
import shutil
import random

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import tianshou as ts
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic
from tianshou.trainer import onpolicy_trainer

import model
import arguments
from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from mycollector import PackCollector
from tools import set_seed  

def make_envs(args):
    train_envs = ts.env.SubprocVectorEnv(
        [lambda: gym.make(args.env.id,
            container_size=args.env.container_size,
            item_set=args.env.box_size_set,
            enable_rotation=args.env.rot,
            data_type=args.env.box_type,
            reward_type=args.train.reward_type,
            action_scheme=args.env.scheme,
            k_placement=args.env.k_placement,
            k_buffer=args.env.k_buffer,
            num_bins=args.env.num_bins 
        ) for _ in range(args.train.num_processes)]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [lambda: gym.make(args.env.id,
            container_size=args.env.container_size,
            item_set=args.env.box_size_set,
            enable_rotation=args.env.rot,
            data_type=args.env.box_type,
            reward_type=args.train.reward_type,
            action_scheme=args.env.scheme,
            k_placement=args.env.k_placement,
            k_buffer=args.env.k_buffer,
            num_bins=args.env.num_bins,  
        ) for _ in range(1)]
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    return train_envs, test_envs

def build_net(args, device):
    feature_net = model.ShareNet(
        k_placement=args.env.k_placement,
        k_buffer=args.env.k_buffer,
        box_max_size=max(args.env.box_size_set[-1]),
        container_size=args.env.container_size,
        embed_size=args.model.embed_dim,
        num_layers=args.model.num_layers,
        forward_expansion=args.model.forward_expansion,
        heads=args.model.heads,
        dropout=args.model.dropout,
        device=device,
        place_gen=args.env.scheme,
        num_bins=args.env.num_bins  # Số thùng cố định
    )

    actor = model.ActorHead(
        preprocess_net=feature_net,
        embed_size=args.model.embed_dim,
        padding_mask=args.model.padding_mask,
        device=device
    ).to(device)

    critic = model.CriticHead(
        k_placement=args.env.k_placement,
        preprocess_net=feature_net,
        embed_size=args.model.embed_dim,
        num_bins=args.env.num_bins,  # Số thùng cố định
        padding_mask=args.model.padding_mask,
        device=device
    ).to(device)

    return actor, critic

def train(args):
    date = time.strftime(r'%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    time_str = (f"{args.env.id}_"
                f"{args.env.container_size[0]}-{args.env.container_size[1]}-{args.env.container_size[2]}_"
                f"{args.env.scheme}_{args.env.k_placement}_"
                f"{args.env.box_type}_"
                f"{args.train.algo}_seed{args.seed}_"
                f"{args.opt.optimizer}_{date}")

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # Environments 
    train_envs, test_envs = make_envs(args)

    # Network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    # Optimizer
    if args.opt.optimizer == 'Adam':
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    elif args.opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(
            actor_critic.parameters(),
            lr=args.opt.lr,
            eps=args.opt.eps,
            alpha=args.opt.alpha,
        )
    else:
        raise NotImplementedError

    # Scheduler
    lr_scheduler = None
    if args.opt.lr_decay:
        max_update_num = np.ceil(args.train.step_per_epoch / args.train.step_per_collect) * args.train.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    # RL agent 
    dist = CategoricalMasked
    if args.train.algo == 'PPO':
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=args.train.gamma,
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler,
            k_placement=args.env.k_placement,
            num_bins=args.env.num_bins, 
        )
    elif args.train.algo == 'A2C':    
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler,
            k_placement=args.env.k_placement,
            num_bins=args.env.num_bins, 
        )
    else:
        raise NotImplementedError

    log_path = './logs/' + time_str
    is_debug = True if sys.gettrace() else False
    if not is_debug:
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer=writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval
        )
        # backup the config file, os.path.join(,)
        shutil.copy(args.config, log_path)  # config file
        shutil.copy("model.py", log_path)  # network
        shutil.copy("arguments.py", log_path)  # network 
    else:
        logger = LazyLogger()

    # ======== callback functions used during training =========
    def train_fn(epoch, env_step):
        pass

    def save_best_fn(policy):
        if not is_debug:
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_best.pth'))

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_final.pth'))
        print(log_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if not is_debug:
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            torch.save({"model": policy.state_dict(), "optim": optim.state_dict()}, ckpt_path)
            return ckpt_path
        return None

    def watch(train_info):
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=10)
        total_ratio = result["total_ratio"]
        total_ratio_std = result["total_ratio_std"]
        rew = result["rew"]
        print(f"The result (over {result['n/ep']} episodes): "
              f"total_ratio={total_ratio:.4f}, total_ratio_std={total_ratio_std:.4f}, rew={rew:.4f}")
        with open(os.path.join(log_path, f"{total_ratio:.4f}_{total_ratio_std:.4f}_{rew:.4f}.txt"), "w") as file:
            file.write(str(train_info).replace("{", "").replace("}", "").replace(", ", "\n"))


    if args.ckp != None:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        print(f"Load checkpoint from {args.ckp}")

    # Buffer and Collector
    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = PackCollector(policy, train_envs, buffer)
    test_collector = PackCollector(policy, test_envs)

    # Trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.train.epoch,
        step_per_epoch=args.train.step_per_epoch,
        repeat_per_collect=args.train.repeat_per_collect,
        episode_per_test=10,
        batch_size=args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False
    )

    final_save_fn(policy)
    pprint.pprint(f'Finished training! \n{result}')
    # watch(result)

if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps
    train(args)
