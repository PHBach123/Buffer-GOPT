import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import random
import gymnasium as gym
import torch
from tianshou.data import VectorReplayBuffer

from tools import *
from ts_train import build_net  
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
import arguments
from tools import set_seed  

def test(args):

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
    
    set_seed(args.seed, args.cuda, args.cuda_deterministic)
    print(f"Using device: {device}")

    # Environment
    test_env = gym.make(args.env.id,
        container_size=args.env.container_size,
        item_set=args.env.box_size_set,
        enable_rotation=args.env.rot,
        data_type=args.env.box_type,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        k_buffer=args.env.k_buffer,
        num_bins=args.env.num_bins,  
        is_render=args.render
    )

    # Network
    actor, critic = build_net(args, device)
    actor_critic = torch.nn.ModuleList([actor, critic])  

    # Optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)

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
            k_placement=args.env.k_placement,
            num_bins=args.env.num_bins, 
        )
    else:
        raise NotImplementedError

    
    policy.eval()
    try:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        print(f"Loaded model from {args.ckp}")
    except FileNotFoundError:
        print("No model found at", args.ckp)
        exit()

    # Collector
    test_collector = PackCollector(policy, test_env)

    # Evaluation
    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    for i in range(args.test_episode):
        print(f"Episode {i+1}\t => "
              f"\t| Reward: {result['rews'][i]:.4f} ")
    print('All cases have been done!')
    print('----------------------------------------------')
    # print(f"Average space utilization (total_ratio): {result['total_ratio']:.4f}")
    print(f"Reward: {result['rew']:.4f}")
    print(f"Average reward: {result['rew']/args.env.num_bins:.4f}")
    print("Standard variance: %.4f"%(result['rew_std']))
    # print(f"Standard deviation of total_ratio: {result['total_ratio_std']:.4f}")

if __name__ == '__main__':
    registration_envs()
    torch.set_printoptions(sci_mode=False, precision=4)
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps

    if args.render:
        args.test_episode = 1  
    else:
        args.test_episode = 1


    test(args)

    # for i in range(6):
    #     for j in range(6):
    #         args.env.k_buffer = i + 1
    #         args.env.num_bins = j + 1
    #         print(f"Buffer size: {args.env.k_buffer}")
    #         print(f"Number of bins: {args.env.num_bins}")
    #         test(args)

