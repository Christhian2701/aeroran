#!/usr/bin/env python3
"""
ppo_trainer.py novo

Online PPO trainer for the energy-saving use case on top of the
scenario-hierarchical-xangai-UAV ns-3 simulation.

This script reuses the existing hierarchical online control loop:
- ns-o-ran-gym NsOranEnv runtime
- hierarchical_actions.csv control file
- semaphore synchronization with the running simulation

Unlike the SAC scripts in this repository, this trainer is ES-only:
- observation: only the aggregated ES feature vector
- action: 7 binary decisions, one per NR gNB (cells 2..8)
- semantics: agent bit 1 -> cell ON, agent bit 0 -> cell OFF
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


SCRIPT_DIR = Path(__file__).parent.resolve()
from ppo_env import UavEnergySavingPpoEnv


DEFAULT_OUTPUT_FOLDER = str(SCRIPT_DIR / "output_ppo_es_uav")
DEFAULT_MODEL_PREFIX = str(SCRIPT_DIR / "ppo_es_uav_model")
DEFAULT_QOS_CSV = "qos_hierarchical_metrics_PPO_ES_UAV.csv"
DEFAULT_TENSORBOARD_DIR = str(SCRIPT_DIR / "ppo_es_uav_tensorboard")


def load_json_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_default_scenario_config(args: argparse.Namespace) -> dict:
    return {
        "simTime": [float(args.sim_time)],
        "ues": [int(args.ues)],
        "RngRun": [int(args.rng_run)],
        "configuration": [int(args.configuration)],
        "trafficModel": [int(args.traffic_model)],
        "numberOfRaPreambles": [64],
        "reducedPmValues": [0],
        "outageThreshold": [float(args.outage_threshold)],
        "handoverMode": [args.handover_mode],
        "indicationPeriodicity": [float(args.indication_periodicity)],
        "controlFileName": ["hierarchical_actions.csv"],
        "useSemaphores": [1],
        "scheduleControlMessages": [0],
        "positionAllocator": [int(args.position_allocator)],
        "nBsNoUesAlloc": [0],
        "minSpeed": [float(args.min_speed)],
        "maxSpeed": [float(args.max_speed)],
        "uavMobilityMode": [int(args.uav_mobility_mode)],
        "uavFlightPattern": [int(args.uav_flight_pattern)],
        "heuristicType": [-1],
        "enableTraces": [str(args.enable_traces).lower()],
        "pathGymOkMetrics": [str(args.path_gym_ok_metrics).lower()],
    }


def make_env(rank: int, seed: int, env_kwargs: dict):
    def _init():
        local_env_kwargs = dict(env_kwargs)
        local_env_kwargs["randomization_seed"] = seed + rank
        return UavEnergySavingPpoEnv(**local_env_kwargs)

    set_random_seed(seed)
    return _init


def unpack_vec_step(step_result):
    """
    Support both:
    - SB3 VecEnv API: obs, rewards, dones, infos
    - Gymnasium-like custom wrappers: obs, rewards, terminated, truncated, infos
    """
    if len(step_result) == 4:
        obs, rewards, dones, infos = step_result
        return obs, rewards, dones, infos

    if len(step_result) == 5:
        obs, rewards, terminated, truncated, infos = step_result
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos

    raise ValueError(f"Unexpected VecEnv step result length: {len(step_result)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent online for ES-only control on scenario-hierarchical-xangai-UAV"
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON scenario config. If omitted, a working default config is built from CLI args.")
    parser.add_argument("--ns3_path", type=str, default=str(SCRIPT_DIR),
                        help="Path to the ns-3 project root.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER,
                        help="Folder where per-run simulation outputs and QoS CSVs are stored.")
    parser.add_argument("--save_path", type=str, default=DEFAULT_MODEL_PREFIX,
                        help="Prefix for the trained PPO model.")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Optional PPO model path prefix to continue training from.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for PPO and env factories.")
    parser.add_argument("--total_timesteps", type=int, default=10000,
                        help="Total PPO timesteps for training.")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Number of evaluation episodes after training.")
    parser.add_argument("--eval_freq", type=int, default=2000,
                        help="Evaluate every N timesteps during training.")
    parser.add_argument("--patience_evals", type=int, default=4,
                        help="Stop after this many eval rounds without improvement.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="PPO learning rate.")
    parser.add_argument("--n_steps", type=int, default=256,
                        help="PPO rollout steps per update.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="PPO minibatch size.")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="PPO optimization epochs per update.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda.")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient.")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value loss coefficient.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device for PPO.")
    parser.add_argument("--optimized", action="store_true",
                        help="Use optimized ns-3 build/profile.")
    parser.add_argument("--verbose_env", action="store_true",
                        help="Enable detailed environment logging.")

    parser.add_argument("--sim_time", type=float, default=10.0)
    parser.add_argument("--ues", type=int, default=9)
    parser.add_argument("--rng_run", type=int, default=400)
    parser.add_argument("--configuration", type=int, default=1)
    parser.add_argument("--traffic_model", type=int, default=3)
    parser.add_argument("--outage_threshold", type=float, default=2.0)
    parser.add_argument("--handover_mode", type=str, default="DynamicTtt")
    parser.add_argument("--indication_periodicity", type=float, default=0.1)
    parser.add_argument("--position_allocator", type=int, default=2)
    parser.add_argument("--min_speed", type=float, default=2.0)
    parser.add_argument("--max_speed", type=float, default=4.0)
    parser.add_argument("--uav_mobility_mode", type=int, default=1)
    parser.add_argument("--uav_flight_pattern", type=int, default=1)
    parser.add_argument("--enable_traces", action="store_true")
    parser.add_argument("--path_gym_ok_metrics", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.ns3_path):
        raise FileNotFoundError(f"ns3_path not found: {args.ns3_path}")

    os.makedirs(args.output_folder, exist_ok=True)
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    tensorboard_log = DEFAULT_TENSORBOARD_DIR if tensorboard_available else None

    if not tensorboard_available:
        print("TensorBoard not installed; continuing without tensorboard logging.")

    if args.config:
        scenario_configuration = load_json_config(args.config)
    else:
        scenario_configuration = create_default_scenario_config(args)

    UavEnergySavingPpoEnv.QOS_CSV_DIR = args.output_folder
    UavEnergySavingPpoEnv.QOS_CSV_BASENAME = DEFAULT_QOS_CSV

    env_kwargs = {
        "ns3_path": args.ns3_path,
        "scenario_configuration": scenario_configuration,
        "output_folder": args.output_folder,
        "optimized": args.optimized,
        "verbose": args.verbose_env,
    }

    print("Creating PPO ES-only UAV environment...")
    env = DummyVecEnv([make_env(rank=0, seed=args.seed, env_kwargs=env_kwargs)])
    eval_env = DummyVecEnv([make_env(rank=1, seed=args.seed + 1000, env_kwargs=env_kwargs)])

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Scenario config: {json.dumps(scenario_configuration, indent=2)}")

    if args.load_path and os.path.exists(args.load_path + ".zip"):
        print(f"Loading PPO model from {args.load_path}")
        model = PPO.load(args.load_path, env=env, device=args.device)
    else:
        print("Creating new PPO model (MlpPolicy)")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            device=args.device,
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            },
            seed=args.seed,
        )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.patience_evals,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=args.eval_freq,
        n_eval_episodes=1,
        log_path=DEFAULT_TENSORBOARD_DIR,
        best_model_save_path=f"{args.save_path}_best_model",
        deterministic=True,
        render=False,
        verbose=1,
    )

    print(f"Starting PPO training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
        reset_num_timesteps=not bool(args.load_path),
    )

    print(f"Saving PPO model to {args.save_path}")
    model.save(args.save_path)

    print(f"Evaluating trained PPO model for {args.eval_episodes} episode(s)...")
    episode_rewards = []
    for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _info = unpack_vec_step(env.step(action))
            episode_reward += float(reward[0])
            done = bool(dones[0])
            step += 1

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: steps={step}, reward={episode_reward:.4f}")

    if episode_rewards:
        print(
            "Evaluation summary: "
            f"mean_reward={np.mean(episode_rewards):.4f}, "
            f"std_reward={np.std(episode_rewards):.4f}"
        )

    env.close()
    eval_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
