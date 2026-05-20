#!/usr/bin/env python3
"""
ppo_runner.py

Run inference with a trained PPO model against the online
scenario-hierarchical-xangai-UAV simulation.

This launches ns-3 through the same PPO ES-only environment used for training,
steps the model online, and leaves the raw trace files in per-run UUID folders
under the chosen output directory.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from ppo_env import UavEnergySavingPpoEnv


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_FOLDER = str(SCRIPT_DIR / "output_ppo_inference")


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
        description="Run PPO inference online on scenario-hierarchical-xangai-UAV"
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained PPO .zip model.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON scenario config. If omitted, defaults are built from CLI args.")
    parser.add_argument("--ns3_path", type=str, default=str(SCRIPT_DIR),
                        help="Path to the ns-3 project root.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER,
                        help="Folder where per-run simulation outputs are stored.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device for inference.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for environment-side episode randomization.")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of evaluation episodes to run.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic PPO actions.")
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
    parser.add_argument("--enable_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path_gym_ok_metrics", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.ns3_path):
        raise FileNotFoundError(f"ns3_path not found: {args.ns3_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    os.makedirs(args.output_folder, exist_ok=True)

    if args.config:
        scenario_configuration = load_json_config(args.config)
    else:
        scenario_configuration = create_default_scenario_config(args)

    UavEnergySavingPpoEnv.QOS_CSV_DIR = args.output_folder
    UavEnergySavingPpoEnv.QOS_CSV_BASENAME = "qos_hierarchical_metrics_PPO_ES_UAV.csv"

    env_kwargs = {
        "ns3_path": args.ns3_path,
        "scenario_configuration": scenario_configuration,
        "output_folder": args.output_folder,
        "optimized": args.optimized,
        "verbose": args.verbose_env,
    }

    print("Creating PPO ES-only UAV inference environment...")
    env = DummyVecEnv([make_env(rank=0, seed=args.seed, env_kwargs=env_kwargs)])
    print(f"Scenario config: {json.dumps(scenario_configuration, indent=2)}")

    print(f"Loading PPO model from {args.model_path}")
    model = PPO.load(args.model_path, env=env, device=args.device)

    episode_rewards = []

    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _infos = unpack_vec_step(env.step(action))
            episode_reward += float(rewards[0])
            done = bool(dones[0])
            step += 1

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: steps={step}, reward={episode_reward:.4f}")

    if episode_rewards:
        print(
            "Inference summary: "
            f"mean_reward={np.mean(episode_rewards):.4f}, "
            f"std_reward={np.std(episode_rewards):.4f}"
        )

    env.close()
    print(f"Raw traces are under per-run UUID folders inside: {args.output_folder}")


if __name__ == "__main__":
    main()

