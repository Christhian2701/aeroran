#!/usr/bin/env python3
"""
run_uav_with_sac1_env.py

Executa o cenário UAV usando o ambiente HierarchicalEnv original do sac1
com o modelo SAC treinado.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'sac1'))
sys.path.insert(0, os.path.join(script_dir, 'ns-o-ran-gym', 'src'))

from hierarchical_env import HierarchicalEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import json

if __name__ == '__main__':
    ns3_path = script_dir
    output_folder = os.path.join(script_dir, 'output_hierarchical_uav')
    model_path = os.path.join(
        script_dir,
        'sac1',
        'sac_hierarchical_model_best_model',
        'best_model.zip',
    )
    
    scenario_config = {
        "simTime": [5.0],
        "ues": [9],
        "RngRun": [400],
        "configuration": [1],
        "trafficModel": [3],
        "numberOfRaPreambles": [64],
        "reducedPmValues": [0],
        "outageThreshold": [2.0],
        "handoverMode": ["DynamicTtt"],
        "indicationPeriodicity": [0.10],
        "controlFileName": ["hierarchical_actions.csv"],
        "useSemaphores": [1],
        "positionAllocator": [2],
        "nBsNoUesAlloc": [0],
    }
    
    HierarchicalEnv.QOS_CSV_DIR = output_folder
    HierarchicalEnv.QOS_CSV_BASENAME = "qos_hierarchical_metrics_UAV.csv"
    
    print(f"Criando ambiente UAV...")
    env = HierarchicalEnv(
        ns3_path=ns3_path,
        scenario_configuration=scenario_config,
        output_folder=output_folder,
        optimized=True,
        do_heuristic=False,
        ts_reward_weight=1.0,
        verbose=True,
        scenario_name='scenario-hierarchical-xangai-UAV'
    )
    
    print(f"Carregando modelo SAC de {model_path}...")
    model = SAC.load(model_path, env=env)
    
    print("Iniciando episódio...")
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        print(f"Step {step}: reward={reward:.4f}")
    
    print(f"Episódio finalizado com {step} steps")
    env.close()
    print(f"CSV salvo em: {output_folder}/qos_hierarchical_metrics_UAV.csv")
