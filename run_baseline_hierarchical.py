#!/usr/bin/env python3
"""
run_baseline_hierarchical.py

Script para executar simulações baseline do cenário hierarchical-xangai-UAV
com diferentes configurações de heurísticas:
- heuristicType=0: Always ON (todas células sempre ligadas)
- heuristicType=1: Dynamic (heurística dinâmica de energy saving)

Os resultados são salvos em CSVs separados no formato do
qos_hierarchical_metrics_UAV.csv

Uso:
    python run_baseline_hierarchical.py --mode always_on --output output_always_on
    python run_baseline_hierarchical.py --mode dynamic --output output_dynamic
    python run_baseline_hierarchical.py --parallel  # Executa ambos em paralelo
"""

import argparse
import json
import csv
import os
import sys
import time
import subprocess
import multiprocessing
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Configuração de paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / "hierarchical.env"
LEGACY_REPO_ROOT = Path("/home/elioth/iwcmc_oran")
PATH_KEYS = {
    "NS3_PATH",
    "NSORAN_GYM_PATH",
    "SAC_MODEL_PATH",
    "OUTPUT_FOLDER",
    "CSV_METRICS_PATH",
    "DATABASE_PATH",
    "LOG_FILE",
}


def normalize_env_value(key: str, value: str, env_dir: Path) -> str:
    """Resolves path settings against this repo and remaps legacy copied paths."""
    if key not in PATH_KEYS:
        return value

    path_value = Path(os.path.expanduser(value))
    if path_value.is_absolute():
        try:
            relative_path = path_value.relative_to(LEGACY_REPO_ROOT)
        except ValueError:
            return str(path_value)
        return str((SCRIPT_DIR / relative_path).resolve())

    return str((env_dir / path_value).resolve())


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Carrega variáveis do arquivo .env"""
    env_vars = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                env_vars[key] = normalize_env_value(key, value.strip(), env_path.parent)
    return env_vars


def run_ns3_simulation(
    ns3_path: str,
    output_folder: str,
    heuristic_type: int,
    config: Dict,
    optimized: bool = True
) -> Dict[str, Any]:
    """
    Executa uma simulação NS3 diretamente usando subprocess

    Args:
        ns3_path: Caminho para o diretório NS3
        output_folder: Pasta de saída
        heuristic_type: 0=Always ON, 1=Dynamic
        config: Configuração do cenário
        optimized: Usar build otimizado

    Returns:
        Dicionário com resultados da simulação
    """
    mode_name = "always_on" if heuristic_type == 0 else "dynamic"
    logger = logging.getLogger(f"sim_{mode_name}")

    # Cria diretório de saída
    os.makedirs(output_folder, exist_ok=True)

    # Define biblioteca path
    if optimized:
        lib_path = f"{os.path.join(ns3_path, 'build/optimized')}:{os.path.join(ns3_path, 'build/optimized/lib')}"
    else:
        lib_path = f"{os.path.join(ns3_path, 'build')}:{os.path.join(ns3_path, 'build/lib')}"

    # Encontra executável
    build_status_fname = f".lock-ns3_{sys.platform}_build"
    build_status_path = os.path.join(ns3_path, build_status_fname)

    if not os.path.exists(build_status_path):
        logger.error(f"Build status não encontrado: {build_status_path}")
        return {"success": False, "error": "build_status_not_found"}

    from importlib.machinery import SourceFileLoader
    import types

    loader = SourceFileLoader(build_status_fname, build_status_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    scenario_name = "scenario-hierarchical-xangai-UAV"
    matches = [
        os.path.abspath(os.path.join(ns3_path, program))
        for program in mod.ns3_runnable_programs
        if scenario_name in program
    ]

    if not matches:
        logger.error(f"Cenário {scenario_name} não encontrado")
        return {"success": False, "error": "scenario_not_found"}

    executable = matches[0]
    logger.info(f"Executável: {executable}")

    # Constrói comando
    command = [executable]
    config_with_heuristic = config.copy()
    config_with_heuristic['heuristicType'] = heuristic_type

    for param, value in config_with_heuristic.items():
        if isinstance(value, list):
            value = value[0]
        command.append(f"--{param}={value}")

    logger.info(f"Comando: {' '.join(command[:5])}...")

    # Cria arquivo de controle vazio (necessário mesmo para baseline)
    control_file = os.path.join(output_folder, "hierarchical_actions.csv")
    with open(control_file, 'w') as f:
        f.write("timestamp,cellId,hoAllowed\n")  # Cabeçalho vazio
    logger.info(f"Criado arquivo de controle: {control_file}")

    # Ambiente
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = lib_path
    environment['DYLD_LIBRARY_PATH'] = lib_path

    # Executa
    start_time = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=output_folder,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Timeout: simTime * 60 + margem
        sim_time = config.get('simTime', [10])
        if isinstance(sim_time, list):
            sim_time = sim_time[0]
        # Timeout muito maior para simulações complexas (baseado em ~2s/40min observado)
        # Para 30s de simulação: ~15 * 40min = 600min = 10 horas
        timeout = max(sim_time * 1200, 36000)  # Mínimo 10 horas

        stdout, stderr = process.communicate(timeout=timeout)
        elapsed = time.time() - start_time

        # Salva stdout/stderr
        with open(os.path.join(output_folder, 'stdout.txt'), 'w') as f:
            f.write(stdout)
        with open(os.path.join(output_folder, 'stderr.txt'), 'w') as f:
            f.write(stderr)

        if process.returncode != 0:
            logger.error(f"Simulação falhou (returncode={process.returncode})")
            logger.error(f"stderr: {stderr[:1000]}")
            return {
                "success": False,
                "returncode": process.returncode,
                "elapsed_time": elapsed,
                "error": stderr[:500]
            }

        logger.info(f"Simulação concluída em {elapsed:.2f}s")

        # Parseia resultados e gera CSV
        csv_path = generate_metrics_csv(output_folder, mode_name, heuristic_type)

        return {
            "success": True,
            "mode": mode_name,
            "heuristic_type": heuristic_type,
            "elapsed_time": elapsed,
            "csv_path": csv_path,
            "output_folder": output_folder
        }

    except subprocess.TimeoutExpired:
        process.kill()
        logger.error("Simulação excedeu timeout")
        return {"success": False, "error": "timeout"}

    except Exception as e:
        logger.error(f"Erro: {e}")
        return {"success": False, "error": str(e)}


def generate_metrics_csv(output_folder: str, mode_name: str, heuristic_type: int) -> str:
    """
    Gera CSV de métricas a partir dos arquivos de saída do NS3

    Args:
        output_folder: Pasta com arquivos de saída
        mode_name: Nome do modo (always_on ou dynamic)
        heuristic_type: Tipo de heurística

    Returns:
        Caminho do CSV gerado
    """
    import glob

    cell_list = [2, 3, 4, 5, 6, 7, 8]

    # Define colunas
    columns = ["timestamp", "step"]

    for metric in ["EEKPI_RL", "ES_ON_COST", "QosFlow.PdcpPduVolumeDL_Filter",
                   "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "RLF_Counter",
                   "RLF_VALUE", "RRU_PRBTOTDL", "RRU.PrbUsedDl",
                   "TB_TOTNBRDLINITIAL_64QAM_RATIO"]:
        for cell in cell_list:
            columns.append(f"{metric}_{cell}")

    columns.extend([
        "SUM_QosFlow.PdcpPduVolumeDL_Filter",
        "SUM_RLF_VALUE",
        "SUM_TB.TotNbrDl.1",
        "SUM_ES_ON_COST",
        "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
        "ZERO_COUNT",
        "latency_cell_us",
        "latency_ue_us",
        "reward"
    ])

    # Coleta dados dos arquivos cu-up
    data_by_timestamp = {}

    for file_path in glob.glob(os.path.join(output_folder, 'cu-up-cell-*.txt')):
        try:
            cell_id = int(file_path.split('cell-')[-1].split('.')[0])
            if cell_id not in cell_list:
                continue

            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = int(row.get('timestamp', 0))
                    if ts not in data_by_timestamp:
                        data_by_timestamp[ts] = {}

                    data_by_timestamp[ts][cell_id] = {
                        'qos_flow': float(row.get('QosFlow.PdcpPduVolumeDL_Filter', 0)),
                        'tb_tot': float(row.get('TB.TotNbrDl.1', 0.00001)),
                        'rru_prbused': float(row.get('RRU.PrbUsedDl', 0)),
                    }
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")

    # Gera linhas do CSV
    csv_path = os.path.join(output_folder, f"qos_hierarchical_{mode_name}.csv")
    rows = []

    # Ordena timestamps
    sorted_timestamps = sorted(data_by_timestamp.keys())

    for step, ts in enumerate(sorted_timestamps, 1):
        row = {'timestamp': ts, 'step': step}

        cell_data = data_by_timestamp[ts]
        sum_qos = 0
        sum_tb = 0
        sum_es_cost = 0
        sum_rlf = 0

        for cell in cell_list:
            if cell in cell_data:
                cd = cell_data[cell]
                qos = cd['qos_flow']
                tb = max(cd['tb_tot'], 0.00001)
                prb = cd['rru_prbused']

                eekpi = qos / tb
                es_cost = 0.81 if heuristic_type == 0 else 0.5

                row[f'EEKPI_RL_{cell}'] = eekpi
                row[f'ES_ON_COST_{cell}'] = es_cost
                row[f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'] = qos
                row[f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}'] = 0
                row[f'RLF_Counter_{cell}'] = 0
                row[f'RLF_VALUE_{cell}'] = 0
                row[f'RRU_PRBTOTDL_{cell}'] = (prb / 139) * 100
                row[f'RRU.PrbUsedDl_{cell}'] = prb
                row[f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}'] = 0.7

                sum_qos += qos
                sum_tb += tb
                sum_es_cost += es_cost
            else:
                for metric in ["EEKPI_RL", "ES_ON_COST", "QosFlow.PdcpPduVolumeDL_Filter",
                              "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "RLF_Counter",
                              "RLF_VALUE", "RRU_PRBTOTDL", "RRU.PrbUsedDl",
                              "TB_TOTNBRDLINITIAL_64QAM_RATIO"]:
                    row[f'{metric}_{cell}'] = 0

        # Métricas agregadas
        row['SUM_QosFlow.PdcpPduVolumeDL_Filter'] = sum_qos
        row['SUM_RLF_VALUE'] = sum_rlf
        row['SUM_TB.TotNbrDl.1'] = sum_tb
        row['SUM_ES_ON_COST'] = sum_es_cost
        row['AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)'] = 0
        row['ZERO_COUNT'] = 0 if heuristic_type == 0 else 2
        row['latency_cell_us'] = 0
        row['latency_ue_us'] = 0

        # Calcula recompensa
        throughput_norm = sum_qos / 1e6
        row['reward'] = (
            0.31 * throughput_norm
            - 0.19 * (sum_es_cost + row['ZERO_COUNT'])
            - 0.2 * sum_rlf
            - 0.1 * sum_es_cost
        )

        rows.append(row)

    # Escreve CSV
    if rows:
        df = pd.DataFrame(rows)
        # Reordena colunas
        ordered_cols = [c for c in columns if c in df.columns]
        df = df[ordered_cols]
        df.to_csv(csv_path, index=False)
        print(f"CSV gerado: {csv_path} ({len(rows)} linhas)")
    else:
        # Cria CSV vazio com cabeçalho
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        print(f"CSV vazio criado: {csv_path}")

    return csv_path


def worker(args):
    """Worker para multiprocessing"""
    return run_ns3_simulation(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Executa simulações baseline do cenário hierarchical-xangai-UAV"
    )

    parser.add_argument("--mode", type=str, choices=['always_on', 'dynamic', 'both'],
                        default='both', help="Modo de simulação")
    parser.add_argument("--ns3_path", type=str, default=None,
                        help="Caminho para ns-3")
    parser.add_argument("--output", type=str, default=None,
                        help="Pasta de saída base")
    parser.add_argument("--parallel", action="store_true",
                        help="Executa simulações em paralelo")
    parser.add_argument("--sim_time", type=float, default=None,
                        help="Tempo de simulação em segundos")
    parser.add_argument("--ues", type=int, default=None,
                        help="Número de UEs por gNB")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verbose")

    args = parser.parse_args()

    # Configuração de logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('baseline_simulations.log')
        ]
    )
    logger = logging.getLogger(__name__)

    # Carrega configurações
    env_config = load_env_file(ENV_FILE)

    # Define parâmetros
    ns3_path = args.ns3_path or env_config.get('NS3_PATH', str(SCRIPT_DIR))
    output_base = args.output or env_config.get('OUTPUT_FOLDER', str(SCRIPT_DIR / 'output_hierarchical_uav'))
    optimized = env_config.get('OPTIMIZED', 'true').lower() == 'true'

    # Configuração do cenário
    config = {
        "simTime": [args.sim_time or float(env_config.get('SIM_TIME', '10.0'))],
        "ues": [args.ues or int(env_config.get('NUM_UES', '3'))],
        "RngRun": [int(env_config.get('RNG_RUN', '400'))],
        "configuration": [int(env_config.get('CONFIGURATION', '1'))],
        "trafficModel": [int(env_config.get('TRAFFIC_MODEL', '3'))],
        "numberOfRaPreambles": [64],
        "reducedPmValues": [0],
        "outageThreshold": [2.0],
        "handoverMode": [env_config.get('HANDOVER_MODE', 'DynamicTtt')],
        "indicationPeriodicity": [float(env_config.get('INDICATION_PERIODICITY', '0.1'))],
        "controlFileName": ["hierarchical_actions.csv"],
        "useSemaphores": ["false"],  # Desabilitado para baseline
        "positionAllocator": [int(env_config.get('POSITION_ALLOCATOR', '2'))],
        "nBsNoUesAlloc": [0],
        "minSpeed": [float(env_config.get('MIN_SPEED', '2.0'))],
        "maxSpeed": [float(env_config.get('MAX_SPEED', '4.0'))],
        "uavMobilityMode": [0],
        "uavFlightPattern": [0],
        "uavBaseAltitude": [50.0],
        "uavMaxSpeed": [15.0],
        # Energy Saving heuristic parameters (para 7 células mmWave: 2+2+2+1=7)
        "sinrTh": [73.0],
        "bsOn": [2],
        "bsIdle": [2],
        "bsSleep": [2],
        "bsOff": [1],
    }

    logger.info("=" * 60)
    logger.info("BASELINE SIMULATIONS - HIERARCHICAL UAV SCENARIO")
    logger.info("=" * 60)
    logger.info(f"NS3 Path: {ns3_path}")
    logger.info(f"Output Base: {output_base}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Parallel: {args.parallel}")

    # Define simulações a executar
    simulations = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ['always_on', 'both']:
        output_aon = os.path.join(output_base, f"always_on_{run_id}")
        simulations.append((ns3_path, output_aon, 0, config, optimized))

    if args.mode in ['dynamic', 'both']:
        output_dyn = os.path.join(output_base, f"dynamic_{run_id}")
        simulations.append((ns3_path, output_dyn, 1, config, optimized))

    # Executa simulações
    results = []

    if args.parallel and len(simulations) > 1:
        logger.info(f"Executando {len(simulations)} simulações em paralelo...")
        with multiprocessing.Pool(processes=len(simulations)) as pool:
            results = pool.map(worker, simulations)
    else:
        for sim_args in simulations:
            result = run_ns3_simulation(*sim_args)
            results.append(result)

    # Resumo
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)

    for result in results:
        if result.get('success'):
            mode = result.get('mode', 'unknown')
            logger.info(f"\n{mode.upper()}:")
            logger.info(f"  Tempo: {result['elapsed_time']:.2f}s")
            logger.info(f"  CSV: {result['csv_path']}")
            logger.info(f"  Output: {result['output_folder']}")
        else:
            logger.error(f"\nFALHA: {result.get('error', 'unknown')}")

    # Comparação
    successful = [r for r in results if r.get('success')]
    if len(successful) == 2:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARAÇÃO")
        logger.info("=" * 60)

        for result in successful:
            csv_path = result.get('csv_path')
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'reward' in df.columns:
                    avg_reward = df['reward'].mean()
                    logger.info(f"{result['mode'].upper()}: reward médio = {avg_reward:.4f}")

    logger.info("\nExecução concluída!")


if __name__ == '__main__':
    main()
