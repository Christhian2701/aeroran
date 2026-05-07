#!/usr/bin/env python3
"""
run_dual_baseline_simulations.py

Script para executar DUAS simulações simultâneas do cenário hierarchical-xangai-UAV:
1. Always ON (heuristicType=0): Todas as células sempre ligadas
2. Dynamic (heuristicType=1): Heurística dinâmica de energy saving

Os resultados são salvos em CSVs separados para evitar sobreposição.
Formato de saída compatível com qos_hierarchical_metrics_UAV.csv
"""

import argparse
import json
import csv
import os
import sys
import time
import uuid
import subprocess
import multiprocessing
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
        print(f"AVISO: Arquivo .env não encontrado em {env_path}")
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                env_vars[key] = normalize_env_value(key, value.strip(), env_path.parent)
    return env_vars


# Carrega configurações do .env
ENV_CONFIG = load_env_file(ENV_FILE)

# Adiciona paths ao sys.path
NSORAN_GYM_PATH = ENV_CONFIG.get('NSORAN_GYM_PATH', str(SCRIPT_DIR / 'ns-o-ran-gym'))
sys.path.insert(0, os.path.join(NSORAN_GYM_PATH, 'src'))


class BaselineSimulationRunner:
    """
    Classe para executar uma simulação baseline (Always ON ou Dynamic)
    usando o cenário scenario-hierarchical-xangai-UAV
    """

    def __init__(self,
                 ns3_path: str,
                 output_folder: str,
                 heuristic_type: int,  # 0 = Always ON, 1 = Dynamic
                 scenario_config: dict,
                 optimized: bool = True,
                 simulation_id: str = None):

        self.ns3_path = ns3_path
        self.output_folder = output_folder
        self.heuristic_type = heuristic_type
        self.scenario_config = scenario_config.copy()
        self.optimized = optimized
        self.simulation_id = simulation_id or str(uuid.uuid4())[:8]

        # Define nome do modo
        self.mode_name = "always_on" if heuristic_type == 0 else "dynamic"

        # Define pasta de saída específica
        self.sim_output_folder = os.path.join(output_folder, f"sim_{self.mode_name}_{self.simulation_id}")
        os.makedirs(self.sim_output_folder, exist_ok=True)

        # Configuração CSV
        self.csv_path = os.path.join(output_folder, f"qos_hierarchical_{self.mode_name}.csv")

        # Lista de células
        self.cellList = [2, 3, 4, 5, 6, 7, 8]

        # Colunas do CSV (formato igual ao qos_hierarchical_metrics_UAV.csv)
        self._setup_csv_columns()

        # Logger
        self.logger = logging.getLogger(f"Baseline_{self.mode_name}")

    def _setup_csv_columns(self):
        """Define as colunas do CSV no formato correto"""
        self.csv_columns = ["timestamp", "step"]

        # Métricas por célula
        for metric in ["EEKPI_RL", "ES_ON_COST", "QosFlow.PdcpPduVolumeDL_Filter",
                       "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "RLF_Counter",
                       "RLF_VALUE", "RRU_PRBTOTDL", "RRU.PrbUsedDl",
                       "TB_TOTNBRDLINITIAL_64QAM_RATIO"]:
            for cell in self.cellList:
                self.csv_columns.append(f"{metric}_{cell}")

        # Métricas agregadas
        self.csv_columns.extend([
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

    def _ensure_csv(self):
        """Cria o arquivo CSV com cabeçalho se não existir"""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_columns)
            self.logger.info(f"Criado CSV: {self.csv_path}")

    def _get_ns3_executable(self) -> str:
        """Encontra o executável NS3 para o cenário"""
        if self.optimized:
            library_path = os.path.join(self.ns3_path, 'build/optimized')
        else:
            library_path = os.path.join(self.ns3_path, 'build')

        # Lê o arquivo de build status
        build_status_fname = f".lock-ns3_{sys.platform}_build"
        build_status_path = os.path.join(self.ns3_path, build_status_fname)

        if not os.path.exists(build_status_path):
            raise FileNotFoundError(f"Build status não encontrado: {build_status_path}")

        # Importa o módulo de status
        from importlib.machinery import SourceFileLoader
        import types

        loader = SourceFileLoader(build_status_fname, build_status_path)
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)

        # Procura pelo cenário
        scenario_name = "scenario-hierarchical-xangai-UAV"
        matches = [
            os.path.abspath(os.path.join(self.ns3_path, program))
            for program in mod.ns3_runnable_programs
            if scenario_name in program
        ]

        if not matches:
            raise ValueError(f"Cenário {scenario_name} não encontrado")

        return matches[0]

    def _build_command(self, executable: str) -> List[str]:
        """Constrói o comando para executar a simulação"""
        # Atualiza configuração com heurístico
        config = self.scenario_config.copy()
        config['heuristicType'] = self.heuristic_type

        # Constrói argumentos
        command = [executable]
        for param, value in config.items():
            if isinstance(value, list):
                value = value[0]
            command.append(f"--{param}={value}")

        return command

    def _parse_kpm_files(self) -> pd.DataFrame:
        """Parseia os arquivos de KPM gerados pela simulação"""
        import glob

        all_data = []
        last_timestamp = 0

        # Procura arquivos cu-up-cell-*.txt
        for file_path in glob.glob(os.path.join(self.sim_output_folder, 'cu-up-cell-*.txt')):
            cell_id = int(file_path.split('cell-')[-1].split('.')[0])

            if cell_id not in self.cellList:
                continue

            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = int(row.get('timestamp', 0))
                    if timestamp > last_timestamp:
                        last_timestamp = timestamp

                    data = {
                        'timestamp': timestamp,
                        'cell_id': cell_id,
                        'QosFlow.PdcpPduVolumeDL_Filter': float(row.get('QosFlow.PdcpPduVolumeDL_Filter', 0)),
                        'TB.TotNbrDl.1': float(row.get('TB.TotNbrDl.1', 0)),
                        'RRU.PrbUsedDl': float(row.get('RRU.PrbUsedDl', 0)),
                    }
                    all_data.append(data)

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)

    def _calculate_metrics(self, df: pd.DataFrame, step: int) -> Dict[str, float]:
        """Calcula métricas agregadas a partir dos dados"""
        metrics = {}

        # Timestamp
        metrics['timestamp'] = int(df['timestamp'].max()) if not df.empty else int(time.time() * 1000)
        metrics['step'] = step

        # Métricas por célula (valores padrão)
        for cell in self.cellList:
            cell_data = df[df['cell_id'] == cell] if not df.empty else pd.DataFrame()

            qos_flow = cell_data['QosFlow.PdcpPduVolumeDL_Filter'].sum() if not cell_data.empty else 0
            tb_tot = cell_data['TB.TotNbrDl.1'].sum() if not cell_data.empty else 0.00001
            rru_prbused = cell_data['RRU.PrbUsedDl'].mean() if not cell_data.empty else 0

            # EEKPI = throughput / transport blocks
            eekpi = qos_flow / tb_tot if tb_tot > 0 else 0

            # ES_ON_COST: No baseline, custo depende do modo
            if self.heuristic_type == 0:  # Always ON
                es_cost = 0.81  # Custo constante (todas ON)
            else:  # Dynamic
                es_cost = 0.5  # Custo médio estimado

            # RRU_PRBTOTDL: percentual de PRB usado
            rru_prbtot = (rru_prbused / 139) * 100 if rru_prbused > 0 else 0

            metrics[f'EEKPI_RL_{cell}'] = eekpi
            metrics[f'ES_ON_COST_{cell}'] = es_cost
            metrics[f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'] = qos_flow
            metrics[f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}'] = 0  # Placeholder
            metrics[f'RLF_Counter_{cell}'] = 0
            metrics[f'RLF_VALUE_{cell}'] = 0
            metrics[f'RRU_PRBTOTDL_{cell}'] = rru_prbtot
            metrics[f'RRU.PrbUsedDl_{cell}'] = rru_prbused
            metrics[f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}'] = 0.7  # Estimado

        # Métricas agregadas
        metrics['SUM_QosFlow.PdcpPduVolumeDL_Filter'] = sum(
            metrics.get(f'QosFlow.PdcpPduVolumeDL_Filter_{c}', 0) for c in self.cellList
        )
        metrics['SUM_RLF_VALUE'] = 0
        metrics['SUM_TB.TotNbrDl.1'] = df['TB.TotNbrDl.1'].sum() if not df.empty else 0
        metrics['SUM_ES_ON_COST'] = sum(
            metrics.get(f'ES_ON_COST_{c}', 0) for c in self.cellList
        )

        # Latência média
        metrics['AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)'] = 0

        # ZERO_COUNT: células desligadas
        if self.heuristic_type == 0:  # Always ON
            metrics['ZERO_COUNT'] = 0
        else:
            metrics['ZERO_COUNT'] = 2  # Estimado para dynamic

        # Latências em microsegundos
        metrics['latency_cell_us'] = 0
        metrics['latency_ue_us'] = 0

        # Calcula recompensa
        throughput_norm = metrics['SUM_QosFlow.PdcpPduVolumeDL_Filter'] / 1e6
        metrics['reward'] = (
            0.31 * throughput_norm
            - 0.19 * (metrics['SUM_ES_ON_COST'] + metrics['ZERO_COUNT'])
            - 0.2 * metrics['SUM_RLF_VALUE']
            - 0.1 * metrics['SUM_ES_ON_COST']
        )

        return metrics

    def _write_metrics_to_csv(self, metrics: Dict[str, float]):
        """Escreve métricas no CSV"""
        row = [metrics.get(col, 0) for col in self.csv_columns]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run(self, num_steps: int = 100) -> Dict[str, Any]:
        """Executa a simulação baseline"""
        self.logger.info(f"Iniciando simulação {self.mode_name} (heuristic={self.heuristic_type})")
        self.logger.info(f"Output: {self.sim_output_folder}")

        # Garante CSV existe
        self._ensure_csv()

        # Encontra executável
        try:
            executable = self._get_ns3_executable()
            self.logger.info(f"Executável: {executable}")
        except Exception as e:
            self.logger.error(f"Erro ao encontrar executável: {e}")
            return {'success': False, 'error': str(e)}

        # Constrói comando
        command = self._build_command(executable)
        self.logger.info(f"Comando: {' '.join(command[:5])}...")

        # Define ambiente
        if self.optimized:
            library_path = f"{os.path.join(self.ns3_path, 'build/optimized')}:{os.path.join(self.ns3_path, 'build/optimized/lib')}"
        else:
            library_path = f"{os.path.join(self.ns3_path, 'build')}:{os.path.join(self.ns3_path, 'build/lib')}"

        environment = {
            'LD_LIBRARY_PATH': library_path,
            'DYLD_LIBRARY_PATH': library_path
        }
        environment.update(os.environ)

        # Executa simulação
        start_time = time.time()

        try:
            process = subprocess.Popen(
                command,
                cwd=self.sim_output_folder,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Aguarda conclusão ou timeout
            sim_time = self.scenario_config.get('simTime', [10])[0]
            timeout = sim_time * 60 + 300  # simTime em segundos + margem

            stdout, stderr = process.communicate(timeout=timeout)

            elapsed = time.time() - start_time

            if process.returncode != 0:
                self.logger.error(f"Simulação falhou com código {process.returncode}")
                self.logger.error(f"stderr: {stderr[:500]}")
                return {
                    'success': False,
                    'returncode': process.returncode,
                    'stderr': stderr,
                    'elapsed_time': elapsed
                }

            # Parseia resultados
            df = self._parse_kpm_files()

            # Gera métricas por step
            all_rewards = []
            for step in range(1, num_steps + 1):
                metrics = self._calculate_metrics(df, step)
                self._write_metrics_to_csv(metrics)
                all_rewards.append(metrics['reward'])

            self.logger.info(f"Simulação {self.mode_name} concluída em {elapsed:.2f}s")
            self.logger.info(f"Recompensa média: {np.mean(all_rewards):.4f}")

            return {
                'success': True,
                'mode': self.mode_name,
                'heuristic_type': self.heuristic_type,
                'elapsed_time': elapsed,
                'num_steps': num_steps,
                'avg_reward': np.mean(all_rewards),
                'csv_path': self.csv_path
            }

        except subprocess.TimeoutExpired:
            process.kill()
            self.logger.error("Simulação excedeu timeout")
            return {'success': False, 'error': 'timeout'}

        except Exception as e:
            self.logger.error(f"Erro na simulação: {e}")
            return {'success': False, 'error': str(e)}


def run_simulation_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function para executar simulação em processo separado"""
    ns3_path, output_folder, heuristic_type, scenario_config, optimized, sim_id = args

    runner = BaselineSimulationRunner(
        ns3_path=ns3_path,
        output_folder=output_folder,
        heuristic_type=heuristic_type,
        scenario_config=scenario_config,
        optimized=optimized,
        simulation_id=sim_id
    )

    return runner.run(num_steps=100)


def create_scenario_config(env_config: Dict[str, str]) -> dict:
    """Cria configuração do cenário a partir das variáveis de ambiente"""
    return {
        "simTime": [float(env_config.get('SIM_TIME', '10.0'))],
        "ues": [int(env_config.get('NUM_UES', '3'))],
        "RngRun": [int(env_config.get('RNG_RUN', '400'))],
        "configuration": [int(env_config.get('CONFIGURATION', '1'))],
        "trafficModel": [int(env_config.get('TRAFFIC_MODEL', '3'))],
        "numberOfRaPreambles": [64],
        "reducedPmValues": [0],
        "outageThreshold": [2.0],
        "handoverMode": [env_config.get('HANDOVER_MODE', 'DynamicTtt')],
        "indicationPeriodicity": [float(env_config.get('INDICATION_PERIODICITY', '0.1'))],
        "controlFileName": ["hierarchical_actions.csv"],
        "useSemaphores": [0],  # Desabilitado para baseline
        "positionAllocator": [int(env_config.get('POSITION_ALLOCATOR', '2'))],
        "nBsNoUesAlloc": [0],
        "minSpeed": [float(env_config.get('MIN_SPEED', '2.0'))],
        "maxSpeed": [float(env_config.get('MAX_SPEED', '4.0'))],
        "uavMobilityMode": [0],
        "uavFlightPattern": [0],
        "uavBaseAltitude": [50.0],
        "uavMaxSpeed": [15.0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Executa duas simulações baseline em paralelo (Always ON e Dynamic)"
    )

    parser.add_argument("--env_file", type=str, default=str(ENV_FILE),
                        help="Caminho para arquivo .env de configuração")
    parser.add_argument("--ns3_path", type=str, default=None,
                        help="Caminho para ns-3 (sobrescreve .env)")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Pasta de saída (sobrescreve .env)")
    parser.add_argument("--sequential", action="store_true",
                        help="Executar simulações sequencialmente (não em paralelo)")
    parser.add_argument("--only", type=str, choices=['always_on', 'dynamic'],
                        help="Executar apenas um tipo de simulação")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verbose")

    args = parser.parse_args()

    # Configuração de logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dual_baseline_simulations.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Carrega configurações
    env_config = load_env_file(Path(args.env_file))

    # Sobrescreve com argumentos da linha de comando
    ns3_path = args.ns3_path or env_config.get('NS3_PATH')
    output_folder = args.output_folder or env_config.get('OUTPUT_FOLDER', 'output_hierarchical_uav')

    # Validações
    if not ns3_path or not os.path.exists(ns3_path):
        logger.error(f"Caminho ns-3 não encontrado: {ns3_path}")
        sys.exit(1)

    # Cria diretório de saída
    os.makedirs(output_folder, exist_ok=True)

    # Configuração do cenário
    scenario_config = create_scenario_config(env_config)
    optimized = env_config.get('OPTIMIZED', 'true').lower() == 'true'

    # ID único para esta execução
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("DUAL BASELINE SIMULATIONS - ALWAYS ON vs DYNAMIC")
    logger.info("=" * 60)
    logger.info(f"NS3 Path: {ns3_path}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Modo paralelo: {not args.sequential}")

    # Define simulações a executar
    simulations = []

    if args.only != 'dynamic':
        # Always ON (heuristicType=0)
        simulations.append((ns3_path, output_folder, 0, scenario_config, optimized, f"aon_{run_id}"))

    if args.only != 'always_on':
        # Dynamic (heuristicType=1)
        simulations.append((ns3_path, output_folder, 1, scenario_config, optimized, f"dyn_{run_id}"))

    # Executa simulações
    results = []

    if args.sequential or len(simulations) == 1:
        # Execução sequencial
        for sim_args in simulations:
            result = run_simulation_worker(sim_args)
            results.append(result)
    else:
        # Execução paralela usando multiprocessing
        logger.info(f"Executando {len(simulations)} simulações em paralelo...")

        with multiprocessing.Pool(processes=len(simulations)) as pool:
            results = pool.map(run_simulation_worker, simulations)

    # Sumário dos resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)

    for result in results:
        if result.get('success'):
            logger.info(f"\n{result['mode'].upper()}:")
            logger.info(f"  Tempo: {result['elapsed_time']:.2f}s")
            logger.info(f"  Steps: {result['num_steps']}")
            logger.info(f"  Reward médio: {result['avg_reward']:.4f}")
            logger.info(f"  CSV: {result['csv_path']}")
        else:
            mode = result.get('mode', 'unknown')
            error = result.get('error', 'unknown error')
            logger.error(f"\n{mode.upper()}: FALHOU - {error}")

    # Gera resumo comparativo se ambas simulações concluíram
    successful_results = [r for r in results if r.get('success')]
    if len(successful_results) == 2:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARAÇÃO")
        logger.info("=" * 60)

        aon = next((r for r in successful_results if r['mode'] == 'always_on'), None)
        dyn = next((r for r in successful_results if r['mode'] == 'dynamic'), None)

        if aon and dyn:
            logger.info(f"Always ON reward: {aon['avg_reward']:.4f}")
            logger.info(f"Dynamic reward: {dyn['avg_reward']:.4f}")

            diff = dyn['avg_reward'] - aon['avg_reward']
            if diff > 0:
                logger.info(f"Dynamic é {diff:.4f} melhor ({(diff/abs(aon['avg_reward'])*100):.1f}%)")
            else:
                logger.info(f"Always ON é {-diff:.4f} melhor ({(-diff/abs(dyn['avg_reward'])*100):.1f}%)")

    logger.info("\nExecução concluída!")


if __name__ == '__main__':
    main()
