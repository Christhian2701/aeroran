#!/usr/bin/env python3
"""
run_hierarchical_sac.py

Script para executar o modelo SAC treinado no cenário hierarchical-xangai-UAV
usando o ns-O-RAN Gymnasium environment.

Baseado no projeto sac1/ e ns-o-ran-gym/, este script:
1. Carrega as configurações do arquivo hierarchical.env
2. Carrega o modelo SAC pré-treinado
3. Executa simulações NS3 com o cenário UAV
4. Coleta métricas em banco de dados SQLite
5. Exporta resultados para CSV
"""

import argparse
import json
import csv
import os
import sys
import time
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Configuração de paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / "hierarchical.env"

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
                env_vars[key.strip()] = value.strip()
    return env_vars

# Carrega configurações do .env
ENV_CONFIG = load_env_file(ENV_FILE)

# Adiciona paths ao sys.path
NSORAN_GYM_PATH = ENV_CONFIG.get('NSORAN_GYM_PATH', str(SCRIPT_DIR / 'ns-o-ran-gym'))
sys.path.insert(0, os.path.join(NSORAN_GYM_PATH, 'src'))
sys.path.insert(0, str(SCRIPT_DIR / 'sac1'))

# Imports após configuração de paths
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
except ImportError as e:
    print(f"ERRO: Dependências não encontradas: {e}")
    print("Instale com: pip install stable-baselines3 torch")
    sys.exit(1)

try:
    from nsoran.ns_env import NsOranEnv
    from nsoran.datalake import SQLiteDatabaseAPI
    from nsoran.action_controller import ActionController
except ImportError as e:
    print(f"ERRO: nsoran não encontrado: {e}")
    print(f"Verifique se NSORAN_GYM_PATH está correto: {NSORAN_GYM_PATH}")
    sys.exit(1)

from gymnasium import spaces


class HierarchicalMetricsDB:
    """Gerenciador de banco de dados SQLite para métricas hierárquicas"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._ensure_db()

    def _ensure_db(self):
        """Cria o banco de dados e tabelas se não existirem"""
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Tabela de episódios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                total_steps INTEGER,
                total_reward REAL,
                avg_latency_us REAL,
                avg_throughput REAL,
                num_handovers INTEGER,
                cells_on_avg REAL
            )
        ''')

        # Tabela de métricas por step
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS step_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step INTEGER,
                timestamp INTEGER,
                reward REAL,
                latency_cell_us REAL,
                latency_ue_us REAL,
                sum_throughput REAL,
                sum_rlf REAL,
                sum_es_on_cost REAL,
                zero_count INTEGER,
                action_es TEXT,
                action_ts TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            )
        ''')

        # Tabela de métricas por célula
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cell_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step INTEGER,
                cell_id INTEGER,
                eekpi_rl REAL,
                es_on_cost REAL,
                qos_flow REAL,
                pdcp_latency REAL,
                rlf_counter REAL,
                rlf_value REAL,
                rru_prbtotdl REAL,
                rru_prbused REAL,
                tb_64qam_ratio REAL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            )
        ''')

        self.conn.commit()

    def start_episode(self) -> int:
        """Inicia um novo episódio e retorna o ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO episodes (start_time) VALUES (?)
        ''', (datetime.now().isoformat(),))
        self.conn.commit()
        return cursor.lastrowid

    def end_episode(self, episode_id: int, total_steps: int, total_reward: float,
                    avg_latency: float, avg_throughput: float,
                    num_handovers: int, cells_on_avg: float):
        """Finaliza um episódio com estatísticas"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE episodes SET
                end_time = ?,
                total_steps = ?,
                total_reward = ?,
                avg_latency_us = ?,
                avg_throughput = ?,
                num_handovers = ?,
                cells_on_avg = ?
            WHERE episode_id = ?
        ''', (datetime.now().isoformat(), total_steps, total_reward,
              avg_latency, avg_throughput, num_handovers, cells_on_avg, episode_id))
        self.conn.commit()

    def insert_step_metrics(self, episode_id: int, step: int, timestamp: int,
                           reward: float, latency_cell: float, latency_ue: float,
                           throughput: float, rlf: float, es_cost: float,
                           zero_count: int, action_es: str, action_ts: str):
        """Insere métricas de um step"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO step_metrics
            (episode_id, step, timestamp, reward, latency_cell_us, latency_ue_us,
             sum_throughput, sum_rlf, sum_es_on_cost, zero_count, action_es, action_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (episode_id, step, timestamp, reward, latency_cell, latency_ue,
              throughput, rlf, es_cost, zero_count, action_es, action_ts))
        self.conn.commit()

    def insert_cell_metrics(self, episode_id: int, step: int, cell_id: int,
                           metrics: Dict[str, float]):
        """Insere métricas de uma célula"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO cell_metrics
            (episode_id, step, cell_id, eekpi_rl, es_on_cost, qos_flow, pdcp_latency,
             rlf_counter, rlf_value, rru_prbtotdl, rru_prbused, tb_64qam_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (episode_id, step, cell_id,
              metrics.get('eekpi', 0), metrics.get('es_cost', 0),
              metrics.get('qos_flow', 0), metrics.get('latency', 0),
              metrics.get('rlf_counter', 0), metrics.get('rlf_value', 0),
              metrics.get('rru_prbtotdl', 0), metrics.get('rru_prbused', 0),
              metrics.get('tb_64qam', 0)))
        self.conn.commit()

    def export_to_csv(self, csv_path: str):
        """Exporta todos os dados para CSV no formato do projeto SAC1"""
        df_steps = pd.read_sql_query('''
            SELECT * FROM step_metrics ORDER BY episode_id, step
        ''', self.conn)

        df_cells = pd.read_sql_query('''
            SELECT * FROM cell_metrics ORDER BY episode_id, step, cell_id
        ''', self.conn)

        # Pivot das métricas de células
        cell_list = [2, 3, 4, 5, 6, 7, 8]

        # Cria DataFrame no formato SAC1
        rows = []
        for _, step_row in df_steps.iterrows():
            row = {
                'timestamp': step_row['timestamp'],
                'step': step_row['step']
            }

            # Adiciona métricas por célula
            step_cells = df_cells[
                (df_cells['episode_id'] == step_row['episode_id']) &
                (df_cells['step'] == step_row['step'])
            ]

            for cell in cell_list:
                cell_data = step_cells[step_cells['cell_id'] == cell]
                if len(cell_data) > 0:
                    cd = cell_data.iloc[0]
                    row[f'EEKPI_RL_{cell}'] = cd['eekpi_rl']
                    row[f'ES_ON_COST_{cell}'] = cd['es_on_cost']
                    row[f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'] = cd['qos_flow']
                    row[f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}'] = cd['pdcp_latency']
                    row[f'RLF_Counter_{cell}'] = cd['rlf_counter']
                    row[f'RLF_VALUE_{cell}'] = cd['rlf_value']
                    row[f'RRU_PRBTOTDL_{cell}'] = cd['rru_prbtotdl']
                    row[f'RRU.PrbUsedDl_{cell}'] = cd['rru_prbused']
                    row[f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}'] = cd['tb_64qam_ratio']
                else:
                    for metric in ['EEKPI_RL', 'ES_ON_COST', 'QosFlow.PdcpPduVolumeDL_Filter',
                                   'DRB.PdcpSduDelayDl.UEID (pdcpLatency)', 'RLF_Counter',
                                   'RLF_VALUE', 'RRU_PRBTOTDL', 'RRU.PrbUsedDl',
                                   'TB_TOTNBRDLINITIAL_64QAM_RATIO']:
                        row[f'{metric}_{cell}'] = 0.0

            # Adiciona métricas agregadas
            row['SUM_QosFlow.PdcpPduVolumeDL_Filter'] = step_row['sum_throughput']
            row['SUM_RLF_VALUE'] = step_row['sum_rlf']
            row['SUM_TB.TotNbrDl.1'] = 0  # Placeholder
            row['SUM_ES_ON_COST'] = step_row['sum_es_on_cost']
            row['AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)'] = step_row['latency_ue_us']
            row['ZERO_COUNT'] = step_row['zero_count']
            row['latency_cell_us'] = step_row['latency_cell_us']
            row['latency_ue_us'] = step_row['latency_ue_us']
            row['reward'] = step_row['reward']

            rows.append(row)

        if rows:
            df_export = pd.DataFrame(rows)
            df_export.to_csv(csv_path, index=False)
            print(f"Exportado {len(rows)} registros para {csv_path}")
        else:
            print("Nenhum dado para exportar")

    def close(self):
        """Fecha a conexão com o banco"""
        if self.conn:
            self.conn.close()


class HierarchicalUAVEnv(NsOranEnv):
    """
    Ambiente Gymnasium para o cenário hierarchical-xangai-UAV
    Combina Energy Saving (ES) e Traffic Steering (TS) com mobilidade UAV
    """

    def __init__(self, ns3_path: str, scenario_configuration: dict,
                 output_folder: str, optimized: bool = True,
                 do_heuristic: bool = False, ts_reward_weight: float = 1.0,
                 verbose: bool = False, scenario_name: str = 'scenario-hierarchical-xangai-UAV',
                 metrics_db: Optional[HierarchicalMetricsDB] = None):

        super().__init__(
            ns3_path=ns3_path,
            scenario=scenario_name,
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=['action_type', 'param1', 'param2'],
            log_file='HierarchicalActions.txt',
            control_file='hierarchical_actions.csv'
        )

        self.metrics_db = metrics_db
        self.episode_id = None
        self.verbose = verbose
        self.heur = do_heuristic
        self.ts_reward_weight = ts_reward_weight

        # Configuração das células
        self.cellList = [2, 3, 4, 5, 6, 7, 8]
        self.n_gnbs = len(self.cellList)
        self.n_ues_per_gnb = scenario_configuration.get('ues', 9)
        self.n_ues_total = self.n_ues_per_gnb * self.n_gnbs

        # Lista de ações ES válidas
        self.action_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 24, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 48,
            49, 50, 52, 56, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 80, 81, 82,
            84, 88, 96, 97, 98, 100, 104, 112
        ]

        # Colunas de estado ES
        self.es_columns_state = (
            [f"EEKPI_RL_{c}" for c in self.cellList]
            + [f"ES_ON_COST_{c}" for c in self.cellList]
            + [f"QosFlow.PdcpPduVolumeDL_Filter_{c}" for c in self.cellList]
            + [f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{c}" for c in self.cellList]
            + [f"RLF_Counter_{c}" for c in self.cellList]
            + [f"RLF_VALUE_{c}" for c in self.cellList]
            + [f"RRU_PRBTOTDL_{c}" for c in self.cellList]
            + [f"RRU.PrbUsedDl_{c}" for c in self.cellList]
            + [f"TB_TOTNBRDLINITIAL_64QAM_RATIO_{c}" for c in self.cellList]
            + ["SUM_QosFlow.PdcpPduVolumeDL_Filter", "SUM_RLF_VALUE", "SUM_TB.TotNbrDl.1",
               "SUM_ES_ON_COST", "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "ZERO_COUNT"]
        )

        # Colunas de estado TS (por UE)
        self.ts_columns_state = [
            'RRU.PrbUsedDl', 'L3 serving SINR', 'DRB.MeanActiveUeDl',
            'TB.TotNbrDlInitial.Qpsk', 'TB.TotNbrDlInitial.16Qam',
            'TB.TotNbrDlInitial.64Qam', 'TB.TotNbrDlInitial'
        ]

        # Espaço de ação SAC (contínuo)
        action_dims = []
        if self.heur:
            action_dims.extend([2] * self.n_gnbs)
        else:
            action_dims.append(len(self.action_list))
        action_dims.extend([self.n_gnbs + 1] * self.n_ues_total)

        self.original_action_dims = np.array(action_dims)
        n_actions_total = len(self.original_action_dims)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions_total,), dtype=np.float32)

        # Espaço de observação
        es_obs_shape = (len(self.es_columns_state),)
        ts_obs_shape = (self.n_ues_total, len(self.ts_columns_state))

        self.observation_space = spaces.Dict({
            "es_obs": spaces.Box(shape=es_obs_shape, low=-np.inf, high=np.inf, dtype=np.float64),
            "ts_obs": spaces.Box(shape=ts_obs_shape, low=-np.inf, high=np.inf, dtype=np.float64)
        })

        # Estado interno
        self.num_steps = 0
        self.cells_states = {cell: 1 for cell in self.cellList}  # Todas ON inicialmente
        self.observations = pd.DataFrame()
        self.previous_kpms_ts = None
        self.handovers_dict = {}

        # Métricas do episódio
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_throughputs = []
        self.episode_handovers = 0

        # Logger
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def reset(self, seed=None, options=None):
        """Reset do ambiente e início de novo episódio"""
        obs, info = super().reset(seed=seed, options=options)

        self.num_steps = 0
        self.cells_states = {cell: 1 for cell in self.cellList}
        self.previous_kpms_ts = None
        self.handovers_dict = {}

        # Reset métricas do episódio
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_throughputs = []
        self.episode_handovers = 0

        # Inicia novo episódio no banco de dados
        if self.metrics_db:
            self.episode_id = self.metrics_db.start_episode()

        # Retorna observação inicial
        return self._get_obs(), info

    def _get_obs(self) -> dict:
        """Obtém observação atual do estado do ambiente"""
        es_obs = np.zeros(len(self.es_columns_state), dtype=np.float64)
        ts_obs = np.zeros((self.n_ues_total, len(self.ts_columns_state)), dtype=np.float64)

        try:
            # Tenta obter dados do datalake
            if hasattr(self, 'datalake') and self.datalake:
                # ES observation (agregar métricas por célula)
                for i, col in enumerate(self.es_columns_state):
                    if col in self.observations.columns:
                        es_obs[i] = float(self.observations[col].iloc[0]) if len(self.observations) > 0 else 0.0

                # TS observation (métricas por UE)
                # Simplificado: usa valores padrão se não disponíveis
                pass

        except Exception as e:
            self.logger.warning(f"Erro ao obter observação: {e}")

        return {"es_obs": es_obs, "ts_obs": ts_obs}

    def _compute_action(self, continuous_action: np.ndarray) -> list:
        """
        Converte ação contínua SAC para formato NS3
        """
        # Normaliza de [-1, 1] para [0, 1]
        normalized = (continuous_action + 1.0) / 2.0

        # Converte para índices discretos
        discrete_actions = []
        for i, (norm_val, n_dim) in enumerate(zip(normalized, self.original_action_dims)):
            idx = int(np.floor(norm_val * n_dim))
            idx = min(idx, n_dim - 1)  # Clamp ao máximo
            discrete_actions.append(idx)

        # Extrai ações ES e TS
        if self.heur:
            es_action = discrete_actions[:self.n_gnbs]
            ts_actions = discrete_actions[self.n_gnbs:]
        else:
            es_idx = discrete_actions[0]
            if es_idx < len(self.action_list):
                es_action_int = self.action_list[es_idx]
            else:
                es_action_int = 127  # Todas ON
            # Converte para lista binária
            es_action = [(es_action_int >> i) & 1 for i in range(self.n_gnbs)]
            ts_actions = discrete_actions[1:]

        # Formata ações para NS3
        actions_list = []

        # Ações ES (tipo 0)
        for cell_idx, state in enumerate(es_action):
            cell_id = self.cellList[cell_idx]
            actions_list.append((0, cell_id, state))

        # Ações TS (tipo 1) - apenas se diferente de 0 (nenhuma ação)
        for ue_idx, target_cell in enumerate(ts_actions):
            if target_cell > 0:  # 0 = sem handover
                ue_imsi = ue_idx + 1
                target_cell_id = self.cellList[target_cell - 1] if target_cell <= len(self.cellList) else 0
                if target_cell_id > 0:
                    actions_list.append((1, ue_imsi, target_cell_id))
                    self.episode_handovers += 1

        return actions_list

    def _compute_reward(self) -> float:
        """Calcula recompensa combinada ES + TS"""
        reward = 0.0

        try:
            # Componentes da recompensa ES
            throughput = self.observations.get('SUM_QosFlow.PdcpPduVolumeDL_Filter', pd.Series([0])).iloc[0]
            rlf = self.observations.get('SUM_RLF_VALUE', pd.Series([0])).iloc[0]
            es_cost = self.observations.get('SUM_ES_ON_COST', pd.Series([0])).iloc[0]
            latency_us = self.observations.get('AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)', pd.Series([0])).iloc[0]
            zero_count = self.observations.get('ZERO_COUNT', pd.Series([0])).iloc[0]

            # Normalização
            throughput_norm = throughput / 1e6 if throughput > 0 else 0

            # Penalidade de latência (> 50ms)
            latency_ms = latency_us / 1000.0
            latency_penalty = 0.0
            if latency_ms > 50.0:
                latency_penalty = 20.0 * (latency_ms - 50.0) / 50.0

            # Recompensa ES
            reward_es = (
                0.31 * throughput_norm
                - 0.19 * (es_cost + zero_count)
                - 0.2 * rlf
                - 0.1 * es_cost
                - latency_penalty
            )

            # Recompensa TS (simplificada)
            reward_ts = 0.0

            reward = reward_es + self.ts_reward_weight * reward_ts

        except Exception as e:
            self.logger.warning(f"Erro ao calcular recompensa: {e}")
            reward = -1.0

        return float(reward)

    def step(self, action):
        """Executa um passo no ambiente"""
        self.num_steps += 1

        # Converte e aplica ação
        actions = self._compute_action(action)

        # Executa step base (envia ações para NS3)
        obs, reward, terminated, truncated, info = super().step(actions)

        # Atualiza observações
        self._update_observations()

        # Calcula recompensa
        reward = self._compute_reward()

        # Coleta métricas
        latency_cell = self.observations.get('latency_cell_us', pd.Series([0])).iloc[0] if 'latency_cell_us' in self.observations.columns else 0
        latency_ue = self.observations.get('AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)', pd.Series([0])).iloc[0] if 'AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)' in self.observations.columns else 0

        self.episode_rewards.append(reward)
        self.episode_latencies.append(latency_ue)

        throughput = self.observations.get('SUM_QosFlow.PdcpPduVolumeDL_Filter', pd.Series([0])).iloc[0] if 'SUM_QosFlow.PdcpPduVolumeDL_Filter' in self.observations.columns else 0
        self.episode_throughputs.append(throughput)

        # Salva no banco de dados
        if self.metrics_db and self.episode_id:
            # Métricas do step
            es_action_str = str(actions[:self.n_gnbs])
            ts_action_str = str(actions[self.n_gnbs:])

            self.metrics_db.insert_step_metrics(
                episode_id=self.episode_id,
                step=self.num_steps,
                timestamp=int(self.last_timestamp) if hasattr(self, 'last_timestamp') else int(time.time() * 1000),
                reward=reward,
                latency_cell=latency_cell,
                latency_ue=latency_ue,
                throughput=throughput,
                rlf=self.observations.get('SUM_RLF_VALUE', pd.Series([0])).iloc[0] if 'SUM_RLF_VALUE' in self.observations.columns else 0,
                es_cost=self.observations.get('SUM_ES_ON_COST', pd.Series([0])).iloc[0] if 'SUM_ES_ON_COST' in self.observations.columns else 0,
                zero_count=int(self.observations.get('ZERO_COUNT', pd.Series([0])).iloc[0]) if 'ZERO_COUNT' in self.observations.columns else 0,
                action_es=es_action_str,
                action_ts=ts_action_str
            )

            # Métricas por célula
            for cell in self.cellList:
                cell_metrics = {
                    'eekpi': self.observations.get(f'EEKPI_RL_{cell}', pd.Series([0])).iloc[0] if f'EEKPI_RL_{cell}' in self.observations.columns else 0,
                    'es_cost': self.observations.get(f'ES_ON_COST_{cell}', pd.Series([0])).iloc[0] if f'ES_ON_COST_{cell}' in self.observations.columns else 0,
                    'qos_flow': self.observations.get(f'QosFlow.PdcpPduVolumeDL_Filter_{cell}', pd.Series([0])).iloc[0] if f'QosFlow.PdcpPduVolumeDL_Filter_{cell}' in self.observations.columns else 0,
                    'latency': self.observations.get(f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}', pd.Series([0])).iloc[0] if f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}' in self.observations.columns else 0,
                    'rlf_counter': self.observations.get(f'RLF_Counter_{cell}', pd.Series([0])).iloc[0] if f'RLF_Counter_{cell}' in self.observations.columns else 0,
                    'rlf_value': self.observations.get(f'RLF_VALUE_{cell}', pd.Series([0])).iloc[0] if f'RLF_VALUE_{cell}' in self.observations.columns else 0,
                    'rru_prbtotdl': self.observations.get(f'RRU_PRBTOTDL_{cell}', pd.Series([0])).iloc[0] if f'RRU_PRBTOTDL_{cell}' in self.observations.columns else 0,
                    'rru_prbused': self.observations.get(f'RRU.PrbUsedDl_{cell}', pd.Series([0])).iloc[0] if f'RRU.PrbUsedDl_{cell}' in self.observations.columns else 0,
                    'tb_64qam': self.observations.get(f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}', pd.Series([0])).iloc[0] if f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}' in self.observations.columns else 0,
                }
                self.metrics_db.insert_cell_metrics(
                    episode_id=self.episode_id,
                    step=self.num_steps,
                    cell_id=cell,
                    metrics=cell_metrics
                )

        # Obtém nova observação
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _update_observations(self):
        """Atualiza DataFrame de observações a partir do datalake"""
        try:
            if hasattr(self, 'datalake') and self.datalake:
                # Placeholder: aqui seria feita a leitura real do datalake
                # Similar ao que é feito em hierarchical_env.py
                pass
        except Exception as e:
            self.logger.warning(f"Erro ao atualizar observações: {e}")

    def close(self):
        """Fecha o ambiente e finaliza episódio"""
        # Finaliza episódio no banco de dados
        if self.metrics_db and self.episode_id:
            avg_latency = np.mean(self.episode_latencies) if self.episode_latencies else 0
            avg_throughput = np.mean(self.episode_throughputs) if self.episode_throughputs else 0
            total_reward = sum(self.episode_rewards)
            cells_on_avg = sum(self.cells_states.values()) / len(self.cells_states)

            self.metrics_db.end_episode(
                episode_id=self.episode_id,
                total_steps=self.num_steps,
                total_reward=total_reward,
                avg_latency=avg_latency,
                avg_throughput=avg_throughput,
                num_handovers=self.episode_handovers,
                cells_on_avg=cells_on_avg
            )

        super().close()


def create_scenario_config(env_config: Dict[str, str]) -> dict:
    """Cria configuração do cenário a partir das variáveis de ambiente"""
    return {
        "simTime": [float(env_config.get('SIM_TIME', '10.0'))],
        "ues": [int(env_config.get('NUM_UES', '9'))],
        "RngRun": [int(env_config.get('RNG_RUN', '400'))],
        "configuration": [int(env_config.get('CONFIGURATION', '1'))],
        "trafficModel": [int(env_config.get('TRAFFIC_MODEL', '3'))],
        "numberOfRaPreambles": [64],
        "reducedPmValues": [0],
        "outageThreshold": [2.0],
        "handoverMode": [env_config.get('HANDOVER_MODE', 'DynamicTtt')],
        "indicationPeriodicity": [float(env_config.get('INDICATION_PERIODICITY', '0.1'))],
        "controlFileName": ["hierarchical_actions.csv"],
        "useSemaphores": [int(env_config.get('USE_SEMAPHORES', '1'))],
        "positionAllocator": [int(env_config.get('POSITION_ALLOCATOR', '2'))],
        "nBsNoUesAlloc": [0],
        "minSpeed": [float(env_config.get('MIN_SPEED', '2.0'))],
        "maxSpeed": [float(env_config.get('MAX_SPEED', '4.0'))],
    }


def make_env(env_kwargs: dict, metrics_db: HierarchicalMetricsDB):
    """Factory function para criar ambiente"""
    def _init():
        return HierarchicalUAVEnv(**env_kwargs, metrics_db=metrics_db)
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Executa modelo SAC treinado no cenário hierarchical-xangai-UAV"
    )

    parser.add_argument("--env_file", type=str, default=str(ENV_FILE),
                        help="Caminho para arquivo .env de configuração")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Caminho para modelo SAC (sobrescreve .env)")
    parser.add_argument("--ns3_path", type=str, default=None,
                        help="Caminho para ns-3 (sobrescreve .env)")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Pasta de saída (sobrescreve .env)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Número de episódios a executar")
    parser.add_argument("--deterministic", action="store_true",
                        help="Usar ações determinísticas")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verbose")
    parser.add_argument("--export_csv", action="store_true", default=True,
                        help="Exportar métricas para CSV ao final")

    args = parser.parse_args()

    # Carrega configurações
    env_config = load_env_file(Path(args.env_file))

    # Sobrescreve com argumentos da linha de comando
    model_path = args.model_path or env_config.get('SAC_MODEL_PATH')
    ns3_path = args.ns3_path or env_config.get('NS3_PATH')
    output_folder = args.output_folder or env_config.get('OUTPUT_FOLDER', 'output_hierarchical_uav')
    num_episodes = args.num_episodes or int(env_config.get('NUM_EPISODES', '10'))
    deterministic = args.deterministic or env_config.get('DETERMINISTIC', 'true').lower() == 'true'
    verbose = args.verbose or env_config.get('VERBOSE', 'false').lower() == 'true'

    # Configuração de logging
    log_level = getattr(logging, env_config.get('LOG_LEVEL', 'INFO'))
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(env_config.get('LOG_FILE', 'hierarchical_run.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Validações
    if not model_path or not os.path.exists(model_path):
        logger.error(f"Modelo SAC não encontrado: {model_path}")
        sys.exit(1)

    if not ns3_path or not os.path.exists(ns3_path):
        logger.error(f"Caminho ns-3 não encontrado: {ns3_path}")
        sys.exit(1)

    # Cria diretório de saída
    os.makedirs(output_folder, exist_ok=True)

    # Inicializa banco de dados
    db_path = env_config.get('DATABASE_PATH', os.path.join(output_folder, 'hierarchical_metrics.db'))
    metrics_db = HierarchicalMetricsDB(db_path)

    # Configuração do cenário
    scenario_config = create_scenario_config(env_config)

    logger.info("=" * 60)
    logger.info("HIERARCHICAL SAC RUNNER - UAV MOBILITY SCENARIO")
    logger.info("=" * 60)
    logger.info(f"Modelo: {model_path}")
    logger.info(f"NS3 Path: {ns3_path}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"Episódios: {num_episodes}")
    logger.info(f"Determinístico: {deterministic}")

    # Carrega modelo SAC
    logger.info("Carregando modelo SAC...")
    try:
        model = SAC.load(model_path)
        logger.info("Modelo carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        sys.exit(1)

    # Cria ambiente
    env_kwargs = {
        'ns3_path': ns3_path,
        'scenario_configuration': scenario_config,
        'output_folder': output_folder,
        'optimized': env_config.get('OPTIMIZED', 'true').lower() == 'true',
        'do_heuristic': env_config.get('DO_HEURISTIC', 'false').lower() == 'true',
        'ts_reward_weight': float(env_config.get('TS_REWARD_WEIGHT', '1.0')),
        'verbose': verbose,
        'scenario_name': env_config.get('SCENARIO_NAME', 'scenario-hierarchical-xangai-UAV'),
    }

    logger.info("Criando ambiente...")
    env = DummyVecEnv([make_env(env_kwargs, metrics_db)])

    # Atualiza ambiente no modelo
    model.set_env(env)

    # Loop de execução
    all_rewards = []
    all_latencies = []

    for episode in range(num_episodes):
        logger.info(f"\n{'='*40}")
        logger.info(f"EPISÓDIO {episode + 1}/{num_episodes}")
        logger.info(f"{'='*40}")

        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            step += 1

            # Predição do modelo
            action, _ = model.predict(obs, deterministic=deterministic)

            # Executa ação
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]

            if verbose:
                logger.debug(f"  Step {step}: reward={reward[0]:.4f}")

            done = done[0]

        logger.info(f"Episódio {episode + 1} concluído: {step} steps, reward={episode_reward:.2f}")
        all_rewards.append(episode_reward)

    # Estatísticas finais
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS FINAIS")
    logger.info("=" * 60)
    logger.info(f"Episódios executados: {num_episodes}")
    logger.info(f"Recompensa média: {np.mean(all_rewards):.2f} (+/- {np.std(all_rewards):.2f})")
    logger.info(f"Recompensa máxima: {np.max(all_rewards):.2f}")
    logger.info(f"Recompensa mínima: {np.min(all_rewards):.2f}")

    # Exporta para CSV
    if args.export_csv:
        csv_path = env_config.get('CSV_METRICS_PATH', os.path.join(output_folder, 'qos_hierarchical_metrics_UAV.csv'))
        logger.info(f"\nExportando métricas para: {csv_path}")
        metrics_db.export_to_csv(csv_path)

    # Cleanup
    env.close()
    metrics_db.close()

    logger.info("\nExecução concluída!")


if __name__ == '__main__':
    main()
