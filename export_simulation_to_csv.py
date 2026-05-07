#!/usr/bin/env python3
"""
export_simulation_to_csv.py

Script para extrair dados de simulações NS3 e exportar para CSV.
Lê os arquivos de KPM gerados pelo NS3 e converte para o formato
compatível com qos_hierarchical_metrics_UAV.csv

Uso:
    python export_simulation_to_csv.py --sim_folder <pasta_da_simulacao> --output <arquivo.csv>
    python export_simulation_to_csv.py --sim_folder output_hierarchical_uav/always_on_xxx/ --output results_always_on.csv
"""

import argparse
import csv
import os
import sys
import glob
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd


class SimulationDataExtractor:
    """Extrai dados de uma simulação NS3 e exporta para CSV"""

    def __init__(self, sim_folder: str, verbose: bool = False):
        self.sim_folder = sim_folder
        self.verbose = verbose
        self.cell_list = [2, 3, 4, 5, 6, 7, 8]

        # Dados extraídos
        self.cu_up_data = {}  # {timestamp: {cell_id: {metric: value}}}
        self.cu_cp_data = {}
        self.du_data = {}
        self.bs_state_data = {}  # {timestamp: {cell_id: state}}

    def log(self, msg):
        if self.verbose:
            print(f"[INFO] {msg}")

    def parse_cu_up_files(self):
        """Parseia arquivos cu-up-cell-*.txt"""
        for file_path in glob.glob(os.path.join(self.sim_folder, 'cu-up-cell-*.txt')):
            try:
                cell_id = int(file_path.split('cell-')[-1].split('.')[0])
                self.log(f"Parsing {file_path} (cell {cell_id})")

                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = int(row.get('timestamp', 0))
                        if ts not in self.cu_up_data:
                            self.cu_up_data[ts] = {}
                        if cell_id not in self.cu_up_data[ts]:
                            self.cu_up_data[ts][cell_id] = {}

                        # Extrai métricas - nomes com sufixos entre parênteses
                        latency = self._safe_float(row.get('DRB.PdcpSduDelayDl.UEID(pdcpLatency)', 0))
                        throughput = self._safe_float(row.get('DRB.PdcpSduBitRateDl.UEID(pdcpThroughput)', 0))
                        qos_flow = self._safe_float(row.get('QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)', 0))

                        # Acumula valores por timestamp/célula (soma de todos os UEs)
                        existing = self.cu_up_data[ts][cell_id]
                        existing['DRB.PdcpSduDelayDl.UEID (pdcpLatency)'] = max(
                            existing.get('DRB.PdcpSduDelayDl.UEID (pdcpLatency)', 0), latency)
                        existing['DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)'] = (
                            existing.get('DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)', 0) + throughput)
                        existing['QosFlow.PdcpPduVolumeDL_Filter'] = (
                            existing.get('QosFlow.PdcpPduVolumeDL_Filter', 0) + qos_flow)
            except Exception as e:
                self.log(f"Erro ao parsear {file_path}: {e}")

    def parse_cu_cp_files(self):
        """Parseia arquivos cu-cp-cell-*.txt"""
        for file_path in glob.glob(os.path.join(self.sim_folder, 'cu-cp-cell-*.txt')):
            try:
                cell_id = int(file_path.split('cell-')[-1].split('.')[0])
                self.log(f"Parsing {file_path} (cell {cell_id})")

                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = int(row.get('timestamp', 0))
                        if ts not in self.cu_cp_data:
                            self.cu_cp_data[ts] = {}
                        if cell_id not in self.cu_cp_data[ts]:
                            self.cu_cp_data[ts][cell_id] = {}

                        self.cu_cp_data[ts][cell_id].update({
                            'L3 serving SINR': self._safe_float(row.get('L3 serving SINR', 0)),
                            'numActiveUes': self._safe_int(row.get('numActiveUes', 0)),
                        })
            except Exception as e:
                self.log(f"Erro ao parsear {file_path}: {e}")

    def parse_du_files(self):
        """Parseia arquivos du-cell-*.txt"""
        for file_path in glob.glob(os.path.join(self.sim_folder, 'du-cell-*.txt')):
            try:
                cell_id = int(file_path.split('cell-')[-1].split('.')[0])
                self.log(f"Parsing {file_path} (cell {cell_id})")

                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = int(row.get('timestamp', 0))
                        if ts not in self.du_data:
                            self.du_data[ts] = {}
                        if cell_id not in self.du_data[ts]:
                            self.du_data[ts][cell_id] = {}

                        # Acumula valores por timestamp/célula
                        existing = self.du_data[ts][cell_id]
                        existing['TB.TotNbrDl.1'] = existing.get('TB.TotNbrDl.1', 0) + self._safe_float(row.get('TB.TotNbrDl.1', 0))
                        existing['RRU.PrbUsedDl'] = existing.get('RRU.PrbUsedDl', 0) + self._safe_float(row.get('RRU.PrbUsedDl', 0))
                        existing['TB.TotNbrDlInitial.Qpsk'] = existing.get('TB.TotNbrDlInitial.Qpsk', 0) + self._safe_float(row.get('TB.TotNbrDlInitial.Qpsk', 0))
                        existing['TB.TotNbrDlInitial.16Qam'] = existing.get('TB.TotNbrDlInitial.16Qam', 0) + self._safe_float(row.get('TB.TotNbrDlInitial.16Qam', 0))
                        existing['TB.TotNbrDlInitial.64Qam'] = existing.get('TB.TotNbrDlInitial.64Qam', 0) + self._safe_float(row.get('TB.TotNbrDlInitial.64Qam', 0))
                        existing['DRB.MeanActiveUeDl'] = max(existing.get('DRB.MeanActiveUeDl', 0), self._safe_float(row.get('DRB.MeanActiveUeDl', 0)))
                        existing['DRB.UEThpDl.UEID'] = existing.get('DRB.UEThpDl.UEID', 0) + self._safe_float(row.get('DRB.UEThpDl.UEID', 0))
                        # QosFlow do DU (agregado por célula)
                        existing['QosFlow.PdcpPduVolumeDL_Filter'] = existing.get('QosFlow.PdcpPduVolumeDL_Filter', 0) + self._safe_float(row.get('QosFlow.PdcpPduVolumeDL_Filter', 0))
            except Exception as e:
                self.log(f"Erro ao parsear {file_path}: {e}")

    def parse_bs_state(self):
        """Parseia arquivo bsState.txt"""
        bs_state_path = os.path.join(self.sim_folder, 'bsState.txt')
        if not os.path.exists(bs_state_path):
            self.log("bsState.txt não encontrado")
            return

        try:
            self.log(f"Parsing {bs_state_path}")
            with open(bs_state_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=' ')
                for row in reader:
                    ts = int(row.get('UNIX', 0))
                    cell_id = int(row.get('Id', 0))
                    state = int(row.get('State', 1))

                    if ts not in self.bs_state_data:
                        self.bs_state_data[ts] = {}
                    self.bs_state_data[ts][cell_id] = state
        except Exception as e:
            self.log(f"Erro ao parsear bsState.txt: {e}")

    def _safe_float(self, value, default=0.0):
        try:
            if value is None or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default=0):
        try:
            if value is None or value == '':
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def extract_all(self):
        """Extrai dados de todos os arquivos"""
        self.log(f"Extraindo dados de {self.sim_folder}")
        self.parse_cu_up_files()
        self.parse_cu_cp_files()
        self.parse_du_files()
        self.parse_bs_state()

        # Combina todos os timestamps
        all_timestamps = set()
        all_timestamps.update(self.cu_up_data.keys())
        all_timestamps.update(self.cu_cp_data.keys())
        all_timestamps.update(self.du_data.keys())
        all_timestamps.update(self.bs_state_data.keys())

        self.log(f"Total de timestamps: {len(all_timestamps)}")
        return sorted(all_timestamps)

    def compute_metrics_for_timestamp(self, ts: int, step: int) -> Dict[str, Any]:
        """Calcula métricas agregadas para um timestamp"""
        row = {'timestamp': ts, 'step': step}

        # Métricas por célula
        sum_qos = 0
        sum_tb = 0
        sum_rlf = 0
        sum_es_cost = 0
        latency_values = []
        cells_off = 0

        for cell in self.cell_list:
            # Dados CU-UP
            cu_up = self.cu_up_data.get(ts, {}).get(cell, {})
            latency = cu_up.get('DRB.PdcpSduDelayDl.UEID (pdcpLatency)', 0)

            # Dados DU
            du = self.du_data.get(ts, {}).get(cell, {})
            tb_tot = du.get('TB.TotNbrDl.1', 0.00001)
            rru_prbused = du.get('RRU.PrbUsedDl', 0)
            tb_qpsk = du.get('TB.TotNbrDlInitial.Qpsk', 0)
            tb_16qam = du.get('TB.TotNbrDlInitial.16Qam', 0)
            tb_64qam = du.get('TB.TotNbrDlInitial.64Qam', 0)

            # QosFlow: prioriza DU (agregado), fallback para CU-UP
            qos_flow = du.get('QosFlow.PdcpPduVolumeDL_Filter', 0)
            if qos_flow == 0:
                qos_flow = cu_up.get('QosFlow.PdcpPduVolumeDL_Filter', 0)

            # Dados CU-CP
            cu_cp = self.cu_cp_data.get(ts, {}).get(cell, {})
            sinr = cu_cp.get('L3 serving SINR', 0)

            # Estado da célula
            state = self.bs_state_data.get(ts, {}).get(cell, 1)  # 1=ON por padrão

            # Calcula métricas derivadas
            if tb_tot == 0:
                tb_tot = 0.00001
            eekpi = qos_flow / tb_tot if tb_tot > 0 else 0

            # RRU_PRBTOTDL: percentual de PRB usado
            rru_prbtot = (rru_prbused / 139) * 100 if rru_prbused > 0 else 0

            # TB_TOTNBRDLINITIAL_64QAM_RATIO
            tb_sum = tb_qpsk + tb_16qam + tb_64qam
            tb_64qam_ratio = tb_64qam / tb_sum if tb_sum > 0 else 0

            # ES_ON_COST: custo de energia (simplificado)
            es_cost = 0.81 if state == 1 else 0  # Custo se célula ON

            # RLF baseado em SINR < -5 dB
            rlf_value = 1 if sinr < -5 else 0
            rlf_counter = 100 if sinr < -5 else 0

            # Preenche métricas por célula
            row[f'EEKPI_RL_{cell}'] = eekpi
            row[f'ES_ON_COST_{cell}'] = es_cost
            row[f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'] = qos_flow
            row[f'DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}'] = latency
            row[f'RLF_Counter_{cell}'] = rlf_counter
            row[f'RLF_VALUE_{cell}'] = rlf_value
            row[f'RRU_PRBTOTDL_{cell}'] = rru_prbtot
            row[f'RRU.PrbUsedDl_{cell}'] = rru_prbused
            row[f'TB_TOTNBRDLINITIAL_64QAM_RATIO_{cell}'] = tb_64qam_ratio

            # Acumula para métricas agregadas
            sum_qos += qos_flow
            sum_tb += tb_tot
            sum_rlf += rlf_value
            sum_es_cost += es_cost
            if latency > 0:
                latency_values.append(latency)
            if state == 0:
                cells_off += 1

        # Métricas agregadas
        row['SUM_QosFlow.PdcpPduVolumeDL_Filter'] = sum_qos
        row['SUM_RLF_VALUE'] = sum_rlf
        row['SUM_TB.TotNbrDl.1'] = sum_tb
        row['SUM_ES_ON_COST'] = sum_es_cost
        row['AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)'] = np.mean(latency_values) if latency_values else 0
        row['ZERO_COUNT'] = cells_off

        # Latências em microsegundos
        row['latency_cell_us'] = np.mean(latency_values) if latency_values else 0
        row['latency_ue_us'] = np.mean(latency_values) if latency_values else 0

        # Calcula recompensa
        throughput_norm = sum_qos / 1e6 if sum_qos > 0 else 0
        row['reward'] = (
            0.31 * throughput_norm
            - 0.19 * (sum_es_cost + cells_off)
            - 0.2 * sum_rlf
            - 0.1 * sum_es_cost
        )

        return row

    def export_to_csv(self, output_path: str):
        """Exporta dados para CSV"""
        timestamps = self.extract_all()

        if not timestamps:
            print("Nenhum dado encontrado para exportar")
            return False

        # Define colunas
        columns = ["timestamp", "step"]

        for metric in ["EEKPI_RL", "ES_ON_COST", "QosFlow.PdcpPduVolumeDL_Filter",
                       "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "RLF_Counter",
                       "RLF_VALUE", "RRU_PRBTOTDL", "RRU.PrbUsedDl",
                       "TB_TOTNBRDLINITIAL_64QAM_RATIO"]:
            for cell in self.cell_list:
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

        # Gera linhas
        rows = []
        for step, ts in enumerate(timestamps, 1):
            row = self.compute_metrics_for_timestamp(ts, step)
            rows.append(row)

        # Escreve CSV
        df = pd.DataFrame(rows)

        # Reordena colunas
        ordered_cols = [c for c in columns if c in df.columns]
        df = df[ordered_cols]

        df.to_csv(output_path, index=False)
        print(f"Exportado {len(rows)} registros para {output_path}")

        # Estatísticas
        print(f"\nEstatísticas:")
        print(f"  Timestamps: {len(timestamps)}")
        print(f"  Primeiro timestamp: {timestamps[0]}")
        print(f"  Último timestamp: {timestamps[-1]}")
        if 'reward' in df.columns:
            print(f"  Reward médio: {df['reward'].mean():.4f}")
            print(f"  Reward min: {df['reward'].min():.4f}")
            print(f"  Reward max: {df['reward'].max():.4f}")
        if 'SUM_QosFlow.PdcpPduVolumeDL_Filter' in df.columns:
            print(f"  Throughput médio: {df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].mean():.2f}")

        return True


def export_from_database(db_path: str, output_path: str):
    """Exporta dados diretamente de um banco SQLite"""
    if not os.path.exists(db_path):
        print(f"Banco de dados não encontrado: {db_path}")
        return False

    conn = sqlite3.connect(db_path)

    # Lista tabelas
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tabelas encontradas: {[t[0] for t in tables]}")

    # Exporta cada tabela para CSV separado
    base_path = os.path.splitext(output_path)[0]

    for table_name in [t[0] for t in tables]:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        table_csv_path = f"{base_path}_{table_name}.csv"
        df.to_csv(table_csv_path, index=False)
        print(f"Exportado {len(df)} registros de {table_name} para {table_csv_path}")

    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Exporta dados de simulação NS3 para CSV"
    )

    parser.add_argument("--sim_folder", type=str, required=True,
                        help="Pasta da simulação NS3")
    parser.add_argument("--output", type=str, default=None,
                        help="Arquivo CSV de saída")
    parser.add_argument("--from_db", action="store_true",
                        help="Exporta do banco SQLite (database.db)")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verbose")

    args = parser.parse_args()

    # Valida pasta
    if not os.path.isdir(args.sim_folder):
        print(f"Pasta não encontrada: {args.sim_folder}")
        sys.exit(1)

    # Define output
    if args.output:
        output_path = args.output
    else:
        folder_name = os.path.basename(os.path.normpath(args.sim_folder))
        output_path = f"exported_{folder_name}.csv"

    if args.from_db:
        # Exporta do banco SQLite
        db_path = os.path.join(args.sim_folder, 'database.db')
        export_from_database(db_path, output_path)
    else:
        # Exporta dos arquivos txt
        extractor = SimulationDataExtractor(args.sim_folder, args.verbose)
        extractor.export_to_csv(output_path)


if __name__ == '__main__':
    main()
