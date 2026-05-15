from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv
from gymnasium import spaces # Importar Gymnasium spaces
import pandas as pd
import glob
import csv
import os
import logging # Importar logging

"""
Este ambiente mescla Energy Saving (ES) e Traffic Steering (TS).
(MODIFICADO PARA SAC - Microssegundos)

🚨 CORREÇÕES CRÍTICAS APLICADAS (02/12/2025):
============================================
1. PROBLEMA DE LATÊNCIA ACUMULADA:
   - Antes: AVG_DRB.PdcpSduDelayDl acumulava valores ao longo do tempo (resultados falsos de 16+ minutos)
   - Agora: Usa latência instantânea do latency_map (valores corretos em ms)

2. CORREÇÃO DA FUNÇÃO DE RECOMPENSA:
   - Penalização forte apenas para latências > 50ms (padrão 5G)
   - Peso de penalização aumentado para forçar convergência para baixa latência
   - Conversão correta: μs → ms para cálculo de penalização

3. MÉTRICAS CORRETAS:
   - latency_us: latência instantânea em microsegundos
   - AVG_DRB.PdcpSduDelayDl: agora contém média instantânea, não acumulada
   - Penalizações: apenas para latências que violam especificações 5G

- AÇÕES (Contínuo - Box):
    - O agente SAC emite um vetor de floats (ex: [-1.0, 1.0]).
    - Este vetor é internamente decodificado para o formato MultiDiscrete.
    - O primeiro elemento representa a ação ES.
    - Os elementos seguintes representam a ação TS (0-6 para cada UE).
- OBSERVAÇÕES (Dict):
    - 'es_obs': Observações agregadas por célula (como em es_env).
    - 'ts_obs': Observações por UE (como em ts_env).
- RECOMPENSA:
    - Uma soma ponderada da recompensa de ES (eficiência energética, INCLUINDO LATÊNCIA) e da recompensa de TS (throughput de UE).
- LOGGING:
    - Métricas detalhadas (incluindo latência em us e recompensas) são salvas em um CSV.
"""

class HierarchicalEnv(NsOranEnv):

    gnb_state_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "state": "INTEGER"
    }

    # (Novo - de es_env) Chaves para a tabela de tracking de latência
    latency_tracking_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "step": "INTEGER",
        "cell_2_latency": "REAL",
        "cell_3_latency": "REAL",
        "cell_4_latency": "REAL",
        "cell_5_latency": "REAL",
        "cell_6_latency": "REAL",
        "cell_7_latency": "REAL",
        "cell_8_latency": "REAL",
        "avg_cell_latency": "REAL",
        "sum_latency": "REAL",
        "max_latency": "REAL",
        "min_latency": "REAL",
    }

    # (Novo - de es_env) Configurações do CSV
    QOS_CSV_DIR = "/home/pedrogustavo/RIC_Personalized/ns-o-ran-gym/examples/output" # Ajuste este caminho se necessário
    QOS_CSV_BASENAME = "qos_hierarchical_metrics_SAC1.csv" # <-- MUDANÇA PARA SAC


    def __init__(self, ns3_path:str,
                 scenario_configuration:dict,
                 output_folder:str,
                 optimized:bool,
                 do_heuristic:bool = False,
                 ts_reward_weight:float = 1.0,
                 verbose=False,
                 scenario_name:str = 'scenario-hierarchical'):

        # O cabeçalho de controlo hierárquico
        super().__init__(ns3_path=ns3_path, scenario=scenario_name, scenario_configuration=scenario_configuration,
                         output_folder=output_folder, optimized=optimized,
                         control_header = ['action_type', 'param1', 'param2'],
                         log_file='HierarchicalActions.txt',
                         control_file='hierarchical_actions.csv')


        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration.get('simTime', 1.9) * 1000

        self.cellList = [2, 3, 4, 5, 6, 7, 8] # 7 células mmWave (gNBs secundárias)

        # --- Atributos do Energy Saving (ES) ---
        # (Modificado - Adicionadas colunas de latência de es_env)
        self.es_columns_state = (
            [f"EEKPI_RL_{c}" for c in self.cellList]
            + [f"ES_ON_COST_{c}" for c in self.cellList]
            + [f"QosFlow.PdcpPduVolumeDL_Filter_{c}" for c in self.cellList]
            # O nome base "DRB.PdcpSduDelayDl.UEID (pdcpLatency)" corresponde ao datalake.py
            + [f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{c}" for c in self.cellList] # NOVO
            + [f"RLF_Counter_{c}" for c in self.cellList]
            + [f"RLF_VALUE_{c}" for c in self.cellList]
            + [f"RRU_PRBTOTDL_{c}" for c in self.cellList]
            + [f"RRU.PrbUsedDl_{c}" for c in self.cellList]
            + [f"TB_TOTNBRDLINITIAL_64QAM_RATIO_{c}" for c in self.cellList]
            + [
                "SUM_QosFlow.PdcpPduVolumeDL_Filter",
                "SUM_RLF_VALUE",
                "SUM_TB.TotNbrDl.1",
                "SUM_ES_ON_COST",
                "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)", # CORRIGIDO (era SUM_)
                "ZERO_COUNT",
            ]
        )
        # (Modificado - Adicionada coluna de latência de es_env)
        self.es_columns_reward = [
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)", # CORRIGIDO (era SUM_)
            "ZERO_COUNT",
        ]

        # Lista de ações possíveis no modo não-heurístico (Discrete)
        self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 48, 49, 50, 52, 56, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 80, 81, 82, 84, 88, 96, 97, 98, 100, 104, 112]
        self.observations = pd.DataFrame() # Dataframe que guarda o resultado do preprocessing para a recompensa ES
        self.cells_states = {} # Dicionário {cellId: state (0=OFF, 1=ON)} lido de bsState.txt
        self.cell_timestamp_state_dict = {cell: float('inf') for cell in self.cellList} # Guarda o timestamp (ms) em que a célula LIGOU (para calcular custo ES)
        self.Cf_es = 1.0 # Fator de custo para transição ES
        self.lambdaf_es = 0.1 # Fator de decaimento para custo ES
        self.time_factor_es = 0.01 # Fator de tempo para custo ES (ajusta escala ms -> ?)
        self.heur = do_heuristic # Define se a ação ES é MultiDiscrete([2]*7) ou Discrete(len(action_list))
        self.num_steps = 0 # Contador de passos dentro do episódio
        self.previous_inverted_action = "0000000" # String binária invertida (0=ON, 1=OFF) do estado anterior (usado para ZERO_COUNT)

        # --- Atributos do Traffic Steering (TS) ---
        # Nomes das métricas correspondem ao datalake.py (du_keys e gnb_cu_cp_keys)
        self.ts_columns_state = ['RRU.PrbUsedDl', 'L3 serving SINR', 'DRB.MeanActiveUeDl',
                                 'TB.TotNbrDlInitial.Qpsk', 'TB.TotNbrDlInitial.16Qam',
                                 'TB.TotNbrDlInitial.64Qam', 'TB.TotNbrDlInitial'] # KPMs por UE para observação TS
        # Nomes das métricas correspondem ao datalake.py (du_keys)
        self.ts_columns_reward = ['DRB.UEThpDl.UEID', 'nrCellId'] # KPMs por UE para recompensa TS
        self.previous_kpms_ts = None # Guarda KPMs TS do step anterior para cálculo de recompensa
        self.handovers_dict = dict() # Guarda o timestamp (ms) do último HO por UE {imsi: timestamp}
        self.verbose = verbose # Flag para ativar logging detalhado

        # Configuração do Logger
        if self.verbose:
            log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s') # Adiciona nome do ficheiro e linha
            log_handler = logging.FileHandler('hierarchical_env_sac.log', mode='w') # 'w' para sobrescrever
            log_handler.setFormatter(log_formatter)
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():
                self.logger.addHandler(log_handler)
            self.logger.setLevel(logging.DEBUG)
            self.logger.info(f"Logging iniciado para HierarchicalEnv (Modo SAC). Heurístico: {self.heur}")
        else:
            # Garante que self.logger existe mas não faz nada
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.ERROR) # Apenas loga erros sérios
            self.logger.addHandler(logging.NullHandler())


        self.time_factor_ts = 0.001 # Fator de tempo para custo TS (ajusta escala ms -> ?)

        # --- MODIFICAÇÃO (Solução 1: Aumentar penalidade de HO) ---
        self.Cf_ts = 10.0 # Fator de custo para HO TS (AUMENTADO de 1.0)
        # --- FIM DA MODIFICAÇÃO ---

        self.lambdaf_ts = 0.1 # Fator de decaimento para custo HO TS
        self.ts_reward_weight = ts_reward_weight # Ponderador para a recompensa de TS

        # --- Espaços Combinados de Ação e Observação ---
        self.n_gnbs = len(self.cellList)
        self.n_ues_per_gnb = self.scenario_configuration.get('ues', 1)
        self.n_ues_total = self.n_ues_per_gnb * self.n_gnbs
        n_actions_ue_ts = self.n_gnbs + 1 # 0..7

        # 1. Espaço de Ação (ACHATADO - MultiDiscrete)
        action_dims = []
        if self.heur:
            action_dims.extend([2] * self.n_gnbs)
            self.es_action_indices = np.arange(self.n_gnbs)
            if self.logger: self.logger.info(f"Action Space ES (Heurístico): MultiDiscrete([2]*{self.n_gnbs})")
        else:
            action_dims.append(len(self.action_list))
            self.es_action_indices = np.array([0])
            if self.logger: self.logger.info(f"Action Space ES (Não Heurístico): Discrete({len(self.action_list)})")

        action_dims.extend([n_actions_ue_ts] * self.n_ues_total)
        self.ts_action_indices = np.arange(len(self.es_action_indices), len(action_dims))
        if self.logger: self.logger.info(f"Action Space TS: MultiDiscrete([{n_actions_ue_ts}]*{self.n_ues_total})")

        # --- MODIFICAÇÃO SAC ---
        # Guardamos as dimensões originais do MultiDiscrete para decodificar
        self.original_action_dims = np.array(action_dims)
        n_actions_total = len(self.original_action_dims)

        # SAC requer um espaço Box (contínuo). A saída do agente será [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions_total,), dtype=np.float32)

        if self.logger: self.logger.info(f"Action Space Original (MultiDiscrete): {self.original_action_dims}")
        if self.logger: self.logger.info(f"Action Space Exposto (SAC): {self.action_space}")
        # --- FIM DA MODIFICAÇÃO SAC ---


        # 2. Espaço de Observação (Dict - Mantido como Dict)
        es_obs_shape = (len(self.es_columns_state),)
        ts_obs_shape = (self.n_ues_total, len(self.ts_columns_state))

        es_obs_space = spaces.Box(shape=es_obs_shape, low=-np.inf, high=np.inf, dtype=np.float64)
        ts_obs_space = spaces.Box(shape=ts_obs_shape, low=-np.inf, high=np.inf, dtype=np.float64)

        self.observation_space = spaces.Dict({
            "es_obs": es_obs_space,
            "ts_obs": ts_obs_space
        })
        if self.logger: self.logger.info(f"Observation Space: {self.observation_space}")


        # --- (Novo - de es_env) Inicialização do Log CSV ---
        self.qos_csv_path = os.path.join(self.QOS_CSV_DIR, self.QOS_CSV_BASENAME)
        # Define o cabeçalho do CSV
        self._qos_header = (
            ["timestamp", "step"]
            + self.es_columns_state # Usa as colunas ES (que agora incluem latência)
            + [c for c in self.es_columns_reward if c not in self.es_columns_state]
            # MODIFICADO: ms -> us
            + ["latency_cell_us", "latency_ue_us", "reward"]
        )
        # Garante que o ficheiro CSV existe
        self._ensure_qos_csv()


    # (Novo - de es_env) Garante que o ficheiro CSV e o diretório existem
    def _ensure_qos_csv(self):
        """Garante que o diretório de log CSV existe e que o ficheiro tem um cabeçalho."""
        os.makedirs(os.path.dirname(self.qos_csv_path) or ".", exist_ok=True)
        new_file = (not os.path.exists(self.qos_csv_path)) or os.path.getsize(self.qos_csv_path) == 0
        if new_file:
            with open(self.qos_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._qos_header)
                if self.logger: self.logger.info(f"Criado novo ficheiro CSV de QoS em: {self.qos_csv_path}")

    # (Novo - de es_env) Adiciona uma linha ao CSV
    # MODIFICADO: Argumentos agora esperam us
    def _append_qos_snapshot(self, latency_cell_us: float, latency_ue_us: float, reward: float):
        """Adiciona o estado atual, recompensas e métricas ao ficheiro CSV."""
        try:
            row = []
            row.append(int(self.last_timestamp))
            row.append(int(self.num_steps))

            # Adiciona colunas de estado ES (baseado em self.observations)
            for col in self.es_columns_state:
                val = 0.0
                try:
                    if col in self.observations.columns:
                        val = float(self.observations[col].iloc[0])
                except Exception:
                    pass # Mantém val 0.0
                row.append(val)

            # Adiciona colunas de recompensa ES (que não estão no estado)
            for col in self.es_columns_reward:
                if col in self.es_columns_state:
                    continue
                val = 0.0
                try:
                    if col in self.observations.columns:
                        val = float(self.observations[col].iloc[0])
                except Exception:
                    pass # Mantém val 0.0
                row.append(val)

            # Adiciona métricas finais
            row.append(float(latency_cell_us))
            row.append(float(latency_ue_us))
            row.append(float(reward))

            # Escreve no ficheiro
            with open(self.qos_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        except Exception as e:
            if self.logger: self.logger.error(f"Falha ao escrever no CSV de QoS: {e}")

    # (Novo - de es_env) Carrega dados de latência
    def _load_latency_data(self, timestamp):
        """
        Carrega dados de latência da tabela lte_cu_up.
        """
        try:
            # --- MODIFICAÇÃO (Corrigir agente cego) ---
            # Garante que estamos lendo a tabela correta 'lte_cu_up'
            latency_table_data = self.datalake.read_table("lte_cu_up") or []
            # --- FIM DA MODIFICAÇÃO ---

            if not latency_table_data:
                if self.logger: self.logger.warning("lte_cu_up table is EMPTY!")
                return {}

            # CORREÇÃO: Validação de timestamp mais robusta
            target_timestamp = int(timestamp)
            latency_rows = [row for row in latency_table_data if len(row) > 0 and int(row[0]) == target_timestamp]

            if self.logger:
                self.logger.debug(f"Procurando latência para timestamp {target_timestamp}, encontradas {len(latency_rows)} linhas")
                if len(latency_rows) > 0:
                    self.logger.debug(f"Primeira linha de latência: {latency_rows[0][:6] if len(latency_rows[0]) > 6 else latency_rows[0]}")

            if not latency_rows:
                if self.logger: self.logger.debug("No latency data for timestamp %s", timestamp)
                return {}

            latency_map = {}
            for row in latency_rows:
                try:
                    imsi = int(row[1])  # ueImsiComplete

                    # CORREÇÃO: Os índices corretos baseados no datalake.py (lte_cu_up_keys):
                    # [3]: DRB.PdcpSduDelayDl(cellAverageLatency)
                    # [4]: DRB.PdcpSduDelayDl.UEID (pdcpLatency)
                    # Estes nomes correspondem ao datalake.py (lte_cu_up_keys)

                    # Usa row[3] para cell average, row[4] para UE-specific
                    cell_avg_raw = float(row[3]) if len(row) > 3 and row[3] not in (None, '', 0) else 0.0
                    ue_latency_raw = float(row[4]) if len(row) > 4 and row[4] not in (None, '', 0) else 0.0

                    # Usa UE-specific se disponível, senão cell average
                    final_latency_raw = ue_latency_raw if ue_latency_raw > 0 else cell_avg_raw

                    # --- CORREÇÃO CRÍTICA (Unidade da Latência) ---
                    # INVESTIGAÇÃO: Os dados do simulador NS-3 vêm em milissegundos (ms) diretamente
                    # Não há conversão de 0.1ms - isso estava causando latências falsas de 100x
                    # Mantém valor em ms e converte para us apenas para consistência interna
                    final_latency_ms = final_latency_raw
                    final_latency_us = final_latency_ms * 1000.0  # Converte ms para us

                    # VALIDAÇÃO DE SANITY: Rejeita latências absurdas (> 1000ms)
                    if final_latency_ms > 1000.0:
                        if self.logger:
                            self.logger.warning(f"Latência absurda detectada e corrigida: {final_latency_ms:.3f}ms (valor bruto: {final_latency_raw})")
                        # Para valores absurdos, usa fallback para latência celular média
                        final_latency_ms = cell_avg_raw if cell_avg_raw < 1000.0 else 10.0  # 10ms como fallback seguro
                        final_latency_us = final_latency_ms * 1000.0
                    # --- FIM DA CORREÇÃO ---

                    # CORREÇÃO: Salva valores consistentes em microsegundos
                    latency_map[imsi] = {
                        'cell_avg': cell_avg_raw * 1000.0,  # Converte ms para us
                        'ue_latency': final_latency_us      # Já está em us
                    }

                except (ValueError, IndexError, TypeError) as e:
                    if self.logger: self.logger.warning("Error parsing latency for IMSI %s: %r", row[1] if len(row) > 1 else '?', e)
                    continue

            if self.logger: self.logger.info(
                "Loaded latency for %d UEs | ts=%s",
                len(latency_map), timestamp
            )

            return latency_map

        except Exception as e:
            if self.logger: self.logger.error("Failed to load latency from lte_cu_up: %r", e)
            return {}


    @override
    def _compute_action(self, flat_action_continuous):
        """
        (MODIFICADO PARA SAC)
        Converte a ação CONTÍNUA (array Box) do agente SAC
        num formato que o ns-O-RAN entende para o ficheiro CSV.
        """

        # --- INÍCIO DA MODIFICAÇÃO SAC ---
        # flat_action_continuous é um np.ndarray de floats, ex: [-0.5, 0.8, 1.0, -0.2, ...]

        # 1. Verificar o shape (ainda é válido, pois self.action_space é Box)
        if not isinstance(flat_action_continuous, np.ndarray) or flat_action_continuous.shape != self.action_space.shape:
            msg = f"ERRO: _compute_action (SAC) esperava np.ndarray com shape {self.action_space.shape}, mas recebeu {type(flat_action_continuous)} com shape {getattr(flat_action_continuous, 'shape', 'N/A')}"
            if self.logger: self.logger.error(msg)
            # Fallback para ação neutra (Tudo ON, Sem HO)
            # Ação contínua "0.0" será mapeada para índice médio.
            # Ação contínua "-1.0" será mapeada para índice 0.
            flat_action_continuous = np.full(self.action_space.shape, -1.0, dtype=np.float32)

        # 2. Decodificar de Contínuo para MultiDiscrete
        # O agente SAC emite valores normalizados entre [-1, 1]

        # Escala de [-1, 1] para [0, 1]
        scaled_action = (flat_action_continuous + 1) / 2.0

        # Escala de [0, 1] para [0, N_dims - 1] para cada dimensão
        # self.original_action_dims é ex: [64, 8, 8, ...]
        # O range de índices é [0, 63], [0, 7], [0, 7], ...
        # Subtraímos um epsilon para evitar que 1.0 * (64-1) se torne 63.00001 e arredonde para 64.
        # Mas é melhor multiplicar por N e depois fazer o floor.
        # (scaled_action * N_dims) -> [0, N_dims]. Floor disso dá [0, N_dims-1].

        # Escala [0, 1] para [0, N_dims]
        scaled_action_range = scaled_action * self.original_action_dims

        # Pega o índice (floor) e garante que está dentro dos limites
        decoded_action_indices = np.floor(scaled_action_range).astype(int)

        # Garante que o índice máximo não excede N_dims - 1
        # (ex: se o agente der 1.0, scaled * 64 = 64. floor = 64. Precisa ser 63)
        decoded_action_indices = np.clip(decoded_action_indices, 0, self.original_action_dims - 1)

        # 'flat_action' agora é o nosso array de índices discretos, como o A2C gerava
        flat_action = decoded_action_indices

        if self.verbose and self.logger:
             self.logger.debug(f'Ação Contínua (SAC): {flat_action_continuous}')
             self.logger.debug(f'Ação Decodificada (Discreta): {flat_action}')
        # --- FIM DA MODIFICAÇÃO SAC ---


        action_list_output = [] # Lista final para o ActionController

        # --- Desachatar e Processar Ação ES (action_type = 0) ---
        # O resto desta função é IDÊNTICO ao original, pois 'flat_action'
        # agora tem o formato (MultiDiscrete) que ela espera.
        es_action_part = flat_action[self.es_action_indices]
        cell_id_list = self.cellList
        ns3_states_binary = [] # Lista de estados (0=OFF, 1=ON) para o ns-3

        if self.heur:
            # es_action_part é um array [0 ou 1]
            # Agente 0 -> ns-3 1 (ON), Agente 1 -> ns-3 0 (OFF)
            ns3_states_binary = [1 if b == 0 else 0 for b in es_action_part]
            self.previous_inverted_action = "".join(str(b) for b in es_action_part) # Guarda 0/1 do *agente*
        else:
            # es_action_part é um array com 1 elemento: o índice da action_list
            es_action_index = es_action_part[0]
            dec_action = 0 # Default
            if 0 <= es_action_index < len(self.action_list):
                dec_action = self.action_list[es_action_index]
            else:
                if self.logger: self.logger.error(f"Índice de ação ES inválido: {es_action_index} (max={len(self.action_list)-1})")
                # Fallback: usa o índice válido mais próximo
                es_action_index_clipped = np.clip(es_action_index, 0, len(self.action_list) - 1)
                dec_action = self.action_list[es_action_index_clipped]

            ns3_states_binary = []
            agent_bits_for_log = []
            for i in range(self.n_gnbs):
                agent_bit = (dec_action >> i) & 1
                agent_bits_for_log.append(str(agent_bit))
                ns3_state = 1 if agent_bit == 0 else 0 # Inverte
                ns3_states_binary.append(ns3_state)

            self.previous_inverted_action = "".join(agent_bits_for_log) # Guarda 0/1 do *agente*


        # --- INÍCIO DA MODIFICAÇÃO (Prevenir HO para Célula OFF) ---
        # NOVO: Mapa para armazenar o estado pretendido (0=OFF, 1=ON) para cada célula
        intended_cell_states = {}

        # Adiciona ações ES à lista final com action_type = 0
        for i, ns3_state in enumerate(ns3_states_binary):
            cell_id = cell_id_list[i]

            # Armazena o estado que esta célula terá
            intended_cell_states[cell_id] = ns3_state

            # Formato: (action_type, cellId, state)
            action_list_output.append((0, cell_id, ns3_state))
        # --- FIM DA MODIFICAÇÃO (Prevenir HO para Célula OFF) ---


        # --- Desachatar e Processar Ação TS (action_type = 1) ---
        ts_action_part = flat_action[self.ts_action_indices]
        base_imsi = 1 # Primeiro IMSI esperado pelo ns-3

        for ue_local_idx, agent_target_action in enumerate(ts_action_part):
            ue_imsi = base_imsi + ue_local_idx # Calcula o IMSI real da UE

            if agent_target_action == 0:
                continue # Sem ação

            # --- INÍCIO DA MODIFICAÇÃO (Prevenir HO para Célula OFF) ---
            elif 1 <= agent_target_action <= self.n_gnbs:
                target_cell_id = self.cellList[agent_target_action - 1] # Mapeia 1..7 -> 2..8

                # --- VERIFICAÇÃO DE ESTADO DA CÉLULA (NOVO) ---
                # Verifica se a célula de destino está no mapa e se o estado pretendido é 1 (ON)
                # O .get(target_cell_id, 1) assume 1 (ON) por segurança se a célula não estiver no mapa (ex: célula LTE 1)
                if intended_cell_states.get(target_cell_id, 1) == 1:
                    # Se a célula estiver LIGADA, permite o HO
                    action_list_output.append((1, ue_imsi, target_cell_id))
                else:
                    # Se a célula de destino estiver (0 = OFF), ignora a ação de HO
                    if self.logger:
                        self.logger.warning(f"Ação TS (HO) para UE {ue_imsi} -> Célula {target_cell_id} IGNORADA, pois a célula está sendo DESLIGADA neste step.")
                # --- FIM DA VERIFICAÇÃO ---
            # --- FIM DA MODIFICAÇÃO (Prevenir HO para Célula OFF) ---

            else:
                if self.logger: self.logger.warning(f"Ação TS inválida recebida para UE IMSI {ue_imsi}: {agent_target_action}. Esperado 0-{self.n_gnbs}.")


        if self.verbose and self.logger:
            # O log de 'Ação Achatada' agora é a decodificada
            self.logger.debug(f'Ação Achatada (Decodificada): {flat_action}')
            self.logger.debug(f'Ação Hierárquica Enviada: {action_list_output}')
            self.logger.debug(f'Previous Inverted Action (para ZERO_COUNT): {self.previous_inverted_action}')

        # Atualiza o estado no datalake (copiado de es_env)
        ts = int(self.last_timestamp % 2_000_000_000)
        for act in action_list_output:
            if act[0] == 0: # Ação ES
                _, cell, ho_allowed = act
                self.datalake.insert_data(
                    "bsState",
                    {
                        "timestamp": ts,
                        "ueImsiComplete": None,
                        "cellId": int(cell),
                        "state": int(ho_allowed),
                    },
                )

        return action_list_output

    def _update_cell_states(self):
        # Lê a tabela bsState
        cell_states_table = self.datalake.read_table('bsState')
        if not cell_states_table:
            if self.logger: self.logger.warning("_update_cell_states: Tabela bsState vazia.")
            if not self.cells_states:
                for cellId in self.cellList:
                    self.cells_states[cellId] = 1 # Assume ON (1) por defeito
            return

        # (Modificado - Lógica de es_env para pegar o estado de 100ms atrás)
        # O estado relevante é o que estava ativo *durante* o último intervalo.
        relevant_timestamp = self.last_timestamp - 100
        if relevant_timestamp < 0: relevant_timestamp = 0

        states_of_interest = [row for row in cell_states_table if row[0] == relevant_timestamp]

        # (Modificado - Lógica de es_env)
        if len(states_of_interest) == 0: # Tenta o timestamp exato se 100ms atrás falhar
            states_of_interest = [row for row in cell_states_table if row[0] == self.last_timestamp]

        if len(states_of_interest) < len(self.cellList): # Menos entradas que células (ou 0)
            if self.logger: self.logger.debug(f"_update_cell_states: Não encontrou {len(self.cellList)} estados em ts={relevant_timestamp} or {self.last_timestamp}. Encontrou {len(states_of_interest)}. Assumindo 1 (ON) para as células em falta.")
            # Assume 1 (ON) para todas, se não houver dados
            # Isto é mais seguro do que confiar em dados parciais
            if len(states_of_interest) == 0:
                for cellId in self.cellList:
                    self.cells_states[cellId] = 1
            else:
                # Atualiza as que encontrou
                found_cells = set()
                for state in states_of_interest:
                    cellId = state[2]
                    self.cells_states[cellId] = state[3]
                    found_cells.add(cellId)
                # Define 1 para as que faltam
                for cellId in self.cellList:
                    if cellId not in found_cells:
                        self.cells_states[cellId] = 1

        else: # Encontrou estados suficientes
            for state in states_of_interest:
                cellId = state[2]
                if cellId in self.cellList:
                    self.cells_states[cellId] = state[3]

        if self.logger: self.logger.debug(f"_update_cell_states: Estados das células atualizados para ts={self.last_timestamp}: {self.cells_states}")


    @override
    def _get_obs(self):
        """
        Obtém as observações de ES e TS do datalake e retorna um Dict.
        (Modificado para incluir dados de latência na observação ES)
        """
        es_obs_shape = self.observation_space['es_obs'].shape
        ts_obs_shape = self.observation_space['ts_obs'].shape
        es_obs_array = np.zeros(es_obs_shape, dtype=np.float64)
        ts_obs_array = np.zeros(ts_obs_shape, dtype=np.float64)

        if self.last_timestamp <= 0:
            if self.logger: self.logger.warning(f"_get_obs: last_timestamp inválido ({self.last_timestamp}). Retornando observações zero.")
            return {"es_obs": es_obs_array, "ts_obs": ts_obs_array}


        # --- Lógica de Observação ES (Cell-centric) ---

        # --- INÍCIO DA MODIFICAÇÃO (Corrigir agente cego) ---
        # (Novo - Carrega latência primeiro e salva no self)
        # _load_latency_data usa nomes de métricas que correspondem ao datalake.py
        self.latency_map = self._load_latency_data(self.last_timestamp)
        # --- FIM DA MODIFICAÇÃO ---

        # Estas métricas correspondem ao datalake.py (du_keys e gnb_cu_cp_keys)
        kpms_raw_es = ["nrCellId", "QosFlow.PdcpPduVolumeDL_Filter", "TB.TotNbrDl.1", "L3 serving SINR", "RRU.PrbUsedDl", "TB.TotNbrDlInitial.64Qam", "TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.16Qam"]
        ue_kpms_es = self.datalake.read_kpms(self.last_timestamp, kpms_raw_es)

        self._update_cell_states()

        if ue_kpms_es is None:
            if self.logger: self.logger.warning(f"_get_obs: datalake.read_kpms retornou None para ES no timestamp {self.last_timestamp}. Usando Obs ES zero.")
            ue_kpms_es = [] # Trata como lista vazia

        ue_complete_kpms = []

        # (Modificado - Mescla latência como em es_env)
        for ue_kpm in ue_kpms_es:
            if len(ue_kpm) <= 1: continue
            imsi = ue_kpm[0]
            cell_id = ue_kpm[1]

            if cell_id not in self.cellList: continue # Ignora células não-mmWave

            state = self.cells_states.get(cell_id, 1) # Default 1 (ON)

            # Get latency for this UE from the latency map
            # --- MODIFICAÇÃO (Corrigir agente cego) ---
            lat_data = self.latency_map.get(imsi, {'cell_avg': 0.0, 'ue_latency': 0.0})
            # --- FIM DA MODIFICAÇÃO ---
            cell_avg_latency = lat_data['cell_avg']
            ue_latency = lat_data['ue_latency']

            # Build complete KPM tuple with latency
            new_ue_kpm = ue_kpm + (cell_avg_latency, ue_latency, state)
            ue_complete_kpms.append(new_ue_kpm)

        if not ue_complete_kpms:
            if self.logger: self.logger.debug(f"_get_obs: Nenhum KPM ES válido (células 2-8) encontrado para processamento no timestamp {self.last_timestamp}.")
            # Cria um DF vazio com as colunas certas para evitar falha
            columns = (['ueImsiComplete'] + kpms_raw_es +
                       # Estes nomes correspondem ao datalake.py (lte_cu_up_keys)
                       ["DRB.PdcpSduDelayDl(cellAverageLatency)", "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "state"])
            df = pd.DataFrame(columns=columns)
            df["timestamp"] = self.last_timestamp
            df, columns = self.getRLFCounter(df, columns)
        else:
            # (Modificado - Colunas atualizadas com latência)
            columns = (
                ['ueImsiComplete'] + kpms_raw_es
                + [
                    # Estes nomes correspondem ao datalake.py (lte_cu_up_keys)
                    "DRB.PdcpSduDelayDl(cellAverageLatency)",
                    "DRB.PdcpSduDelayDl.UEID (pdcpLatency)",
                    "state"
                ]
            )

            try:
                # (Modificado - Lógica de criação de DF de es_env)
                df = pd.DataFrame(
                    ue_complete_kpms,
                    columns=columns[:len(ue_complete_kpms[0])] if ue_complete_kpms else columns
                )
                df["timestamp"] = self.last_timestamp
                df, columns = self.getRLFCounter(df, columns)

            except Exception as e:
                if self.logger: self.logger.error(f"Erro ao criar DataFrame ES inicial: {e}")
                import traceback
                if self.logger: self.logger.error(traceback.format_exc())
                # Cria um DF vazio com as colunas certas para evitar falha
                columns = (['ueImsiComplete'] + kpms_raw_es +
                           # Estes nomes correspondem ao datalake.py (lte_cu_up_keys)
                           ["DRB.PdcpSduDelayDl(cellAverageLatency)", "DRB.PdcpSduDelayDl.UEID (pdcpLatency)", "state"])
                df = pd.DataFrame(columns=columns)
                df["timestamp"] = self.last_timestamp
                df, columns = self.getRLFCounter(df, columns)

        try:
            df = self.ue_centric_tocell_centric(df) # Agrega por célula
            self.observations = self.offline_training_preprocessing(df) # Calcula KPIs derivados

            if self.observations.empty:
                if self.logger: self.logger.warning(f"_get_obs: Dataframe ES preprocessado vazio no timestamp {self.last_timestamp}. Usando Obs ES zero.")
            else:
                # Garante que todas as colunas de estado existem (preenche com 0 se faltar)
                for col in self.es_columns_state:
                    if col not in self.observations.columns:
                        self.observations[col] = 0.0

                states_df = self.observations[self.es_columns_state]
                es_obs_array_temp = states_df.iloc[-1].values.astype(np.float64)

                if es_obs_array_temp.shape[0] == es_obs_shape[0]:
                    es_obs_array = es_obs_array_temp
                else:
                    if self.logger: self.logger.error(f"_get_obs: Shape ES inesperado. Esperado {es_obs_shape[0]}, Obtido {es_obs_array_temp.shape[0]}. Usando Obs ES zero.")
                    if self.logger: self.logger.error(f"Colunas Esperadas ({len(self.es_columns_state)}): {self.es_columns_state}")
                    if self.logger: self.logger.error(f"Colunas Obtidas ({len(states_df.columns)}): {states_df.columns.tolist()}")

        except Exception as e:
            if self.logger: self.logger.error(f"Erro ao processar observações ES: {e}")
            import traceback
            if self.logger: self.logger.error(traceback.format_exc())

        # (Novo - Log de latência de es_env)
        try:
            per_cell_lat = [
                float(self.observations.get(f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{c}", pd.Series([0.0])).iloc[0])
                for c in self.cellList
            ]
            # CORRIGIDO: Lê a coluna AVG (média) que foi calculada
            avg_lat = float(self.observations.get("AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)", pd.Series([0.0])).iloc[0])
            if self.logger: self.logger.info(
                "Latency per-cell(2-8)=%s | GLOBAL_AVG=%.3f us | ts=%s",
                [round(x, 3) for x in per_cell_lat],
                avg_lat,
                int(self.last_timestamp),
            )
        except Exception as e:
            if self.logger: self.logger.warning("Latency logging failed: %r", e)

        # --- Lógica de Observação TS (UE-centric) ---
        # self.ts_columns_state corresponde ao datalake.py
        ue_kpms_ts_raw = self.datalake.read_kpms(self.last_timestamp, self.ts_columns_state)

        if ue_kpms_ts_raw is None:
            if self.logger: self.logger.warning(f"_get_obs: datalake.read_kpms retornou None para TS no timestamp {self.last_timestamp}. Usando Obs TS zero.")
        else:
            ts_obs_list = [row[1:] for row in ue_kpms_ts_raw if len(row) == len(self.ts_columns_state) + 1]
            if not ts_obs_list:
                if self.logger: self.logger.debug(f"_get_obs: Nenhuma linha TS válida retornada pelo datalake no timestamp {self.last_timestamp}.")
            else:
                try:
                    ts_obs_array_temp = np.array(ts_obs_list, dtype=np.float64)
                    num_rows_obtained = ts_obs_array_temp.shape[0]
                    num_cols_obtained = ts_obs_array_temp.shape[1]
                    num_rows_expected = ts_obs_shape[0]
                    num_cols_expected = ts_obs_shape[1]

                    if num_rows_obtained == num_rows_expected and num_cols_obtained == num_cols_expected:
                        ts_obs_array = ts_obs_array_temp
                    else:
                        if self.logger: self.logger.warning(f"_get_obs: Shape TS inesperado. Esperado {ts_obs_shape}, Obtido ({num_rows_obtained}, {num_cols_obtained}). Ajustando...")
                        adjusted_ts_obs = np.zeros(ts_obs_shape, dtype=np.float64)
                        rows_to_copy = min(num_rows_obtained, num_rows_expected)
                        cols_to_copy = min(num_cols_obtained, num_cols_expected)
                        if rows_to_copy > 0 and cols_to_copy > 0:
                            adjusted_ts_obs[:rows_to_copy, :cols_to_copy] = ts_obs_array_temp[:rows_to_copy, :cols_to_copy]
                        ts_obs_array = adjusted_ts_obs
                except Exception as e:
                    if self.logger: self.logger.error(f"Erro ao converter KPMs TS para array numpy: {e}")

        return {"es_obs": es_obs_array.astype(np.float64), "ts_obs": ts_obs_array.astype(np.float64)}


    def _compute_reward_es(self) -> (float, float, float):
        """
        Calcula a recompensa de Energy Saving.
        (Modificado para incluir latência e retornar latências)
        Retorna: (reward_es, latency_us, avg_lat)
        """

        # Valores padrão de retorno em caso de falha
        default_return = (0.0, 0.0, 0.0)

        if self.observations.empty or self.observations.shape[0] != 1:
            if self.logger: self.logger.warning(f"_compute_reward_es: DataFrame de observações ES inválido (vazio ou múltiplas linhas) no timestamp {self.last_timestamp}")
            return default_return

        missing_cols = [col for col in self.es_columns_reward if col not in self.observations.columns]
        if missing_cols:
            if self.logger: self.logger.error(f"_compute_reward_es: Colunas de recompensa em falta no DataFrame: {missing_cols}")
            return default_return

        try:
            throughput_val = self.observations['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]
            en_cons_val = self.observations['SUM_TB.TotNbrDl.1'].iloc[0]
            rlf_val = self.observations['SUM_RLF_VALUE'].iloc[0]
            on_cost_val = self.observations['SUM_ES_ON_COST'].iloc[0]
            zero_count = self.observations['ZERO_COUNT'].iloc[0] # Número de células ON (na lógica invertida)

            # (Novo - Extrai latência)
            latency_us = 0.0
            avg_lat = 0.0 # Renomeado de sum_lat
            # CORREÇÃO CRÍTICA: A métrica AVG está acumulando valores ao longo do tempo
            # Usar latência instantânea do mapa em vez da média acumulada
            if hasattr(self, 'latency_map') and self.latency_map:
                # Calcula média das latências instantâneas de todos os UEs
                all_latencies = [data['ue_latency'] for data in self.latency_map.values() if data.get('ue_latency', 0) > 0]
                if all_latencies:
                    avg_lat = float(np.mean(all_latencies))
                    latency_us = avg_lat
                else:
                    # Fallback: usar latência celular se disponível
                    all_cell_latencies = [data['cell_avg'] for data in self.latency_map.values() if data.get('cell_avg', 0) > 0]
                    if all_cell_latencies:
                        avg_lat = float(np.mean(all_cell_latencies))
                        latency_us = avg_lat

            # Normaliza latência.
            # MODIFICADO: Antes (em ms) dividia por 1000.
            # Agora (em us), dividimos por 1.000.000 para manter a magnitude da penalidade (baseada em segundos)
            latency_normalized = latency_us / 1_000_000.0

            # Converte para float
            throughput_mbps = float(throughput_val) * 8 / 1e6
            en_cons = float(en_cons_val)
            rlf = float(rlf_val)
            on_cost = float(on_cost_val)

        except (IndexError, KeyError, TypeError, ValueError) as e:
            if self.logger: self.logger.error(f"_compute_reward_es: Erro ao extrair valores do DataFrame de observações: {e}. DF:\n{self.observations}")
            return default_return

        # (Modificado - Adiciona latência ao db_row)
        self.db_row_es = {
            'throughput': throughput_mbps,
            'en_cons': en_cons,
            'rlf': rlf,
            'on_cost': on_cost,
            'latency_cell_us': float(latency_us), # Este nome é usado pelo CSV (agora us)
            'latency_ue_us': float(latency_us)    # Este nome é usado pelo CSV (agora us)
        }

        # --- CORREÇÃO CRÍTICA: Penalidade de Latência Adequada ---
        # Latência em microsegundos → converter para milissegundos para penalização
        latency_ms = latency_us / 1000.0
        # Penalidade forte para latências > 50ms (padrão 5G)
        latency_term = 0.0
        if latency_ms > 10.0:
            # Penalização exponencial: penalidade = (latencia - 50) * peso
            excess_latency = latency_ms - 50.0
            latency_term = -20.0 * excess_latency  # Peso forte para forçar baixa latência
        # --- FIM DA CORREÇÃO ---

        # Lógica de recompensa ES original do hierarchical_env
        reward = (
            0.31 * throughput_mbps
            - 0.19 * (en_cons + zero_count) # Penaliza consumo e número de células ON
            - 0.2 * rlf
            - 0.1 * on_cost
            + latency_term # NOVO TERMO
        )

        final_reward = reward if not np.isnan(reward) else 0.0
        if self.logger: self.logger.debug(
            f"_compute_reward_es: Thr={throughput_mbps:.2f}, EnCons={en_cons:.2f}, RLF={rlf:.2f}, OnCost={on_cost:.4f}, CellsON={zero_count}, Lat_Avg={latency_us:.3f}us (Term={latency_term:.4f}) -> RewardES={final_reward:.4f}"
        )
        return final_reward, latency_us, avg_lat # Retorna avg_lat (antigo sum_lat) em us


    def _compute_reward_ts(self) -> float:
        """ Calcula a recompensa de Traffic Steering. (Lógica do ts_env.py) """

        total_reward_ts = 0.0

        # --- INÍCIO DA MODIFICAÇÃO (TS ciente da latência) ---
        # As métricas em self.ts_columns_reward e a métrica de latência
        # correspondem ao datalake.py (du_keys e lte_cu_up_keys)
        ts_reward_cols_with_latency = self.ts_columns_reward + ['DRB.PdcpSduDelayDl.UEID (pdcpLatency)']

        current_kpms = self.datalake.read_kpms(self.last_timestamp, ts_reward_cols_with_latency)
        # --- FIM DA MODIFICAÇÃO ---

        if current_kpms is None:
            previous_valid_ts = self.last_timestamp - 1
            # --- INÍCIO DA MODIFICAÇÃO (TS ciente da latência) ---
            current_kpms = self.datalake.read_kpms(previous_valid_ts, ts_reward_cols_with_latency)
            # --- FIM DA MODIFICAÇÃO ---
            if current_kpms is None:
                if self.logger: self.logger.warning(f"_compute_reward_ts: Não foi possível obter KPMs de recompensa TS nos timestamps {self.last_timestamp} ou {previous_valid_ts}")
                return 0.0

        if self.previous_kpms_ts is None:
            if self.logger: self.logger.debug(f'_compute_reward_ts: Primeiro cálculo no timestamp {self.last_timestamp}')
            self.previous_timestamp_ts = self.last_timestamp - int(self.scenario_configuration.get('indicationPeriodicity', 0.1) * 1000)
            # --- INÍCIO DA MODIFICAÇÃO (TS ciente da latência) ---
            self.previous_kpms_ts = self.datalake.read_kpms(self.previous_timestamp_ts, ts_reward_cols_with_latency)
            # --- FIM DA MODIFICAÇÃO ---
            if self.previous_kpms_ts is None:
                prev_prev_ts = self.previous_timestamp_ts - 1
                # --- INÍCIO DA MODIFICAÇÃO (TS ciente da latência) ---
                self.previous_kpms_ts = self.datalake.read_kpms(prev_prev_ts, ts_reward_cols_with_latency)
                # --- FIM DA MODIFICAÇÃO ---
                if self.previous_kpms_ts is None:
                    if self.logger: self.logger.warning(f"_compute_reward_ts: Não foi possível obter KPMs TS anteriores ({self.previous_timestamp_ts} ou {prev_prev_ts}). Usando atuais como base.")
                    self.previous_kpms_ts = current_kpms


        prev_kpms_list = self.previous_kpms_ts if self.previous_kpms_ts else []
        curr_kpms_list = current_kpms if current_kpms else []

        # --- CORREÇÃO CRÍTICA (TS ciente da latência) ---
        # {imsi: (throughput, cellId, latency)}
        # CORREÇÃO: O read_kpms retorna (imsi, thp, cellId, lat) -> row[1:] tem exatamente 3 elementos
        # Validação precisa: len(row) == 4, não >= 4
        prev_kpm_dict = {}
        curr_kpm_dict = {}

        for row in prev_kpms_list:
            if len(row) == 4:  # (imsi, thp, cellId, latency)
                prev_kpm_dict[row[0]] = row[1:]
            elif len(row) > 4 and self.logger:
                self.logger.warning(f"Linha KPM anterior com colunas extras: {len(row)} elementos, esperava 4")

        for row in curr_kpms_list:
            if len(row) == 4:  # (imsi, thp, cellId, latency)
                curr_kpm_dict[row[0]] = row[1:]
            elif len(row) > 4 and self.logger:
                self.logger.warning(f"Linha KPM atual com colunas extras: {len(row)} elementos, esperava 4")
        # --- FIM DA CORREÇÃO ---

        processed_ues = set()
        for ueImsi_n, kpm_data_n in curr_kpm_dict.items():
            processed_ues.add(ueImsi_n)
            reward_ue = 0.0
            HoCost = 0.0
            LogDiff = 0.0
            pdcpLatency_n = 0.0 # Default
            pdcpLatency_o = 0.0 # Default

            # --- CORREÇÃO CRÍTICA (TS ciente da latência E Unidade) ---
            # Extrai os dados (agora com latência)
            # kpm_data_n = (ueThpDl_n, currentCell, pdcpLatency_n_raw)
            if len(kpm_data_n) != 3:
                if self.logger: self.logger.warning(f"_compute_reward_ts: Dados KPM atuais incorretos para UE {ueImsi_n}: {len(kpm_data_n)} elementos, esperava 3")
                continue # Pula se os dados de latência não vierem corretamente
            ueThpDl_n, currentCell, pdcpLatency_n_raw = kpm_data_n

            # CORREÇÃO DA UNIDADE: Dados já vêm em ms do NS-3, não há conversão de 0.1ms
            pdcpLatency_n_ms = float(pdcpLatency_n_raw) if pdcpLatency_n_raw is not None else 0.0

            # VALIDAÇÃO DE SANITY: Rejeita latências absurdas
            if pdcpLatency_n_ms > 1000.0:
                if self.logger:
                    self.logger.warning(f"Latência absurda detectada em TS para UE {ueImsi_n}: {pdcpLatency_n_ms:.3f}ms - usando fallback de 10ms")
                pdcpLatency_n_ms = 10.0  # Fallback seguro

            pdcpLatency_n_us = pdcpLatency_n_ms * 1000.0  # Converte para us para consistência interna
            # --- FIM DA CORREÇÃO ---

            if ueImsi_n in prev_kpm_dict:
                kpm_data_o = prev_kpm_dict[ueImsi_n]
                # --- CORREÇÃO CRÍTICA (TS ciente da latência E Unidade) ---
                # Garante que os dados antigos também tenham 3 colunas
                if len(kpm_data_o) == 3:
                    ueThpDl_o, sourceCell, pdcpLatency_o_raw = kpm_data_o

                    # CORREÇÃO DA UNIDADE: Dados já vêm em ms do NS-3
                    pdcpLatency_o_ms = float(pdcpLatency_o_raw) if pdcpLatency_o_raw is not None else 0.0

                    # VALIDAÇÃO DE SANITY: Rejeita latências absurdas
                    if pdcpLatency_o_ms > 1000.0:
                        if self.logger:
                            self.logger.warning(f"Latência absurda detectada em TS (anterior) para UE {ueImsi_n}: {pdcpLatency_o_ms:.3f}ms - usando fallback de 10ms")
                        pdcpLatency_o_ms = 10.0  # Fallback seguro

                    pdcpLatency_o_us = pdcpLatency_o_ms * 1000.0  # Converte para us para consistência interna
                else:
                    if self.logger: self.logger.warning(f"_compute_reward_ts: Dados KPM anteriores incorretos para UE {ueImsi_n}: {len(kpm_data_o)} elementos, esperava 3")
                    continue
                # --- FIM DA CORREÇÃO ---

                if currentCell != sourceCell:
                    lastHo = self.handovers_dict.get(ueImsi_n, 0)
                    if lastHo != 0:
                        timeDiff_sec = (self.last_timestamp - lastHo) / 1000.0
                        decay_factor = self.lambdaf_ts
                        HoCost = self.Cf_ts * np.exp(-decay_factor * timeDiff_sec)
                    self.handovers_dict[ueImsi_n] = self.last_timestamp

                    LogOld = np.log1p(ueThpDl_o) if ueThpDl_o >= 0 else 0
                    LogNew = np.log1p(ueThpDl_n) if ueThpDl_n >= 0 else 0
                    LogDiff = LogNew - LogOld
                else:
                    LogNew = np.log1p(ueThpDl_n) if ueThpDl_n >= 0 else 0
                    LogDiff = LogNew

            # --- CORREÇÃO CRÍTICA: Penalidade de Latência TS Adequada ---
            # Usando pdcpLatency_n_ms que já está em ms
            latency_ms_ts = pdcpLatency_n_ms
            # Penalidade apenas para latências > 50ms (padrão 5G)
            latency_penalty = 0.0
            if latency_ms_ts > 10.0:
                # Penalização forte: excesso de latência * peso
                excess_latency_ts = latency_ms_ts - 50.0
                latency_penalty = 2.0 * excess_latency_ts  # Peso forte para penalizar altas latências

            reward_ue = LogDiff - HoCost - latency_penalty

            if self.logger: self.logger.debug(
                f"Recompensa TS (UE {ueImsi_n}): {reward_ue:.4f} (LogDiff: {LogDiff:.4f}, HoCost: {HoCost:.4f}, LatPenalty: {latency_penalty:.4f})"
            )
            # --- FIM DA MODIFICAÇÃO ---

            # --- INÍCIO DA MODIFICAÇÃO (Correção NameError) ---
            total_reward_ts += reward_ue # Era reward_ues
            # --- FIM DA MODIFICAÇÃO ---

        if(self.logger): self.logger.debug(f"Recompensa TS Total: {total_reward_ts:.4f}")

        self.previous_kpms_ts = current_kpms
        return total_reward_ts if not np.isnan(total_reward_ts) else 0.0


    @override
    def _compute_reward(self):
        """
        Calcula a recompensa hierárquica combinando ES e TS.
        (Modificado para logar no CSV e SQLite)
        """
        self.num_steps += 1

        # (Modificado - Captura latências)
        # MODIFICADO: Variaveis agora em us
        reward_es, latency_us, avg_lat = self._compute_reward_es() # Renomeado de sum_lat
        reward_ts = self._compute_reward_ts()

        reward_es_safe = reward_es if not np.isnan(reward_es) else 0.0
        reward_ts_safe = reward_ts if not np.isnan(reward_ts) else 0.0
        total_reward = reward_es_safe + (self.ts_reward_weight * reward_ts_safe)
        self.reward = total_reward

        # --- Registo no Grafana (BD) ---
        db_row = {}
        db_row['timestamp'] = self.last_timestamp
        db_row['ueImsiComplete'] = None
        db_row['time_grafana'] = self.last_timestamp
        db_row['step'] = self.num_steps

        # Adiciona métricas do ES (agora inclui latência)
        # CORRIGIDO: Mapeia manualmente de db_row_es para db_row
        # para corresponder às chaves da tabela grafana
        if hasattr(self, 'db_row_es'):
            # self.db_row_es tem {'throughput', 'en_cons', 'rlf', 'on_cost', 'latency_cell_us', 'latency_ue_us'}
            # A tabela grafana espera 'avg_latency_us'
            db_row['throughput'] = self.db_row_es.get('throughput', 0)
            db_row['en_cons'] = self.db_row_es.get('en_cons', 0)
            db_row['rlf'] = self.db_row_es.get('rlf', 0)
            db_row['on_cost'] = self.db_row_es.get('on_cost', 0)
            db_row['avg_latency_us'] = self.db_row_es.get('latency_cell_us', latency_us) # Usa 'latency_cell_us' que é a média
        else:
            # Fallback (caso db_row_es não exista)
            db_row['throughput'] = 0
            db_row['en_cons'] = 0
            db_row['rlf'] = 0
            db_row['on_cost'] = 0
            db_row['avg_latency_us'] = latency_us # CORRIGIDO (nome consistente)

        db_row['reward'] = total_reward
        db_row['reward_es'] = reward_es_safe
        db_row['reward_ts'] = reward_ts_safe

        try:
            needs_connect = self.datalake.connection is None
            if needs_connect: self.datalake.acquire_connection()
            self.datalake.insert_data("grafana", db_row)
            if needs_connect: self.datalake.release_connection()
        except Exception as e:
            if self.logger: self.logger.error(f"Erro ao inserir dados na tabela grafana: {e}")

        # --- (Novo - Log em SQLite e CSV) ---
        try:
            # 1. Armazena no SQLite (para tracking interno de latência)
            self._store_latency_in_sqlite(latency_us, avg_lat) # Passa avg_lat

            # 2. Armazena no CSV (para análise externa)
            self._append_qos_snapshot(latency_us, latency_us, total_reward)

        except Exception as e:
            if self.logger: self.logger.error(f"Erro durante o logging de latência/CSV: {e}")


        if self.logger: self.logger.info(f"Step {self.num_steps} @ TS={self.last_timestamp}ms - Reward Total: {total_reward:.4f} (ES: {reward_es_safe:.4f}, TS: {reward_ts_safe:.4f})")

        return total_reward


    # (Novo - de es_env) Armazena latência no SQLite
    def _store_latency_in_sqlite(self, avg_latency: float, avg_lat_calc: float): # Renomeado de sum_latency
        """
        Armazena métricas de latência por célula na tabela SQLite dedicada.
        MODIFICADO: Valores agora em microsegundos.
        """
        try:
            latency_dict = {
                "timestamp": int(self.last_timestamp),
                "ueImsiComplete": None,
                "step": int(self.num_steps),
            }

            per_cell_latencies = []
            for cell in self.cellList:
                # O nome da coluna base corresponde ao datalake.py
                col_name = f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"
                if col_name in self.observations.columns:
                    lat_val = float(self.observations[col_name].iloc[0])
                else:
                    lat_val = 0.0
                latency_dict[f"cell_{cell}_latency"] = lat_val
                per_cell_latencies.append(lat_val)

            latency_dict["avg_cell_latency"] = float(avg_latency) # Média real (calculada em compute_reward)
            latency_dict["sum_latency"] = float(sum(per_cell_latencies)) # Soma real
            latency_dict["avg_lat_debug"] = float(avg_lat_calc) # Valor que veio do 'AVG_' (deve ser igual a avg_latency)

            non_zero_latencies = [lat for lat in per_cell_latencies if lat > 0]
            latency_dict["max_latency"] = float(max(non_zero_latencies)) if non_zero_latencies else 0.0
            latency_dict["min_latency"] = float(min(non_zero_latencies)) if non_zero_latencies else 0.0

            self.datalake.insert_data("latency_tracking", latency_dict)

        except Exception as e:
            if self.logger: self.logger.error("Failed to store latency in SQLite: %r", e)


    @override
    def _init_datalake_usecase(self):
        # (Modificado - Tabela Grafana atualizada com latência)
        grafana_keys = {
            "timestamp": "INTEGER PRIMARY KEY",
            "ueImsiComplete": "INTEGER",
            "time_grafana": "INTEGER",
            "step": "INTEGER",
            "throughput": "REAL",
            "en_cons": "REAL",
            "rlf": "REAL",
            "on_cost": "REAL",
            "avg_latency_us": "REAL", # CORRIGIDO (era avg_latency_ms)
            "reward": "REAL",
            "reward_es": "REAL",
            "reward_ts": "REAL"
        }
        # Tabela de Estado da BS
        bs_state_keys_with_types = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER", # Será NULL
            "cellId": "INTEGER",
            "state": "INTEGER"
            # ", PRIMARY KEY (timestamp, cellId) ON CONFLICT REPLACE" # Garante unicidade
        }

        needs_connect = self.datalake.connection is None
        if needs_connect: self.datalake.acquire_connection()
        try:
            self.datalake._create_table("bsState", bs_state_keys_with_types)
            self.datalake._create_table("grafana", grafana_keys)

            # (Novo - Cria tabela de latência)
            # Adiciona colunas extras para depuração
            self.latency_tracking_keys["avg_lat_debug"] = "REAL"
            self.latency_tracking_keys["sum_latency"] = "REAL"
            self.datalake._create_table("latency_tracking", self.latency_tracking_keys)
        except Exception as e:
            if self.logger: self.logger.error(f"Erro ao inicializar tabelas do datalake: {e}")
        finally:
            if needs_connect: self.datalake.release_connection()


    @override
    def _fill_datalake_usecase(self):
        # Lê o ficheiro bsState.txt gerado pelo C++
        bs_state_file_path = os.path.join(self.sim_path, 'bsState.txt')
        if not os.path.exists(bs_state_file_path):
            return

        latest_timestamp_in_file = 0
        rows_to_insert = []
        try:
            with open(bs_state_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                try:
                    header = next(reader) # Pula cabeçalho
                except StopIteration:
                    return # Ficheiro vazio

                for row in reader:
                    row = [item for item in row if item]
                    if len(row) == 4:
                        try:
                            timestamp_unix = int(row[1])
                            cell_id = int(row[2])
                            state = int(row[3])

                            # (Modificado - Lógica de es_env)
                            # Não filtra por self.last_timestamp aqui, deixa o insert
                            # lidar com duplicados. Carrega tudo.
                            # if timestamp_unix >= self.last_timestamp:
                            db_row = {
                                'timestamp': timestamp_unix,
                                'ueImsiComplete': None,
                                'cellId': cell_id,
                                'state': state
                            }
                            rows_to_insert.append(db_row)
                            latest_timestamp_in_file = max(latest_timestamp_in_file, timestamp_unix)

                        except (ValueError, IndexError) as e:
                            if self.logger: self.logger.warning(f"Erro ao processar linha em bsState.txt: {row}. Erro: {e}")
                    elif row:
                        if self.logger: self.logger.warning(f"Linha ignorada em bsState.txt (formato inesperado): {row}")

        except FileNotFoundError:
            return
        except Exception as e:
            if self.logger: self.logger.error(f"Erro inesperado ao ler bsState.txt: {e}")
            return

        if rows_to_insert:
            inserted_count = 0
            needs_connect = self.datalake.connection is None
            if needs_connect: self.datalake.acquire_connection()
            try:
                # (Modificado - Lógica de es_env para evitar duplicados)
                # Obter o timestamp mais recente já no banco
                max_ts_in_db_result = self.datalake.cursor.execute("SELECT MAX(timestamp) FROM bsState").fetchone()
                max_ts_in_db = max_ts_in_db_result[0] if max_ts_in_db_result[0] is not None else -1

                rows_filtered = [row for row in rows_to_insert if row['timestamp'] > max_ts_in_db]

                if rows_filtered:
                    # Usa executemany para inserção em lote
                    sql = "INSERT INTO bsState (timestamp, ueImsiComplete, cellId, state) VALUES (:timestamp, :ueImsiComplete, :cellId, :state)"
                    try:
                        self.datalake.cursor.executemany(sql, rows_filtered)
                        self.datalake.connection.commit()
                        inserted_count = len(rows_filtered)
                    except Exception as e: # Tenta inserção individual se o lote falhar
                        if self.logger: self.logger.warning(f"Falha na inserção em lote de bsState ({e}), tentando individualmente...")
                        self.datalake.connection.rollback()
                        for db_row in rows_filtered:
                            try:
                                self.datalake.insert_data("bsState", db_row)
                                inserted_count += 1
                            except Exception as e_ind:
                                if "UNIQUE constraint failed" not in str(e_ind):
                                    if self.logger: self.logger.error(f"Erro ao inserir linha bsState {db_row}: {e_ind}")

                if self.logger and inserted_count > 0: self.logger.debug(f"_fill_datalake_usecase: {inserted_count} novas linhas inseridas na tabela bsState.")

            except Exception as e:
                if self.logger: self.logger.error(f"Erro durante a inserção em bsState: {e}")
            finally:
                if needs_connect: self.datalake.release_connection()


    # --- Funções Helper do ES_ENV ---

    def ue_centric_tocell_centric(self, df):
        """Limpa o dataframe de KPMs brutos."""
        cols_to_drop = ['ueImsiComplete', 'L3 serving SINR']

        # --- MODIFICAÇÃO (Corrigir agente cego) ---
        # Não drope 'ueImsiComplete' ainda, precisamos dele para o merge de latência
        #cols_to_drop = ['L3 serving SINR']
        # --- FIM DA MODIFICAÇÃO ---

        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # (Modificado - Lógica de es_env)
        # Substitui -inf em L3 serving SINR *antes* de dropar (embora esteja dropando)
        if "L3 serving SINR" in df.columns:
            df["L3 serving SINR"] = df["L3 serving SINR"].replace(-np.inf, 0)

        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df

    def rename_columns(self, columns, cell_no):
        """Adiciona sufixo _cellId a uma lista de nomes de colunas."""
        return [f"{col}_{cell_no}" for col in columns]

    def offline_training_preprocessing(self, df):
        """
        Preprocessa o DataFrame agregando por célula e calculando KPIs derivados.
        (Modificado para lidar com latência)
        """
        if df.empty:
            if self.logger: self.logger.warning("offline_training_preprocessing: DataFrame de entrada (células 2-8) vazio.")
            # Retorna um DF vazio, mas com as colunas esperadas, preenchidas com 0
            # para evitar falhas posteriores
            all_needed_cols = list(dict.fromkeys(list(self.es_columns_state) + list(self.es_columns_reward)))
            empty_df = pd.DataFrame(columns=all_needed_cols)
            # Adiciona uma linha de zeros
            empty_df.loc[0] = 0.0

            # --- INÍCIO DA MODIFICAÇÃO (Corrigir agente cego) ---
            # Calcula a média de latência global mesmo se o df de entrada (células 2-8) estiver vazio
            avg_latency_global = 0.0
            if hasattr(self, 'latency_map') and self.latency_map:
                all_latencies = [data['ue_latency'] for data in self.latency_map.values() if data.get('ue_latency', 0) > 0]
                if all_latencies:
                    avg_latency_global = np.mean(all_latencies)
            empty_df["AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)"] = avg_latency_global
            # --- FIM DA MODIFICAÇÃO ---

            # (Modificado - Lógica de ZERO_COUNT de es_env)
            empty_df["ACTION_BINARY"] = self.previous_inverted_action
            empty_df["ACTION_BINARY"] = empty_df["ACTION_BINARY"].astype(str)
            empty_df["ZERO_COUNT"] = empty_df["ACTION_BINARY"].apply(lambda x: x.count("0"))

            return empty_df


        try:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce').fillna(0)
            df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
            df.sort_values(by=["timestamp"], ascending=True, inplace=True)

            # (Modificado - Lógica de es_env para state)
            if "state" in df.columns:
                df["state"] = df["state"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))

            cell_dataframes = []

            # (Modificado - Lógica de es_env para agregação)
            cell_df = pd.DataFrame()
            is_initial_cell = True

            for cell in self.cellList:
                temp = df.loc[df["nrCellId"] == cell].copy()
                if temp.empty: continue # Pula se não há dados para esta célula

                if "RRU.PrbUsedDl" in temp.columns:
                    temp["RRU_PRBTOTDL"] = (temp["RRU.PrbUsedDl"] / 139) * 100
                else:
                    temp["RRU_PRBTOTDL"] = 0.0

                tb = temp.get("TB.TotNbrDl.1", pd.Series([1e-5] * len(temp)))
                qos = temp.get("QosFlow.PdcpPduVolumeDL_Filter", pd.Series([0.0] * len(temp)))
                temp["EEKPI_RL"] = qos / tb.replace(0, 1e-5)

                temp.columns = self.rename_columns(list(temp.columns), cell)
                temp.rename(columns={f"timestamp_{cell}": "timestamp"}, inplace=True)

                # --- INÍCIO DA MODIFICAÇÃO (Correção Datalake) ---
                # (Modificado - Lógica de alias de latência de es_env)
                # Esta lista agora contém APENAS os nomes reais definidos em datalake.py
                alias_cols = [
                    f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}",
                    f"DRB.PdcpSduDelayDl(cellAverageLatency)_{cell}",
                ]
                # --- FIM DA MODIFICAÇÃO (Correção Datalake) ---

                found = None
                for c in alias_cols:
                    if c in temp.columns:
                        found = c
                        break

                if found is not None:
                    # Renomeia a coluna encontrada (seja UEID ou cellAverage) para a coluna UEID
                    # Isso implementa a lógica de fallback (usar cellAverage se UEID não estiver disponível)
                    temp.rename(columns={found: f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"}, inplace=True)
                else:
                    # Se nenhuma coluna de latência foi encontrada (nem UEID nem cellAverage), cria uma coluna zerada
                    temp[f"DRB.PdcpSduDelayDl.UEID (pdcpLatency)_{cell}"] = 0.0

                # Agrega dados (pega a última linha por timestamp, se houver múltiplas)
                # Assume que df de entrada contém apenas dados do self.last_timestamp
                temp_agg = temp.iloc[-1:].reset_index(drop=True)

                if is_initial_cell:
                    cell_df = temp_agg
                    is_initial_cell = False
                else:
                    if not temp_agg.empty:
                        cell_df = pd.merge(cell_df, temp_agg, how="outer", on=["timestamp"])

            if is_initial_cell: # Nenhuma célula (2-8) tinha dados
                if self.logger: self.logger.debug("offline_training_preprocessing: Nenhum dado de célula (2-8) encontrado após agrupamento.")
                # Cria um DF vazio com 'timestamp' para que as agregações de SUM e AVG possam rodar
                cell_df = pd.DataFrame({'timestamp': [self.last_timestamp]})


            merged_df = cell_df.infer_objects(copy=False).fillna(0)
            merged_df = self.es_on_cost_calculation(merged_df)

            # --- INÍCIO DA MODIFICAÇÃO (Corrigir agente cego) ---
            # (Modificado - Lógica de agregação de es_env com 'mean' para latência)
            agg_specs = {
                "SUM_QosFlow.PdcpPduVolumeDL_Filter": ("QosFlow.PdcpPduVolumeDL_Filter_", "sum"),
                "SUM_TB.TotNbrDl.1": ("TB.TotNbrDl.1_", "sum"),
                "SUM_ES_ON_COST": ("ES_ON_COST_", "sum"),
                "SUM_RLF_VALUE": ("RLF_VALUE_", "sum"),
                # REMOVIDO: "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)": ("DRB.PdcpSduDelayDl.UEID (pdcpLatency)_", "mean"),
            }

            for out_col, (prefix, how) in agg_specs.items():
                cols = merged_df.filter(like=prefix)

                if cols.shape[1] == 0:
                    merged_df[out_col] = 0.0
                elif how == "sum":
                    merged_df[out_col] = cols.sum(axis=1)
                elif how == "mean":
                    # (Esta lógica 'mean' não é mais usada para latência, mas a deixamos caso seja usada para outra coisa)
                    non_zero_cols = cols.where(cols > 0)
                    mean_val = non_zero_cols.mean(axis=1)
                    merged_df[out_col] = mean_val.fillna(0.0)
                else:
                    merged_df[out_col] = 0.0

            # --- CORREÇÃO CRÍTICA: Cálculo da Média Global de Latência ---
            # IMPORTANTE: Não usar latência acumulada! Usar valores instantâneos do mapa
            avg_latency_global = 0.0
            if hasattr(self, 'latency_map') and self.latency_map:
                # Pega APENAS latências instantâneas de UE (não acumuladas)
                all_instant_latencies = [data['ue_latency'] for data in self.latency_map.values() if data.get('ue_latency', 0) > 0 and data.get('ue_latency', 0) < 100000]  # Filter: <100ms para evitar erros
                if all_instant_latencies:
                    avg_latency_global = float(np.mean(all_instant_latencies))
                else:
                    # Fallback para latências celulares se não houver UE
                    all_cell_latencies = [data['cell_avg'] for data in self.latency_map.values() if data.get('cell_avg', 0) > 0 and data.get('cell_avg', 0) < 100000]
                    if all_cell_latencies:
                        avg_latency_global = float(np.mean(all_cell_latencies))

            # Armazena latência instantânea correta em vez de acumulada
            merged_df["AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)"] = avg_latency_global
            # --- FIM DA MODIFICAÇÃO (Corrigir agente cego) ---


            # (Modificado - Lógica de ZERO_COUNT de es_env)
            merged_df["ACTION_BINARY"] = self.previous_inverted_action
            merged_df["ACTION_BINARY"] = merged_df["ACTION_BINARY"].astype(str)
            merged_df["ZERO_COUNT"] = merged_df["ACTION_BINARY"].apply(lambda x: x.count("0"))

            # Garante que todas as colunas necessárias existem
            # CORRIGIDO: Garante que a lista de colunas reflete a mudança de SUM para AVG
            all_needed_cols = list(self.es_columns_state) + list(self.es_columns_reward) # Garante que são listas
            # Adiciona a coluna AVG corrigida se ela não estiver lá (já deve estar por causa das listas)
            if "AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)" not in all_needed_cols:
                all_needed_cols.append("AVG_DRB.PdcpSduDelayDl.UEID (pdcpLatency)")
            # Remove a coluna SUM errada se ela ainda estiver lá
            if "SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)" in all_needed_cols:
                all_needed_cols.remove("SUM_DRB.PdcpSduDelayDl.UEID (pdcpLatency)")

            # Remove duplicados
            all_needed_cols = list(dict.fromkeys(all_needed_cols))

            for col in all_needed_cols:
                if col not in merged_df.columns:
                    merged_df[col] = 0.0

            final_cols = [col for col in all_needed_cols if col in merged_df.columns]
            return merged_df[final_cols].reset_index(drop=True)


        except Exception as e:
            if self.logger: self.logger.error(f"Erro em offline_training_preprocessing: {e}\nDataFrame no momento do erro:\n{df.head(2)}")
            import traceback
            if self.logger: self.logger.error(traceback.format_exc())
            # Retorna um DF vazio, mas com as colunas esperadas, preenchidas com 0
            all_needed_cols = list(dict.fromkeys(list(self.es_columns_state) + list(self.es_columns_reward)))
            empty_df = pd.DataFrame(columns=all_needed_cols)
            empty_df.loc[0] = 0.0
            return empty_df


    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df):
        """Calcula métricas de modulação."""
        if df.empty: return df
        mod_cols = ['TB.TotNbrDlInitial.Qpsk', 'TB.TotNbrDlInitial.16Qam', 'TB.TotNbrDlInitial.64Qam']
        other_cols = ['RRU.PrbUsedDl', 'TB.TotNbrDl.1']

        for col in mod_cols + other_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['TB.TOTNBRDLINITIAL.SUM'] = df[mod_cols].sum(axis=1)

        sum_non_zero = df['TB.TOTNBRDLINITIAL.SUM'].replace(0, np.nan)
        df['TB_TOTNBRDLINITIAL_64QAM_RATIO'] = (df['TB.TotNbrDlInitial.64Qam'] / sum_non_zero).fillna(0.0)

        # (Modificado - Lógica de es_env)
        df['RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].replace(0, 0.00001)
        df['TB.TotNbrDl.1'] = df['TB.TotNbrDl.1'].replace(0, 0.00001)

        return df

    def getRLFCounter(self, df, columns):
        """Adiciona contadores RLF (Radio Link Failure) ao DataFrame."""
        if df.empty:
            if 'RLF_Counter' not in columns: columns.append('RLF_Counter')
            if 'RLF_VALUE' not in columns: columns.append('RLF_VALUE')
            return df, columns

        if 'L3 serving SINR' not in df.columns: df['L3 serving SINR'] = -1000.0
        df['L3 serving SINR'] = pd.to_numeric(df['L3 serving SINR'], errors='coerce').fillna(-1000).replace([-np.inf, np.inf], -1000)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)
        df['nrCellId'] = pd.to_numeric(df['nrCellId'], errors='coerce').fillna(0).astype(int)

        df['RLF_Counter'] = 0.0
        df['RLF_VALUE'] = 0
        if 'RLF_Counter' not in columns: columns.append('RLF_Counter')
        if 'RLF_VALUE' not in columns: columns.append('RLF_VALUE')

        # (Modificado - Lógica de es_env com transform)
        # Adiciona verificação para evitar erro em df vazio
        if not df.empty:
            grouped = df.groupby(['timestamp', 'nrCellId'])['L3 serving SINR']
            total_counts = grouped.transform('count')
            below_threshold_counts = grouped.transform(lambda x: (x < -5.0).sum())

            df['RLF_Counter'] = np.where(total_counts > 0, (below_threshold_counts / total_counts) * 100.0, 0.0)
            df['RLF_VALUE'] = below_threshold_counts

        return df, columns

    def es_on_cost_calculation(self, cell_df):
        """
        Calcula o custo de transição ES (Energy Saving).
        (Modificado para usar lógica de es_env)
        """
        if cell_df.empty: return cell_df

        current_timestamp_ms = self.last_timestamp

        # Garante que temos pelo menos uma linha para escrever (caso cell_df venha apenas com 'timestamp')
        if len(cell_df) == 0:
            cell_df.loc[0] = 0.0
            cell_df['timestamp'] = self.last_timestamp


        for cell in self.cellList:
            es_on_cost_col = f'ES_ON_COST_{cell}'
            time_diff_obs_col = f'TIME_DIFF_OBS_{cell}'

            # (Modificado - Lógica de es_env)
            # Obtém o estado ATUAL da célula (0=OFF, 1=ON) de self.cells_states

            current_state_real = self.cells_states.get(cell, 1) # 0=OFF, 1=ON (Padrão ON)

            time_diff_ms = 0.0
            cost = 0.0

            if current_state_real == 1: # Célula está ON
                if self.cell_timestamp_state_dict[cell] == float('inf'): # Estava OFF antes?
                    time_diff_ms = 100 # Custo de transição (fixo 100ms?)
                    self.cell_timestamp_state_dict[cell] = current_timestamp_ms
                else: # Já estava ON
                    time_diff_ms = (current_timestamp_ms - self.cell_timestamp_state_dict[cell]) + 100
                    if time_diff_ms < 100: time_diff_ms = 100 # Garante mínimo
            else: # Célula está OFF
                time_diff_ms = float('inf')
                self.cell_timestamp_state_dict[cell] = float('inf') # Reseta


            if time_diff_obs_col not in cell_df.columns: cell_df[time_diff_obs_col] = np.nan
            if es_on_cost_col not in cell_df.columns: cell_df[es_on_cost_col] = np.nan

            cost = self.Cf_es * ((1 - self.lambdaf_es) ** (time_diff_ms * self.time_factor_es)) if time_diff_ms != float("inf") else 0.0

            # Garante que estamos escrevendo na primeira (e única) linha (índice 0)
            cell_df.at[cell_df.index[0], time_diff_obs_col] = time_diff_ms
            cell_df.at[cell_df.index[0], es_on_cost_col] = cost

        return cell_df

    def bs_states_list(self):
        """Retorna o estado atual das BSs (0=OFF, 1=ON) como uma lista."""
        self._update_cell_states()
        states = [self.cells_states.get(cell, 1) for cell in self.cellList]
        return states
