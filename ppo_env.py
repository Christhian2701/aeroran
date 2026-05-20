#!/usr/bin/env python3
"""
ppo_env.py novo

Online PPO environment for the ES-only UAV use case.

Important:
- This is an ONLINE trainer environment. It does not have access to the
  offline Quantile Transformer / quantile-normalized reward pipeline used in
  some postprocessed datasets.
- If you train with Stable-Baselines3, wrap this environment with
  `VecNormalize` in the training script so observation and reward scaling are
  learned online during interaction with ns-3.
"""

from pathlib import Path
import sys

import numpy as np
from gymnasium import spaces


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / "ns-o-ran-gym" / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "sac1"))

from hierarchical_env import HierarchicalEnv  # noqa: E402


class UavEnergySavingPpoEnv(HierarchicalEnv):
    """
    ES-only PPO wrapper around the hierarchical UAV environment.

    This strips out Traffic Steering from the public PPO interface:
    - observation is only the ES state vector
    - action is 7 binary ON/OFF decisions for cells 2..8
    - reward is ES-only, with PPO-1 paper weights
    """

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool,
        verbose: bool = False,
        randomization_seed: int = 0,
    ):
        super().__init__(
            ns3_path=ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            do_heuristic=True,
            ts_reward_weight=0.0,
            verbose=verbose,
            scenario_name="scenario-hierarchical-xangai-UAV",
        )

        self._base_scenario_configuration = dict(self.scenario_configuration)
        self._base_rng_run = int(self.scenario_configuration.get("RngRun", 400))
        self._episode_index = 0
        self._randomization_rng = np.random.default_rng(randomization_seed)
        self._hierarchical_observation_space = self.observation_space
        self.action_space = spaces.MultiBinary(self.n_gnbs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.es_columns_state),),
            dtype=np.float32,
        )

    def _randomize_episode_configuration(self):
        """
        Apply per-episode variation before ns-3 is launched.

        Current episode randomization:
        - RngRun: increment each reset
        - uavFlightPattern: sampled from mobile patterns 1..5
        - uavMaxSpeed: sampled uniformly in [8.0, 20.0] m/s
        """
        self.scenario_configuration.update(self._base_scenario_configuration)

        self._episode_index += 1
        self.scenario_configuration["RngRun"] = self._base_rng_run + self._episode_index

        if int(self.scenario_configuration.get("uavMobilityMode", 0)) == 1:
            self.scenario_configuration["uavFlightPattern"] = int(
                self._randomization_rng.choice([1, 2, 3, 4, 5])
            )
            self.scenario_configuration["uavMaxSpeed"] = round(
                float(self._randomization_rng.uniform(8.0, 20.0)), 2
            )

        if self.logger:
            self.logger.info(
                "Episode %s randomized config: RngRun=%s, uavFlightPattern=%s, uavMaxSpeed=%s",
                self._episode_index,
                self.scenario_configuration.get("RngRun"),
                self.scenario_configuration.get("uavFlightPattern"),
                self.scenario_configuration.get("uavMaxSpeed"),
            )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._randomization_rng = np.random.default_rng(seed)
        self._randomize_episode_configuration()
        return super().reset(seed=seed, options=options)

    def _get_obs(self):
        original_space = self.observation_space
        self.observation_space = self._hierarchical_observation_space
        try:
            obs = super()._get_obs()
        finally:
            self.observation_space = original_space
        es_obs = obs.get("es_obs")
        if es_obs is None:
            es_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.asarray(es_obs, dtype=np.float32).reshape(self.observation_space.shape)

    def _compute_action(self, action):
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_array.shape[0] != self.n_gnbs:
            raise ValueError(
                f"Expected {self.n_gnbs} PPO ES bits, got shape {action_array.shape}"
            )

        action_array = np.clip(action_array, 0, 1)

        # Keep inherited ZERO_COUNT semantics aligned with "number of cells ON".
        # The parent preprocessing counts zeroes in previous_inverted_action.
        self.previous_inverted_action = "".join(
            "0" if int(bit) == 1 else "1" for bit in action_array
        )

        actions = []
        for cell_id, agent_bit in zip(self.cellList, action_array):
            ns3_state = int(agent_bit)
            actions.append((0, cell_id, ns3_state))
        return actions

    def _compute_reward_es(self):
        """
        PPO-1 ES reward.

        Uses the paper weights:
        - w1 = 0.51 for throughput
        - w2 = 0.19 for energy/resource usage
        - w3 = 0.20 for RLF
        - w4 = 0.10 for ES on-cost

        Latency is still measured and logged for QoS tracking, but the reward
        itself follows the paper's four-term PPO-1 structure.
        """

        default_return = (0.0, 0.0, 0.0)

        if self.observations.empty or self.observations.shape[0] != 1:
            if self.logger:
                self.logger.warning(
                    "_compute_reward_es: invalid ES observations at ts=%s",
                    self.last_timestamp,
                )
            return default_return

        missing_cols = [col for col in self.es_columns_reward if col not in self.observations.columns]
        if missing_cols:
            if self.logger:
                self.logger.error(
                    "_compute_reward_es: missing reward columns: %s",
                    missing_cols,
                )
            return default_return

        try:
            throughput_val = self.observations["SUM_QosFlow.PdcpPduVolumeDL_Filter"].iloc[0]
            en_cons_val = self.observations["SUM_TB.TotNbrDl.1"].iloc[0]
            rlf_val = self.observations["SUM_RLF_VALUE"].iloc[0]
            on_cost_val = self.observations["SUM_ES_ON_COST"].iloc[0]

            latency_us = 0.0
            avg_lat = 0.0
            if hasattr(self, "latency_map") and self.latency_map:
                all_latencies = [
                    data["ue_latency"]
                    for data in self.latency_map.values()
                    if data.get("ue_latency", 0) > 0
                ]
                if all_latencies:
                    avg_lat = float(np.mean(all_latencies))
                    latency_us = avg_lat
                else:
                    all_cell_latencies = [
                        data["cell_avg"]
                        for data in self.latency_map.values()
                        if data.get("cell_avg", 0) > 0
                    ]
                    if all_cell_latencies:
                        avg_lat = float(np.mean(all_cell_latencies))
                        latency_us = avg_lat

            throughput_mbps = float(throughput_val) * 8 / 1e6
            en_cons = float(en_cons_val)
            rlf = float(rlf_val)
            on_cost = float(on_cost_val)

            cells_on = float(self.previous_inverted_action.count("0"))

        except (IndexError, KeyError, TypeError, ValueError) as exc:
            if self.logger:
                self.logger.error(
                    "_compute_reward_es: failed to extract values: %r",
                    exc,
                )
            return default_return

        self.db_row_es = {
            "throughput": throughput_mbps,
            "en_cons": en_cons,
            "rlf": rlf,
            "on_cost": on_cost,
            "latency_cell_us": float(latency_us),
            "latency_ue_us": float(latency_us),
        }

        reward = (
            0.51 * throughput_mbps
            - 0.19 * (en_cons + cells_on)
            - 0.20 * rlf
            - 0.10 * on_cost
        )

        final_reward = float(reward) if not np.isnan(reward) else 0.0
        if self.logger:
            self.logger.debug(
                "_compute_reward_es[PPO1]: Thr=%.2f, EnCons=%.2f, RLF=%.2f, OnCost=%.4f, CellsON=%.0f, Lat=%.3fus -> Reward=%.4f",
                throughput_mbps,
                en_cons,
                rlf,
                on_cost,
                cells_on,
                latency_us,
                final_reward,
            )

        return final_reward, latency_us, avg_lat

    def _compute_reward(self):
        self.num_steps += 1

        reward_es, latency_us, avg_lat = self._compute_reward_es()
        reward_es_safe = float(reward_es) if not np.isnan(reward_es) else 0.0
        self.reward = reward_es_safe

        db_row = {
            "timestamp": self.last_timestamp,
            "ueImsiComplete": None,
            "time_grafana": self.last_timestamp,
            "step": self.num_steps,
            "throughput": 0.0,
            "en_cons": 0.0,
            "rlf": 0.0,
            "on_cost": 0.0,
            "avg_latency_us": latency_us,
            "reward": reward_es_safe,
            "reward_es": reward_es_safe,
            "reward_ts": 0.0,
        }

        if hasattr(self, "db_row_es"):
            db_row["throughput"] = self.db_row_es.get("throughput", 0.0)
            db_row["en_cons"] = self.db_row_es.get("en_cons", 0.0)
            db_row["rlf"] = self.db_row_es.get("rlf", 0.0)
            db_row["on_cost"] = self.db_row_es.get("on_cost", 0.0)
            db_row["avg_latency_us"] = self.db_row_es.get("latency_cell_us", latency_us)

        try:
            needs_connect = self.datalake.connection is None
            if needs_connect:
                self.datalake.acquire_connection()
            self.datalake.insert_data("grafana", db_row)
            if needs_connect:
                self.datalake.release_connection()
        except Exception as exc:
            if self.logger:
                self.logger.error("Erro ao inserir dados na tabela grafana: %r", exc)

        try:
            self._store_latency_in_sqlite(latency_us, avg_lat)
            self._append_qos_snapshot(latency_us, latency_us, reward_es_safe)
        except Exception as exc:
            if self.logger:
                self.logger.error("Erro durante o logging de latência/CSV: %r", exc)

        if self.logger:
            self.logger.info(
                "Step %s @ TS=%sms - Reward ES: %.4f (Reward TS fixed at 0.0)",
                self.num_steps,
                self.last_timestamp,
                reward_es_safe,
            )

        return reward_es_safe
