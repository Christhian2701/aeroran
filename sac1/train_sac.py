import argparse
import json
import numpy as np
import os
import sys

# --- CORREÇÃO DE PATH PARA IMPORTAÇÃO (Mantido do script PPO) ---
# Adiciona o diretório raiz do projeto ao sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumindo que o diretório raiz é dois níveis acima (e.g., ../../)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

# Importa o ambiente mesclado AGORA que o PATH foi corrigido
from environments.hierarchical_env import HierarchicalEnv 

# --- Imports para Stable Baselines 3 ---
from stable_baselines3 import SAC # <--- MUDANÇA: Importa SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement # NOVO: Para Early Stopping
# ----------------------------------------

# --- Imports para Multi-GPU ---
import torch
import os
# ----------------------------------------

# Definições de Hiperparâmetros SAC Padrão (AJUSTADO PARA IGUALAR O LR do PPO)
# SAC é off-policy e usa Replay Buffer.
SAC_HYPERPARAMS = {
    # Para ser consistente com PPO
    "learning_rate": 0.0001, 
    "gamma": 0.99,             
    # Buffer e Learning Starts são específicos do SAC
    "buffer_size": 1000000,     # Valor razoável para SAC (Original no argparse)
    "learning_starts": 1000,   # Valor razoável para SAC (Original no argparse)
    "tau": 0.001,              # Taxa de atualização da rede alvo (padrão SB3)
    "train_freq": 1,           # Frequência de treino (padrão SB3)
    "ent_coef": 0.01,
}

# Definições de Arquitetura da Rede Neural (Política)
# CORRIGIDO: Deve ser um dicionário que mapeia 'pi' e 'qf' para listas de inteiros.
POLICY_ARCH = {
    # Define 2 camadas ocultas com 256 neurônios cada
    "net_arch": {
        "pi": [256, 256],      # Arquitetura da Rede do Ator (Policy)
        "qf": [256, 256]       # Arquitetura das Redes da Q-Function (Crítico)
    }
}

# --- CONFIGURAÇÕES DE EARLY STOP (Mantido do script PPO) ---
EARLY_STOP_CONFIG = {
    "eval_freq": 10000, # Avalia a cada 10000 timesteps
    "n_eval_episodes": 5, # Número de episódios para calcular a média de recompensa na avaliação
    "patience_evals": 4, # Número de avaliações sem melhoria para parar
}


# Função para criar o ambiente (necessária para VecEnv)
def make_env(env_id, rank, seed=0, env_kwargs=None):
    """
    Função utilitária para ambientes multiprocesso.

    :param env_id: (str) o id do ambiente (não usado aqui, mas padrão SB3)
    :param rank: (int) índice do processo
    :param seed: (int) a seed inicial para este processo
    :param env_kwargs: (dict) argumentos para passar para o construtor do ambiente
    """
    def _init():
        env = HierarchicalEnv(**env_kwargs)
        # set_random_seed(seed + rank) # Chamado fora do _init, mas mantemos o padrão
        return env
    set_random_seed(seed)
    return _init

# --- CONFIGURAÇÕES MULTI-GPU ---
def setup_multi_gpu():
    """Configura o ambiente para usar múltiplas GPUs de forma eficiente"""

    # Verificar dispositivos disponíveis
    device_count = torch.cuda.device_count()
    print(f"Dispositivos CUDA disponíveis: {device_count}")

    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memória: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    if device_count >= 2:
        print("\nUsando múltiplas GPUs para acelerar o treinamento!")

        # Configurações para otimizar uso de GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

        # Otimizações PyTorch
        torch.backends.cudnn.benchmark = True  # Otimiza para tamanhos de input fixos
        torch.backends.cudnn.deterministic = False  # Permite otimizações não determinísticas

        # Configurações de memória
        torch.cuda.empty_cache()  # Limpa cache de GPU

        return True
    else:
        print("\nApenas uma GPU disponível ou nenhuma, usando configuração padrão.")
        return False

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Executa e treina um agente SAC no ambiente Hierárquico (ES + TS)")

    # Define caminhos padrão (Ajuste-os para a sua máquina se necessário)
    default_config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'environments', 'scenario_configurations', 'hierarchical_use_case.json')
    default_ns3_path = os.path.expanduser('~/RIC_Personalized/ns-3-mmwave-oran')

    parser.add_argument("--config", type=str, default=default_config_path,
                        help="Caminho para o ficheiro de configuração do cenário")
    parser.add_argument("--output_folder", type=str, default="output_hierarchical_sac",
                        help="Caminho para a pasta de saída")
    parser.add_argument("--ns3_path", type=str, default=default_ns3_path,
                        help="Caminho para o ambiente ns-3 mmWave O-RAN")
    # --- Argumentos SAC (Ajustados para igualar PPO onde relevante) ---
    # total_timesteps: REDUZIDO para 300000 (25% do PPO) por ser SAC, mas ajustável via CLI.
    parser.add_argument("--total_timesteps", type=int, default=300000, # <-- AJUSTADO
                        help="Número total de 'timesteps' para treinar o agente SAC")
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Número de episódios para avaliação após o treino")
    parser.add_argument("--save_path", type=str, default="sac_hierarchical_model",
                        help="Caminho para guardar o modelo SAC treinado")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Caminho para carregar um modelo SAC pré-treinado (ignora treino)")
    
    # NOVOS ARGUMENTOS PARA HIPERPARÂMETROS SAC (Igualando PPO)
    parser.add_argument("--lr", type=float, default=SAC_HYPERPARAMS["learning_rate"],
                        help="Taxa de aprendizado (Learning Rate) - Mantido igual ao PPO")
    # Argumentos específicos do SAC, mantendo os defaults razoáveis
    parser.add_argument("--buffer_size", type=int, default=SAC_HYPERPARAMS["buffer_size"],
                        help="Tamanho do Replay Buffer do SAC")
    parser.add_argument("--learning_starts", type=int, default=SAC_HYPERPARAMS["learning_starts"],
                        help="Quantos passos dar antes de começar a treinar (para encher o buffer)")
    parser.add_argument("--ent_coef", type=str, default=SAC_HYPERPARAMS["ent_coef"],
                        help="Coeficiente de Entropia (pode ser 'auto' ou um float)")

    # --- Argumentos do Ambiente ---
    parser.add_argument("--optimized", action="store_true",
                        help="Ativar o modo otimizado do ns-3")
    parser.add_argument("--verbose_env", action="store_true", 
                        help="Ativar logs detalhados (verbose) para a recompensa do ambiente")
    parser.add_argument("--ts_weight", type=float, default=1.0,
                        help="Peso a aplicar à recompensa de Traffic Steering (TS)")
    parser.add_argument("--heuristic", action="store_true",
                        help="Usar a ação heurística (MultiDiscrete) para ES (afeta o action space!)")

    # --- Argumentos para otimização de GPU e CPU ---
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Número de ambientes paralelos (None = automático baseado nas GPUs)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Dispositivo para treinamento: 'auto', 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("--cpu_threads", type=int, default=None,
                        help="Número de threads CPU para PyTorch (None = automático)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Tamanho do batch para treinamento")

    args = parser.parse_args()

    # --- Configurar Multi-GPU ---
    use_multi_gpu = setup_multi_gpu()

    # --- Otimizações de CPU para não sobrecarregar servidor ---
    import multiprocessing as mp
    cpu_count = mp.cpu_count()

    if args.cpu_threads is None:
        # Usar menos threads que o total para não sobrecarregar
        n_threads = max(2, cpu_count - 4)  # Deixa pelo menos 4 cores livres
    else:
        n_threads = min(args.cpu_threads, cpu_count)

    torch.set_num_threads(n_threads)
    print(f"Threads PyTorch limitadas para: {n_threads} (total: {cpu_count})")

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    optimized = args.optimized
    verbose_env = args.verbose_env 

    # Validação de caminhos
    if not os.path.exists(ns3_path):
        print(f"ERRO: O caminho do ns-3 '{ns3_path}' não foi encontrado.")
        print("Por favor, especifique o caminho correto com o argumento --ns3_path")
        exit(-1)

    if not os.path.exists(configuration_path):
        print(f"ERRO: O ficheiro de configuração '{configuration_path}' não foi encontrado.")
        print("Por favor, especifique o caminho correto com o argumento --config")
        exit(-1)

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Não foi possível abrir o ficheiro '{configuration_path}', a sair.")
        exit(-1)

    scenario_configuration = json.loads(params)

    # --- Criação do Ambiente ---
    print('A preparar o Ambiente Hierárquico (HierarchicalEnv) para Stable Baselines (SAC)')
    env_kwargs = dict(
        ns3_path=ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=output_folder,
        optimized=optimized,
        do_heuristic=args.heuristic,
        ts_reward_weight=args.ts_weight,
        verbose=verbose_env,
        scenario_name="scenario-hierarchical"
    )

    # Usar DummyVecEnv para ambientes vetorizados
    # Número de ambientes baseado no argumento do usuário
    if args.n_envs is not None:
        n_envs = args.n_envs
    else:
        n_envs = 1  # Usar apenas um ambiente para DummyVecEnv

    print(f"Criando {n_envs} ambiente(s) DummyVecEnv...")

    # Usar apenas DummyVecEnv
    env = DummyVecEnv([make_env(env_id=None, rank=i, seed=0, env_kwargs=env_kwargs) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(env_id=None, rank=n_envs, seed=0, env_kwargs=env_kwargs)])

    print('Ambiente Criado e Vetorizado!')
    print(f'Espaço de Ação: {env.action_space}')
    print(f'Espaço de Observação: {env.observation_space}')


    # --- Treino ou Carregamento do Modelo SAC ---
    if args.load_path and os.path.exists(args.load_path + ".zip"):
        print(f"A carregar modelo SAC de: {args.load_path}")
        model = SAC.load(args.load_path, env=env)
        print("Modelo carregado.")
    else:
        print("A criar um novo modelo SAC ('MultiInputPolicy')")

        # Configurar dispositivo baseado nos argumentos
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if use_multi_gpu:
                device = "cuda:0"  # Usar primeira GPU como principal
        else:
            device = args.device

        print(f"Usando dispositivo: {device}")

        # Configurações para DummyVecEnv
        gradient_steps = 1
        learning_starts_adj = args.learning_starts

        # DEFINIÇÃO DOS HIPERPARÂMETROS SAC e ARQUITETURA DE REDE
        model = SAC("MultiInputPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./sac_hierarchical_tensorboard/",
                    learning_rate=args.lr,
                    buffer_size=args.buffer_size,
                    learning_starts=learning_starts_adj,
                    gamma=SAC_HYPERPARAMS["gamma"],
                    tau=SAC_HYPERPARAMS["tau"],
                    train_freq=SAC_HYPERPARAMS["train_freq"],
                    gradient_steps=gradient_steps,  # Otimizado para múltiplos ambientes
                    ent_coef=args.ent_coef,
                    device=device,  # Configurar dispositivo explicitamente
                    # CONFIGURAÇÃO DA ARQUITETURA DA REDE NEURAL (Baseado na prática comum de DRL)
                    policy_kwargs={
                        "net_arch": POLICY_ARCH["net_arch"], # Usa o dicionário corrigido
                        "activation_fn": torch.nn.ReLU,  # ReLU é mais eficiente que Tanh
                        "optimizer_class": torch.optim.Adam,  # Adam é otimizado para GPU
                    }
        )

        # ----------------------------------------------------
        # 1. CONFIGURAÇÃO DA PARADA ANTECIPADA (EARLY STOP)
        # ----------------------------------------------------

        # A: Callback para parar o treino se não houver melhoria
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=EARLY_STOP_CONFIG["patience_evals"], 
            verbose=1
        )

        # B: Callback para avaliação periódica
        eval_callback = EvalCallback(
            eval_env, 
            callback_on_new_best=stop_callback, # Se o modelo melhorar, reinicia o contador de parada
            eval_freq=EARLY_STOP_CONFIG["eval_freq"], 
            n_eval_episodes=EARLY_STOP_CONFIG["n_eval_episodes"],
            log_path="./sac_hierarchical_tensorboard/",
            best_model_save_path=f'./{args.save_path}_best_model/',
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Combina o callback de parada e avaliação
        combined_callback = [eval_callback]

        print(f"A iniciar treino SAC por {args.total_timesteps} timesteps (Early Stop configurado)...")
        
        # O treino pode demorar bastante dependendo do número de timesteps e da velocidade da simulação ns-3
        model.learn(total_timesteps=args.total_timesteps, 
                    callback=combined_callback, # Adiciona o callback combinado aqui
                    progress_bar=True)
        print("Treino concluído (pode ter sido interrompido pelo Early Stop).")

        # Se o treino terminou devido ao Early Stop, o melhor modelo já está salvo.
        # Caso contrário, salva a versão final.
        print(f"A guardar modelo SAC em: {args.save_path}")
        model.save(args.save_path)
        print("Modelo guardado.")

    # --- Avaliação do Modelo Treinado ---
    print(f"\nA iniciar avaliação por {args.eval_episodes} episódios...")
    mean_reward, std_reward = 0, 0 
    all_episode_rewards = []

    for episode in range(args.eval_episodes):
        obs = env.reset()
        terminated = truncated = False
        episode_reward = 0
        step = 0
        print(f"\n--- Episódio de Avaliação {episode + 1} ---")
        while not terminated and not truncated:
            step += 1
            # Usa o modelo treinado para prever a ação
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            current_reward = reward[0]
            episode_reward += current_reward

            # Nota: Ação SAC é contínua, mostramos apenas o primeiro elemento para simplificar
            print(f" Passo {step}: Ação={action[0]}, Recompensa={current_reward:.2f}, Term={terminated[0]}, Trunc={truncated[0]}")

            terminated = terminated[0]
            truncated = truncated[0]

        print(f" Episódio {episode + 1} concluído. Recompensa Total: {episode_reward:.2f}")
        all_episode_rewards.append(episode_reward)

    # Calcular e imprimir estatísticas da avaliação
    if all_episode_rewards:
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        print("\n--- Resultados da Avaliação ---")
        print(f"Número de episódios: {args.eval_episodes}")
        print(f"Recompensa Média: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
         print("\nNenhum episódio de avaliação concluído.")


    print("\nExecução do exemplo Hierárquico com SAC concluída.")
    env.close()
    print("Ambiente fechado.")
