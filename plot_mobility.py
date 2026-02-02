import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_trajectories(file_path):
    print(f"Lendo arquivo: {file_path}...")
    
    # Ler o arquivo de texto (tab-separated)
    # O C++ usa "#Time" no cabeçalho, o pandas vai ler isso como coluna
    try:
        df = pd.read_csv(file_path, sep='\t')
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado. Rode a simulação NS-3 primeiro.")
        return

    # Limpar nomes das colunas (remover espaços e o #)
    df.columns = df.columns.str.replace('#', '').str.strip()
    
    # Configurar o plot
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
    
    # Dicionário dos padrões definidos no C++ (para a legenda)
    pattern_names = {
        0: "0: Highway (Reto)",
        1: "1: Urban Turn (Curva)",
        2: "2: Intersection (Cruzamento)",
        3: "3: Roundabout (Circular)",
        4: "4: Stop-and-Go (Semáforo)",
        5: "5: Diagonal"
    }
    
    # Definir cores para os 6 padrões
    colors = cm.get_cmap('tab10', 6)
    
    # Agrupar por NodeID para plotar cada trajetória
    unique_nodes = df['NodeID'].unique()
    print(f"Plotando trajetórias para {len(unique_nodes)} UEs...")

    # Variáveis para garantir que a legenda apareça apenas uma vez por padrão
    patterns_plotted = set()

    for node_id in unique_nodes:
        # Filtrar dados do nó
        node_data = df[df['NodeID'] == node_id]
        
        # Determinar o padrão (mesma lógica do C++: ueId % 6)
        pattern_id = node_id % 6
        label = pattern_names[pattern_id]
        color = colors(pattern_id)
        
        # Adicionar à legenda apenas se for a primeira vez que plotamos este padrão
        plot_label = label if pattern_id not in patterns_plotted else None
        patterns_plotted.add(pattern_id)
        
        # Plotar a linha da trajetória
        ax.plot(node_data['X'], node_data['Y'], 
                marker='', linestyle='-', linewidth=1.5, alpha=0.6,
                color=color, label=plot_label)
        
        # Marcar o ponto INICIAL (triângulo verde)
        ax.plot(node_data['X'].iloc[0], node_data['Y'].iloc[0], 
                marker='^', color='green', markersize=4, alpha=0.8)
        
        # Marcar o ponto FINAL (quadrado vermelho)
        ax.plot(node_data['X'].iloc[-1], node_data['Y'].iloc[-1], 
                marker='s', color='red', markersize=4, alpha=0.8)

    # Estilização
    plt.title(f'Mobilidade NS-3: Cenário Shanghai ({len(unique_nodes)} UEs)', fontsize=16)
    plt.xlabel('Posição X (metros)', fontsize=12)
    plt.ylabel('Posição Y (metros)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Importante para não distorcer formas (círculos parecerem ovais)
    
    # Legenda inteligente
    plt.legend(title="Padrões de Movimento", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    output_img = 'shanghai_mobility_map.png'
    plt.savefig(output_img, dpi=300)
    print(f"Gráfico salvo como: {output_img}")
    plt.show()

if __name__ == "__main__":
    import sys
    # Usa arquivo passado como argumento ou mobility-trace.txt por padrão
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'mobility-trace.txt'
    plot_trajectories(file_path) 
