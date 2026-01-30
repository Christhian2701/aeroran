#!/usr/bin/env python3
"""
Script para visualizar posições de UEs e eNBs/gNBs do cenário hierárquico
"""

import matplotlib.pyplot as plt
import re
import os

def parse_gnuplot_file(filename):
    """Parse arquivo gnuplot e extrai posições"""
    positions = []
    labels = []

    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado")
        return [], []

    with open(filename, 'r') as f:
        for line in f:
            # Formato: set label "ID" at X,Y ...
            match = re.search(r'set label "(\d+)" at ([\d.]+),([\d.]+)', line)
            if match:
                labels.append(match.group(1))
                positions.append((float(match.group(2)), float(match.group(3))))

    return labels, positions

def parse_mobility_trace(filename):
    """Parse arquivo de trace de mobilidade"""
    traces = {}  # {node_id: [(time, x, y, z), ...]}

    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado")
        return {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    time = float(parts[0])
                    node_id = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4]) if len(parts) > 4 else 1.5

                    if node_id not in traces:
                        traces[node_id] = []
                    traces[node_id].append((time, x, y, z))
                except ValueError:
                    continue

    return traces

def plot_static_positions():
    """Plota posições estáticas de UEs e eNBs"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Parse UEs
    ue_labels, ue_positions = parse_gnuplot_file('ues.txt')
    if ue_positions:
        ue_x = [p[0] for p in ue_positions]
        ue_y = [p[1] for p in ue_positions]
        ax.scatter(ue_x, ue_y, c='green', marker='o', s=50, label=f'UEs ({len(ue_positions)})', alpha=0.7)
        for i, label in enumerate(ue_labels):
            ax.annotate(label, (ue_x[i], ue_y[i]), fontsize=6, alpha=0.7)

    # Parse eNBs/gNBs
    enb_labels, enb_positions = parse_gnuplot_file('enbs.txt')
    if enb_positions:
        enb_x = [p[0] for p in enb_positions]
        enb_y = [p[1] for p in enb_positions]
        ax.scatter(enb_x, enb_y, c='red', marker='^', s=200, label=f'eNBs/gNBs ({len(enb_positions)})')
        for i, label in enumerate(enb_labels):
            ax.annotate(label, (enb_x[i], enb_y[i]), fontsize=10, fontweight='bold')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Posições Iniciais - Cenário Hierárquico')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('scenario_positions.png', dpi=150)
    print("Gráfico salvo em: scenario_positions.png")
    plt.show()

def plot_mobility_traces():
    """Plota trajetórias de mobilidade ao longo do tempo"""
    traces = parse_mobility_trace('mobility-trace.txt')

    if not traces:
        print("Nenhum trace de mobilidade encontrado.")
        print("Execute a simulação com positionAllocator=2 para gerar o trace.")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    # Cores para diferentes UEs
    colors = plt.cm.tab20(range(min(20, len(traces))))

    for i, (node_id, positions) in enumerate(traces.items()):
        if i >= 20:  # Limita a 20 UEs para legibilidade
            break
        x = [p[1] for p in positions]
        y = [p[2] for p in positions]
        color = colors[i % len(colors)]

        # Plota trajetória
        ax.plot(x, y, '-', color=color, alpha=0.6, linewidth=1)
        # Marca início (círculo) e fim (quadrado)
        ax.scatter(x[0], y[0], c=[color], marker='o', s=100, edgecolors='black', zorder=5)
        ax.scatter(x[-1], y[-1], c=[color], marker='s', s=100, edgecolors='black', zorder=5)
        ax.annotate(f'UE{node_id}', (x[0], y[0]), fontsize=7)

    # Parse eNBs/gNBs
    enb_labels, enb_positions = parse_gnuplot_file('enbs.txt')
    if enb_positions:
        enb_x = [p[0] for p in enb_positions]
        enb_y = [p[1] for p in enb_positions]
        ax.scatter(enb_x, enb_y, c='red', marker='^', s=300, label='eNBs/gNBs', zorder=10)
        for i, label in enumerate(enb_labels):
            ax.annotate(f'Cell {label}', (enb_x[i], enb_y[i]), fontsize=10, fontweight='bold')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajetórias de Mobilidade Shanghai - Cenário Hierárquico')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('mobility_traces.png', dpi=150)
    print(f"Gráfico salvo em: mobility_traces.png")
    print(f"Total de UEs rastreados: {len(traces)}")
    plt.show()

def plot_mobility_animation_frames():
    """Gera frames para animação da mobilidade"""
    traces = parse_mobility_trace('mobility-trace.txt')

    if not traces:
        print("Nenhum trace de mobilidade encontrado.")
        return

    # Encontra tempos únicos
    all_times = set()
    for positions in traces.values():
        for pos in positions:
            all_times.add(pos[0])

    times = sorted(all_times)
    print(f"Gerando {min(50, len(times))} frames de animação...")

    # Cria diretório para frames
    os.makedirs('frames', exist_ok=True)

    # Seleciona frames espaçados uniformemente
    frame_indices = [int(i * len(times) / 50) for i in range(min(50, len(times)))]

    for frame_num, time_idx in enumerate(frame_indices):
        target_time = times[time_idx]

        fig, ax = plt.subplots(figsize=(10, 10))

        for node_id, positions in traces.items():
            # Encontra posição mais próxima do tempo alvo
            closest = min(positions, key=lambda p: abs(p[0] - target_time))
            ax.scatter(closest[1], closest[2], s=50, alpha=0.7)

        # eNBs
        enb_labels, enb_positions = parse_gnuplot_file('enbs.txt')
        if enb_positions:
            enb_x = [p[0] for p in enb_positions]
            enb_y = [p[1] for p in enb_positions]
            ax.scatter(enb_x, enb_y, c='red', marker='^', s=200)

        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 4000)
        ax.set_title(f'Tempo: {target_time:.2f}s')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.savefig(f'frames/frame_{frame_num:03d}.png', dpi=100)
        plt.close()

    print("Frames salvos em: frames/")
    print("Para criar GIF: convert -delay 10 frames/*.png mobility.gif")

if __name__ == '__main__':
    import sys

    print("=" * 50)
    print("Visualização do Cenário Hierárquico Shanghai")
    print("=" * 50)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'mobility':
            plot_mobility_traces()
        elif sys.argv[1] == 'animation':
            plot_mobility_animation_frames()
        else:
            print("Uso: python plot_scenario.py [mobility|animation]")
    else:
        plot_static_positions()
        print("\nPara ver trajetórias: python plot_scenario.py mobility")
