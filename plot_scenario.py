#!/usr/bin/env python3
"""
Script para visualizar posições de UEs e eNBs/gNBs do cenário hierárquico
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
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

def generate_shanghai_trajectories_from_initial(filename, sim_time=30.0):
    """
    Gera trajetórias Shanghai baseadas nas posições iniciais do mobility-trace.txt
    Replica a lógica do ShanghaiMobilityGenerator do C++
    """
    import math

    traces = {}

    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado")
        return {}

    print(f"Lendo posições iniciais de: {filename}...")

    # Lê posições iniciais
    initial_positions = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    time = float(parts[0])
                    node_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    if time == 0 and node_id not in initial_positions:
                        initial_positions[node_id] = (x, y)
                except ValueError:
                    continue

    if not initial_positions:
        print("Nenhuma posição inicial encontrada")
        return {}

    print(f"Gerando trajetórias para {len(initial_positions)} UEs...")

    # Parâmetros do cenário O-RAN 5G FR1
    # ISD = 500m para 5G FR1 3.5 GHz (O-RAN urban macro)
    isd = 500
    center_x, center_y = 2000, 2000
    area_radius = isd * 1.5  # 750m

    # Gera trajetórias para cada UE baseado no padrão (node_id % 6)
    num_waypoints = int(sim_time / 0.4) + 1
    if num_waypoints > 500:
        num_waypoints = 500

    for node_id, (base_x, base_y) in initial_positions.items():
        pattern = node_id % 6
        waypoints = []

        if pattern == 0:  # Highway - movimento reto
            speed = 12.5  # m/s (~45 km/h)
            for i in range(num_waypoints):
                t = i * 0.4
                if t > sim_time:
                    break
                x = base_x + i * speed * 0.4
                y = base_y
                # Wrap around
                if x > center_x + area_radius:
                    x = center_x - area_radius + ((x - center_x - area_radius) % (2.0 * area_radius))
                waypoints.append((t, x, y))

        elif pattern == 1:  # Urban turn - curva
            speed = 7.5
            half = num_waypoints // 2
            for i in range(half):
                t = i * 0.4
                if t > sim_time:
                    break
                x = base_x + i * speed * 0.4
                y = base_y
                if x > center_x + area_radius:
                    x = center_x + area_radius - 10
                waypoints.append((t, x, y))
            last_x = waypoints[-1][1] if waypoints else base_x
            for i in range(1, half):
                t = half * 0.4 + i * 0.4
                if t > sim_time:
                    break
                x = last_x
                y = base_y + i * speed * 0.4 * 0.8
                if y > center_y + area_radius:
                    y = center_y + area_radius - 10
                waypoints.append((t, x, y))

        elif pattern == 2:  # Intersection - cruzamento perpendicular
            speed = 10.0
            for i in range(num_waypoints):
                t = i * 0.3
                if t > sim_time:
                    break
                x = base_x + 50.0
                y = base_y + i * speed * 0.3
                if y > center_y + area_radius:
                    y = center_y - area_radius + ((y - center_y - area_radius) % (2.0 * area_radius))
                waypoints.append((t, x, y))

        elif pattern == 3:  # Roundabout - circular
            radius = 50.0 + (node_id % 5) * 20.0
            angular_speed = 2.0 * math.pi / (num_waypoints * 0.25)
            for i in range(num_waypoints):
                t = i * 0.25
                if t > sim_time:
                    break
                angle = angular_speed * i
                x = base_x + radius * math.cos(angle)
                y = base_y + radius * math.sin(angle)
                x = max(center_x - area_radius + 10, min(x, center_x + area_radius - 10))
                y = max(center_y - area_radius + 10, min(y, center_y + area_radius - 10))
                waypoints.append((t, x, y))

        elif pattern == 4:  # Stop-and-go - semáforo
            current_x = base_x
            current_time = 0.0
            segment = 0
            while current_time < sim_time and segment < 20:
                # Move phase
                for _ in range(8):
                    if current_time >= sim_time:
                        break
                    current_time += 0.3
                    current_x += 2.5
                    if current_x > center_x + area_radius:
                        current_x = center_x - area_radius + 50
                    waypoints.append((current_time, current_x, base_y))
                # Stop phase
                for _ in range(6):
                    if current_time >= sim_time:
                        break
                    current_time += 0.3
                    waypoints.append((current_time, current_x, base_y))
                segment += 1

        elif pattern == 5:  # Diagonal
            speed = 8.0
            for i in range(num_waypoints):
                t = i * 0.35
                if t > sim_time:
                    break
                x = base_x + i * speed * 0.35 * 0.8
                y = base_y + i * speed * 0.35 * 0.6
                if x > center_x + area_radius:
                    x = center_x - area_radius + ((x - center_x - area_radius) % (2.0 * area_radius))
                if y > center_y + area_radius:
                    y = center_y - area_radius + ((y - center_y - area_radius) % (2.0 * area_radius))
                waypoints.append((t, x, y))

        if waypoints:
            traces[str(node_id)] = waypoints

    print(f"Geradas trajetórias para {len(traces)} UEs, tempo: 0s a {sim_time}s")
    return traces

def parse_ns2_mobility_tcl(filename):
    """Parse arquivo TCL de mobilidade no formato NS-2"""
    traces = {}  # {node_id: [(time, x, y), ...]}
    initial_positions = {}  # {node_id: (x, y)}

    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado")
        return {}

    print(f"Lendo arquivo TCL: {filename}...")

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse posição inicial: $node_(ID) set X_ VALUE
            match_x = re.search(r'\$node_\((\d+)\)\s+set\s+X_\s+([-\d.]+)', line)
            match_y = re.search(r'\$node_\((\d+)\)\s+set\s+Y_\s+([-\d.]+)', line)

            if match_x:
                node_id = match_x.group(1)
                x = float(match_x.group(2))
                if node_id not in initial_positions:
                    initial_positions[node_id] = [0, 0]
                initial_positions[node_id][0] = x

            if match_y:
                node_id = match_y.group(1)
                y = float(match_y.group(2))
                if node_id not in initial_positions:
                    initial_positions[node_id] = [0, 0]
                initial_positions[node_id][1] = y

            # Parse movimento: $ns_ at TIME "$node_(ID) setdest X Y SPEED"
            match_move = re.search(r'\$ns_\s+at\s+([\d.]+)\s+"\$node_\((\d+)\)\s+setdest\s+([-\d.]+)\s+([-\d.]+)', line)
            if match_move:
                time = float(match_move.group(1))
                node_id = match_move.group(2)
                x = float(match_move.group(3))
                y = float(match_move.group(4))

                if node_id not in traces:
                    traces[node_id] = []
                traces[node_id].append((time, x, y))

    # Adiciona posições iniciais (tempo 0)
    for node_id, (x, y) in initial_positions.items():
        if node_id not in traces:
            traces[node_id] = []
        # Insere no início
        traces[node_id].insert(0, (0.0, x, y))

    # Ordena por tempo
    for node_id in traces:
        traces[node_id].sort(key=lambda p: p[0])

    print(f"Carregados {len(traces)} nós com mobilidade")
    if traces:
        all_times = set()
        for positions in traces.values():
            for pos in positions:
                all_times.add(pos[0])
        print(f"Período: 0s a {max(all_times):.1f}s")

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

def plot_dynamic_animation(speed=1.0, save_gif=False, tcl_file=None):
    """Cria animação dinâmica mostrando UEs se movendo em tempo real"""

    # Prioridade: 1) mobility-trace.txt completo, 2) gerar trajetórias a partir das posições iniciais
    traces = {}

    # Primeiro tenta o arquivo da simulação NS-3
    if os.path.exists('mobility-trace.txt'):
        traces = parse_mobility_trace('mobility-trace.txt')
        # Verifica se tem dados suficientes (mais que só posições iniciais)
        if traces:
            all_times = set()
            for positions in traces.values():
                for pos in positions:
                    all_times.add(pos[0])
            if len(all_times) <= 1:
                print("mobility-trace.txt só tem posições iniciais.")
                print("Gerando trajetórias Shanghai a partir das posições iniciais...")
                traces = generate_shanghai_trajectories_from_initial('mobility-trace.txt', sim_time=60.0)

    # Se não encontrou dados suficientes e foi especificado TCL
    if not traces:
        if tcl_file and os.path.exists(tcl_file):
            traces = parse_ns2_mobility_tcl(tcl_file)

    if not traces:
        print("Nenhum trace de mobilidade encontrado.")
        return

    # Parse eNBs
    enb_labels, enb_positions = parse_gnuplot_file('enbs.txt')

    # Encontra todos os tempos únicos e ordena
    all_times = set()
    for positions in traces.values():
        for pos in positions:
            all_times.add(pos[0])
    times = sorted(all_times)

    # Limita número de frames para performance
    max_frames = 200
    if len(times) > max_frames:
        step = len(times) // max_frames
        times = times[::step]

    print(f"Criando animação com {len(times)} frames...")
    print(f"Tempo de simulação: {times[0]:.1f}s a {times[-1]:.1f}s")

    # Configura figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # Determina limites do cenário (inclui eNBs)
    all_x = []
    all_y = []
    for positions in traces.values():
        for p in positions:
            all_x.append(p[1])
            all_y.append(p[2] if len(p) > 2 else 0)

    # Inclui posições dos eNBs nos limites
    if enb_positions:
        for pos in enb_positions:
            all_x.append(pos[0])
            all_y.append(pos[1])

    margin = 300
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Plota eNBs/gNBs (fixos)
    # Calcula ISD baseado nas posições dos eNBs (distância entre centro e periféricos)
    if enb_positions and len(enb_positions) > 1:
        center = enb_positions[0]  # Primeiro eNB é o central
        distances = []
        for pos in enb_positions[1:]:
            dist = np.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
            if dist > 100:  # Ignora co-localizados
                distances.append(dist)
        isd = np.mean(distances) if distances else 1700
        # Raio de cobertura = ISD / sqrt(3) para layout hexagonal
        coverage_radius = isd / np.sqrt(3)
        print(f"ISD detectado: {isd:.0f}m, Raio de cobertura: {coverage_radius:.0f}m")
    else:
        coverage_radius = 981  # Default para ISD=1700

    if enb_positions:
        enb_x = [p[0] for p in enb_positions]
        enb_y = [p[1] for p in enb_positions]
        # Triangulos grandes vermelhos para as torres
        ax.scatter(enb_x, enb_y, c='darkred', marker='^', s=400, label=f'eNBs/gNBs (r={coverage_radius:.0f}m)', zorder=10, edgecolors='black', linewidth=1.5)
        for i, label in enumerate(enb_labels):
            ax.annotate(f'Cell {label}', (enb_x[i], enb_y[i]), fontsize=10, fontweight='bold',
                       xytext=(8, 8), textcoords='offset points', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        # Desenha círculos de cobertura com raio correto
        for pos in enb_positions:
            circle = Circle((pos[0], pos[1]), coverage_radius, fill=False, color='red', alpha=0.3, linestyle='--', linewidth=2)
            ax.add_patch(circle)

    # Cores para UEs
    num_ues = len(traces)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_ues)))

    # Cria scatter para UEs (será atualizado)
    ue_scatter = ax.scatter([], [], c=[], s=80, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)

    # Cria trails (rastros) para cada UE
    trail_length = 10  # Número de posições anteriores a mostrar
    trails = {}
    trail_lines = {}
    for i, node_id in enumerate(traces.keys()):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], '-', color=color, alpha=0.3, linewidth=1)
        trail_lines[node_id] = line
        trails[node_id] = []

    # Texto do tempo
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    title = ax.set_title('Mobilidade de UEs - Shanghai Dataset', fontsize=14)

    def init():
        ue_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        for line in trail_lines.values():
            line.set_data([], [])
        return [ue_scatter, time_text] + list(trail_lines.values())

    def animate(frame_idx):
        target_time = times[frame_idx]

        positions_list = []
        colors_list = []

        for i, (node_id, positions) in enumerate(traces.items()):
            # Encontra posição mais próxima do tempo alvo
            closest = min(positions, key=lambda p: abs(p[0] - target_time))
            x = closest[1]
            y = closest[2] if len(closest) > 2 else 0
            positions_list.append([x, y])
            colors_list.append(colors[i % len(colors)])

            # Atualiza trail
            trails[node_id].append((x, y))
            if len(trails[node_id]) > trail_length:
                trails[node_id].pop(0)

            # Atualiza linha do trail
            if trails[node_id]:
                trail_x = [p[0] for p in trails[node_id]]
                trail_y = [p[1] for p in trails[node_id]]
                trail_lines[node_id].set_data(trail_x, trail_y)

        ue_scatter.set_offsets(positions_list)
        ue_scatter.set_color(colors_list)
        time_text.set_text(f'Tempo: {target_time:.2f}s\nUEs: {len(traces)}')

        return [ue_scatter, time_text] + list(trail_lines.values())

    # Cria animação
    interval = max(20, int(100 / speed))  # ms entre frames
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(times), interval=interval,
                                   blit=True, repeat=True)

    if save_gif:
        print("Salvando animação como mobility_animation.gif...")
        try:
            anim.save('mobility_animation.gif', writer='pillow', fps=15)
            print("Animação salva em: mobility_animation.gif")
        except Exception as e:
            print(f"Erro ao salvar GIF: {e}")
            print("Tentando salvar como MP4...")
            try:
                anim.save('mobility_animation.mp4', writer='ffmpeg', fps=15)
                print("Animação salva em: mobility_animation.mp4")
            except:
                print("Não foi possível salvar. Instale pillow ou ffmpeg.")

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return anim

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
        elif sys.argv[1] == 'dynamic':
            # Velocidade opcional: python plot_scenario.py dynamic 2.0
            save_gif = '--save' in sys.argv
            speed = 1.0
            for arg in sys.argv[2:]:
                if arg != '--save':
                    try:
                        speed = float(arg)
                    except ValueError:
                        pass
            plot_dynamic_animation(speed=speed, save_gif=save_gif)
        else:
            print("Uso: python plot_scenario.py [mobility|animation|dynamic]")
            print("  mobility  - Trajetórias estáticas")
            print("  animation - Gera frames PNG")
            print("  dynamic   - Animação em tempo real")
            print("  dynamic 2.0 --save  - Animação 2x mais rápida e salva GIF")
    else:
        plot_static_positions()
        print("\nPara ver trajetórias: python plot_scenario.py mobility")
        print("Para animação dinâmica: python plot_scenario.py dynamic")
