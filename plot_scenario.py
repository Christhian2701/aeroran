#!/usr/bin/env python3
"""
Script para visualizar posições de UEs e eNBs/gNBs do cenário hierárquico
Configurado para O-RAN 5G NR FR1 (3GPP Release 15+)
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import re
import os

# =============================================================================
# Configuração O-RAN 5G FR1 (3GPP TR 38.901 - Urban Macro)
# Deve ser consistente com scenario-hierarchical-xangai-UAV.cc (configuration=1)
# =============================================================================
# Frequência: 3.5 GHz (banda n78, O-RAN típico)
# Bandwidth: 100 MHz (5G NR FR1 max)
FR1_CONFIG = {
    'center_frequency_ghz': 3.5,       # 3.5 GHz (n78 band)
    'bandwidth_mhz': 100,               # 100 MHz
    'isd': 500,                         # Inter-Site Distance: 500m (O-RAN urban macro)
    'center_x': 2000,                   # Centro do cenário X
    'center_y': 2000,                   # Centro do cenário Y
    'area_radius': 750,                 # ISD * 1.5 = 750m (área de mobilidade UEs)
    'uav_coverage_radius': 250,         # ISD / 2 = 250m (raio cobertura UAV FR1)
    'lte_coverage_radius': 500,         # Raio cobertura LTE macro (âncora)
    'uav_altitude': 100,                # Altitude UAV em metros
}
# =============================================================================

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
            # Formato: Time NodeType NodeID X Y Z
            if len(parts) >= 5:
                try:
                    time = float(parts[0])
                    node_type = parts[1]  # LTE, UAV, UE
                    node_id = parts[2]
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5]) if len(parts) > 5 else 1.5

                    # Cria chave única com tipo e id
                    key = f"{node_type}_{node_id}"
                    if key not in traces:
                        traces[key] = {'type': node_type, 'positions': []}
                    traces[key]['positions'].append((time, x, y, z))
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

    # Lê posições iniciais - formato: Time NodeType NodeID X Y Z
    initial_positions = {}  # {(type, id): (x, y, z)}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                try:
                    time = float(parts[0])
                    node_type = parts[1]  # LTE, UAV, UE
                    node_id = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5]) if len(parts) > 5 else 1.5
                    key = (node_type, node_id)
                    if time == 0 and key not in initial_positions:
                        initial_positions[key] = (x, y, z)
                except ValueError:
                    continue

    if not initial_positions:
        print("Nenhuma posição inicial encontrada")
        return {}

    ue_count = sum(1 for (t, _) in initial_positions.keys() if t == 'UE')
    uav_count = sum(1 for (t, _) in initial_positions.keys() if t == 'UAV')
    print(f"Encontrados {ue_count} UEs e {uav_count} UAVs")

    # Parâmetros do cenário O-RAN 5G FR1 (usando configuração global)
    isd = FR1_CONFIG['isd']
    center_x = FR1_CONFIG['center_x']
    center_y = FR1_CONFIG['center_y']
    area_radius = FR1_CONFIG['area_radius']

    # Gera trajetórias para cada UE baseado no padrão (node_id % 6)
    num_waypoints = int(sim_time / 0.4) + 1
    if num_waypoints > 500:
        num_waypoints = 500

    for (node_type, node_id), (base_x, base_y, base_z) in initial_positions.items():
        waypoints = []

        if node_type == 'LTE':
            # LTE base station - fixo
            for i in range(num_waypoints):
                t = i * 0.4
                if t > sim_time:
                    break
                waypoints.append((t, base_x, base_y, base_z))

        elif node_type == 'UAV':
            # UAV - movimento circular lento em torno da posição inicial
            radius = 30.0 + (node_id % 3) * 15.0  # raio 30-60m
            angular_speed = 2.0 * math.pi / 45.0  # uma volta a cada 45s
            for i in range(num_waypoints):
                t = i * 0.4
                if t > sim_time:
                    break
                angle = angular_speed * t + (node_id * math.pi / 4)  # fase inicial diferente
                x = base_x + radius * math.cos(angle)
                y = base_y + radius * math.sin(angle)
                z = base_z + 5 * math.sin(2 * angle)  # pequena variação de altitude
                waypoints.append((t, x, y, z))

        elif node_type == 'UE':
            # =================================================================
            # Shanghai VUR Dataset-based mobility patterns (matching C++ impl)
            # Average speed: 15 km/h (4.2 m/s), Max: 43 km/h (12 m/s)
            # =================================================================
            pattern = node_id % 6

            if pattern == 0:  # Urban main road (~20 km/h with stops)
                base_speed = 5.5  # m/s (~20 km/h) - Shanghai P75
                current_x, current_y = base_x, base_y
                current_time = 0.0
                while current_time < sim_time:
                    speed = base_speed + math.sin(current_time * 0.5 + node_id) * 1.5
                    # Move for 3-5 seconds
                    for _ in range(8 + (node_id % 5)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        current_x += speed * 0.4
                        if current_x > center_x + area_radius:
                            current_x = center_x - area_radius + 50
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Stop at traffic light (2-4 seconds)
                    for _ in range(5 + (node_id % 5)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        waypoints.append((current_time, current_x, current_y, base_z))

            elif pattern == 1:  # Urban slow with turns (~15 km/h)
                speed = 4.2  # m/s (~15 km/h) - Shanghai average
                current_x, current_y = base_x, base_y
                current_time = 0.0
                direction = 0
                while current_time < sim_time:
                    # Move in current direction
                    for _ in range(20 + (node_id % 10)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        if direction == 0: current_x += speed * 0.5
                        elif direction == 1: current_y += speed * 0.5
                        elif direction == 2: current_x -= speed * 0.5
                        else: current_y -= speed * 0.5
                        current_x = max(center_x - area_radius + 20, min(current_x, center_x + area_radius - 20))
                        current_y = max(center_y - area_radius + 20, min(current_y, center_y + area_radius - 20))
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Stop at intersection
                    for _ in range(6 + (node_id % 6)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        waypoints.append((current_time, current_x, current_y, base_z))
                    direction = (direction + 1 + (node_id % 2)) % 4

            elif pattern == 2:  # Intersection crossing (~11 km/h)
                speed = 3.0  # Very slow at intersections
                current_x, current_y = base_x, base_y
                current_time = 0.0
                while current_time < sim_time:
                    # Approach slowly
                    for _ in range(15):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        current_y += speed * 0.4
                        if current_y > center_y + area_radius:
                            current_y = center_y - area_radius + 50
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Wait at intersection
                    for _ in range(12 + (node_id % 12)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Cross faster
                    for _ in range(8):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        current_y += 5.0 * 0.4
                        waypoints.append((current_time, current_x, current_y, base_z))

            elif pattern == 3:  # Roundabout (~12.6 km/h)
                radius = 30.0 + (node_id % 4) * 15.0
                speed = 3.5  # m/s
                angular_speed = speed / radius
                current_time = 0.0
                angle = (node_id % 8) * math.pi / 4
                while current_time < sim_time:
                    # Enter slowly
                    for _ in range(5):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        waypoints.append((current_time, base_x, base_y, base_z))
                    # Circle
                    for _ in range(40):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        angle += angular_speed * 0.5
                        x = base_x + radius * math.cos(angle)
                        y = base_y + radius * math.sin(angle)
                        x = max(center_x - area_radius + 20, min(x, center_x + area_radius - 20))
                        y = max(center_y - area_radius + 20, min(y, center_y + area_radius - 20))
                        waypoints.append((current_time, x, y, base_z))

            elif pattern == 4:  # Stop-and-go dense traffic (~9 km/h) - Most common
                current_x, current_y = base_x, base_y
                current_time = 0.0
                while current_time < sim_time:
                    # Short move
                    for _ in range(4 + (node_id % 4)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        current_x += 2.5 * 0.5
                        current_y += (0.5 if node_id % 2 == 0 else -0.3)
                        if current_x > center_x + area_radius:
                            current_x = center_x - area_radius + 50
                        current_y = max(center_y - area_radius + 20, min(current_y, center_y + area_radius - 20))
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Frequent stops (3-6 seconds)
                    for _ in range(6 + (node_id % 6)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.5
                        waypoints.append((current_time, current_x, current_y, base_z))

            elif pattern == 5:  # Diagonal slow (~14 km/h)
                speed = 4.0  # m/s (~14.4 km/h) - Shanghai typical
                current_x, current_y = base_x, base_y
                current_time = 0.0
                angle_rad = (0.5 + (node_id % 4) * 0.25) * math.pi / 4
                while current_time < sim_time:
                    # Move diagonally
                    for _ in range(15):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        current_x += speed * 0.4 * math.cos(angle_rad)
                        current_y += speed * 0.4 * math.sin(angle_rad)
                        if current_x > center_x + area_radius:
                            current_x = center_x - area_radius + 50
                        if current_y > center_y + area_radius:
                            current_y = center_y - area_radius + 50
                        if current_x < center_x - area_radius:
                            current_x = center_x + area_radius - 50
                        if current_y < center_y - area_radius:
                            current_y = center_y + area_radius - 50
                        waypoints.append((current_time, current_x, current_y, base_z))
                    # Stop periodically
                    for _ in range(4 + (node_id % 5)):
                        if current_time >= sim_time:
                            break
                        current_time += 0.4
                        waypoints.append((current_time, current_x, current_y, base_z))
                    angle_rad += (0.3 if node_id % 2 == 0 else -0.3)

        if waypoints:
            key = f"{node_type}_{node_id}"
            traces[key] = {'type': node_type, 'positions': waypoints}

    ue_traces = sum(1 for k, v in traces.items() if v['type'] == 'UE')
    uav_traces = sum(1 for k, v in traces.items() if v['type'] == 'UAV')
    print(f"Geradas trajetórias para {ue_traces} UEs e {uav_traces} UAVs, tempo: 0s a {sim_time}s")
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
    """Plota posições estáticas de UEs e eNBs com círculos de cobertura"""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Parse eNBs/gNBs primeiro para calcular o ISD automaticamente
    enb_labels, enb_positions = parse_gnuplot_file('enbs.txt')

    # Calcula ISD automaticamente baseado nas posições dos gNBs
    if len(enb_positions) >= 2:
        # Encontra o centro (assume que é o primeiro ou o mais central)
        center_x = np.mean([p[0] for p in enb_positions])
        center_y = np.mean([p[1] for p in enb_positions])

        # Calcula distância média do centro para estimar ISD
        distances = [np.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in enb_positions]
        distances_nonzero = [d for d in distances if d > 10]  # Ignora o central
        if distances_nonzero:
            estimated_isd = np.mean(distances_nonzero)
        else:
            estimated_isd = FR1_CONFIG['isd']

        # Raio de cobertura = ISD/2 (padrão 3GPP)
        uav_coverage_radius = estimated_isd / 2
        lte_coverage_radius = estimated_isd
    else:
        center_x = FR1_CONFIG['center_x']
        center_y = FR1_CONFIG['center_y']
        uav_coverage_radius = FR1_CONFIG['uav_coverage_radius']
        lte_coverage_radius = FR1_CONFIG['lte_coverage_radius']
        estimated_isd = FR1_CONFIG['isd']

    print(f"ISD estimado: {estimated_isd:.0f}m, Raio cobertura UAV: {uav_coverage_radius:.0f}m")

    # Plota círculos de cobertura dos gNBs/UAVs (ANTES dos pontos para ficar atrás)
    if enb_positions:
        for i, (x, y) in enumerate(enb_positions):
            # Verifica se é o LTE central (geralmente o primeiro ou mais próximo do centro)
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            if dist_to_center < estimated_isd * 0.3:  # É o LTE central
                # Círculo de cobertura LTE macro (tracejado)
                circle = Circle((x, y), lte_coverage_radius,
                               fill=False, color='red', alpha=0.4,
                               linestyle='--', linewidth=2)
                ax.add_patch(circle)
            else:  # É um gNB/UAV mmWave
                # Círculo de cobertura UAV (preenchido)
                circle = Circle((x, y), uav_coverage_radius,
                               fill=True, color='orange', alpha=0.15,
                               edgecolor='darkorange', linestyle='-', linewidth=1)
                ax.add_patch(circle)

    # Parse e plota UEs
    ue_labels, ue_positions = parse_gnuplot_file('ues.txt')
    if ue_positions:
        ue_x = [p[0] for p in ue_positions]
        ue_y = [p[1] for p in ue_positions]
        ax.scatter(ue_x, ue_y, c='green', marker='o', s=60,
                   label=f'UEs ({len(ue_positions)})', alpha=0.8,
                   edgecolors='darkgreen', linewidth=0.5, zorder=5)
        for i, label in enumerate(ue_labels):
            ax.annotate(label, (ue_x[i], ue_y[i]), fontsize=6, alpha=0.6)

    # Plota eNBs/gNBs com estilo diferenciado
    if enb_positions:
        for i, (x, y) in enumerate(enb_positions):
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            label = enb_labels[i] if i < len(enb_labels) else str(i)

            if dist_to_center < estimated_isd * 0.3:  # LTE central
                ax.scatter(x, y, c='darkred', marker='^', s=400, zorder=10,
                          edgecolors='black', linewidth=2)
                ax.annotate('gNB', (x, y), fontsize=11, fontweight='bold',
                           xytext=(10, 10), textcoords='offset points', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:  # gNB/UAV mmWave
                ax.scatter(x, y, c='orange', marker='h', s=200, zorder=8,
                          edgecolors='darkorange', linewidth=1.5, alpha=0.9)
                ax.annotate(label, (x, y), fontsize=9, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points')

    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label=f'UEs ({len(ue_positions) if ue_positions else 0})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred',
               markersize=12, label='LTE gNB (âncora)'),
        Line2D([0], [0], marker='h', color='w', markerfacecolor='orange',
               markersize=12, label=f'UAV gNBs ({len(enb_positions)-1 if enb_positions else 0})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Posições Iniciais - Cenário O-RAN com Cobertura', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Ajusta limites para mostrar toda a área
    if ue_positions or enb_positions:
        all_x = [p[0] for p in (ue_positions or [])] + [p[0] for p in (enb_positions or [])]
        all_y = [p[1] for p in (ue_positions or [])] + [p[1] for p in (enb_positions or [])]
        margin = max(uav_coverage_radius, lte_coverage_radius) + 100
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

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
    """Cria animação dinâmica mostrando UEs e UAVs se movendo em tempo real"""

    # Prioridade: 1) mobility-trace.txt completo, 2) gerar trajetórias a partir das posições iniciais
    traces = {}

    # Primeiro tenta o arquivo da simulação NS-3
    if os.path.exists('mobility-trace.txt'):
        traces = parse_mobility_trace('mobility-trace.txt')
        # Verifica se tem dados suficientes (mais que só posições iniciais)
        if traces:
            all_times = set()
            for data in traces.values():
                positions = data['positions'] if isinstance(data, dict) else data
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

    # Separa UEs, UAVs e LTE
    ue_traces = {k: v for k, v in traces.items() if v.get('type') == 'UE'}
    uav_traces = {k: v for k, v in traces.items() if v.get('type') == 'UAV'}
    lte_traces = {k: v for k, v in traces.items() if v.get('type') == 'LTE'}

    print(f"Nós: {len(ue_traces)} UEs, {len(uav_traces)} UAVs, {len(lte_traces)} LTE")

    # Encontra todos os tempos únicos e ordena
    all_times = set()
    for data in traces.values():
        positions = data['positions'] if isinstance(data, dict) else data
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
    fig, ax = plt.subplots(figsize=(14, 12))

    # Determina limites do cenário
    all_x = []
    all_y = []
    for data in traces.values():
        positions = data['positions'] if isinstance(data, dict) else data
        for p in positions:
            all_x.append(p[1])
            all_y.append(p[2] if len(p) > 2 else 0)

    margin = 200
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)

    # Plota LTE base station (fixa)
    if lte_traces:
        for key, data in lte_traces.items():
            pos = data['positions'][0]
            ax.scatter(pos[1], pos[2], c='darkred', marker='^', s=500, zorder=10,
                      edgecolors='black', linewidth=2, label='LTE gNB')
            ax.annotate('gNB', (pos[1], pos[2]), fontsize=12, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            # Círculo de cobertura macro
            # Círculo de cobertura LTE macro (âncora para handover)
            circle = Circle((pos[1], pos[2]), FR1_CONFIG['lte_coverage_radius'],
                           fill=False, color='red', alpha=0.4, linestyle='--', linewidth=2)
            ax.add_patch(circle)

    # Cores para UEs (verde/azul)
    num_ues = len(ue_traces)
    ue_colors = plt.cm.Greens(np.linspace(0.4, 0.9, max(1, num_ues)))

    # Cores para UAVs (laranja/vermelho)
    num_uavs = len(uav_traces)
    uav_colors = plt.cm.Oranges(np.linspace(0.5, 0.9, max(1, num_uavs)))

    # Cria scatter para UEs
    ue_scatter = ax.scatter([], [], c='green', s=60, alpha=0.8, marker='o',
                           edgecolors='darkgreen', linewidth=0.5, zorder=5, label=f'UEs ({num_ues})')

    # Cria scatter para UAVs
    uav_scatter = ax.scatter([], [], c='orange', s=150, alpha=0.9, marker='h',
                            edgecolors='darkorange', linewidth=1.5, zorder=8, label=f'UAVs ({num_uavs})')

    # Círculos de cobertura dos UAVs (serão atualizados)
    uav_coverage_circles = []
    for _ in uav_traces:
        # Círculo de cobertura UAV FR1 (ISD/2 = 250m para 3.5 GHz)
        circle = Circle((0, 0), FR1_CONFIG['uav_coverage_radius'],
                        fill=True, color='orange', alpha=0.15,
                        edgecolor='darkorange', linestyle='-', linewidth=1)
        ax.add_patch(circle)
        uav_coverage_circles.append(circle)

    # Cria trails (rastros) para UEs
    trail_length = 15
    ue_trails = {}
    ue_trail_lines = {}
    for i, node_id in enumerate(ue_traces.keys()):
        color = ue_colors[i % len(ue_colors)]
        line, = ax.plot([], [], '-', color=color, alpha=0.4, linewidth=1)
        ue_trail_lines[node_id] = line
        ue_trails[node_id] = []

    # Cria trails para UAVs
    uav_trails = {}
    uav_trail_lines = {}
    for i, node_id in enumerate(uav_traces.keys()):
        color = uav_colors[i % len(uav_colors)]
        line, = ax.plot([], [], '-', color=color, alpha=0.6, linewidth=2)
        uav_trail_lines[node_id] = line
        uav_trails[node_id] = []

    # Texto do tempo
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_title('Mobilidade de UEs e UAVs - Cenário O-RAN', fontsize=16, fontweight='bold')

    def init():
        ue_scatter.set_offsets(np.empty((0, 2)))
        uav_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        for line in ue_trail_lines.values():
            line.set_data([], [])
        for line in uav_trail_lines.values():
            line.set_data([], [])
        for circle in uav_coverage_circles:
            circle.center = (0, 0)
        return [ue_scatter, uav_scatter, time_text] + list(ue_trail_lines.values()) + list(uav_trail_lines.values()) + uav_coverage_circles

    def animate(frame_idx):
        target_time = times[frame_idx]

        # Atualiza UEs
        ue_positions = []
        for i, (node_id, data) in enumerate(ue_traces.items()):
            positions = data['positions']
            closest = min(positions, key=lambda p: abs(p[0] - target_time))
            x, y = closest[1], closest[2]
            ue_positions.append([x, y])

            # Atualiza trail
            ue_trails[node_id].append((x, y))
            if len(ue_trails[node_id]) > trail_length:
                ue_trails[node_id].pop(0)

            if ue_trails[node_id]:
                trail_x = [p[0] for p in ue_trails[node_id]]
                trail_y = [p[1] for p in ue_trails[node_id]]
                ue_trail_lines[node_id].set_data(trail_x, trail_y)

        if ue_positions:
            ue_scatter.set_offsets(ue_positions)

        # Atualiza UAVs
        uav_positions = []
        for i, (node_id, data) in enumerate(uav_traces.items()):
            positions = data['positions']
            closest = min(positions, key=lambda p: abs(p[0] - target_time))
            x, y = closest[1], closest[2]
            uav_positions.append([x, y])

            # Atualiza círculo de cobertura do UAV
            if i < len(uav_coverage_circles):
                uav_coverage_circles[i].center = (x, y)

            # Atualiza trail
            uav_trails[node_id].append((x, y))
            if len(uav_trails[node_id]) > trail_length:
                uav_trails[node_id].pop(0)

            if uav_trails[node_id]:
                trail_x = [p[0] for p in uav_trails[node_id]]
                trail_y = [p[1] for p in uav_trails[node_id]]
                uav_trail_lines[node_id].set_data(trail_x, trail_y)

        if uav_positions:
            uav_scatter.set_offsets(uav_positions)

        time_text.set_text(f'Tempo: {target_time:.2f}s\nUEs: {len(ue_traces)}\nUAVs: {len(uav_traces)}')

        return [ue_scatter, uav_scatter, time_text] + list(ue_trail_lines.values()) + list(uav_trail_lines.values()) + uav_coverage_circles

    # Cria animação
    interval = max(20, int(100 / speed))  # ms entre frames
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(times), interval=interval,
                                   blit=True, repeat=True)

    if save_gif:
        print("Salvando animação como mobility_animation.gif...")
        try:
            anim.save('mobility_animation.gif', writer='pillow', fps=15, dpi=100)
            print("Animação salva em: mobility_animation.gif")
        except Exception as e:
            print(f"Erro ao salvar GIF: {e}")
            print("Tentando salvar como MP4...")
            try:
                anim.save('mobility_animation.mp4', writer='ffmpeg', fps=15)
                print("Animação salva em: mobility_animation.mp4")
            except:
                print("Não foi possível salvar. Instale pillow ou ffmpeg.")

    ax.legend(loc='upper right', fontsize=11)
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
