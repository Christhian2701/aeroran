#!/usr/bin/env python3
"""
UAV-BS Scenario Visualization Script
=====================================
Visualizes UAV gNBs and UE mobility from ns-3 simulation traces.
Supports the scenario-hierarchical-xangai-UAV.cc simulation.

Features:
- Animated visualization of UAV trajectories
- UE mobility with Shanghai patterns
- LTE eNB static position
- Trail visualization for all nodes
- 2D and 3D plots

Usage:
    python3 plot_scenario_uav.py [mobility-trace.txt] [--animate] [--save] [--3d]
    python3 plot_scenario_uav.py --demo  # Generate demo data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import argparse
import sys
import math
import os
from collections import defaultdict


# ==============================================================================
# Shanghai Mobility Pattern Generator (mirrors C++ implementation)
# ==============================================================================

def generate_shanghai_trajectories(initial_positions, sim_time=30.0, center_x=2000, center_y=2000, area_radius=750):
    """
    Generate Shanghai-like urban mobility patterns for UEs.
    Mirrors the ShanghaiMobilityGenerator from the C++ code.

    Patterns:
        0: Highway - straight movement (~45 km/h)
        1: Urban turn - L-shaped path (~27 km/h)
        2: Intersection - perpendicular crossing (~36 km/h)
        3: Roundabout - circular movement
        4: Stop-and-go - traffic light simulation
        5: Diagonal - diagonal movement
    """
    trajectories = {}

    num_waypoints = int(sim_time / 0.4) + 1
    if num_waypoints > 500:
        num_waypoints = 500

    for node_id, (base_x, base_y) in initial_positions.items():
        pattern = node_id % 6
        times = []
        x_coords = []
        y_coords = []
        z_coords = []

        if pattern == 0:  # Highway - straight movement
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
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)

        elif pattern == 1:  # Urban turn - L-shaped
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
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)
            last_x = x_coords[-1] if x_coords else base_x
            for i in range(1, half):
                t = half * 0.4 + i * 0.4
                if t > sim_time:
                    break
                x = last_x
                y = base_y + i * speed * 0.4 * 0.8
                if y > center_y + area_radius:
                    y = center_y + area_radius - 10
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)

        elif pattern == 2:  # Intersection - perpendicular
            speed = 10.0
            for i in range(num_waypoints):
                t = i * 0.3
                if t > sim_time:
                    break
                x = base_x + 50.0
                y = base_y + i * speed * 0.3
                if y > center_y + area_radius:
                    y = center_y - area_radius + ((y - center_y - area_radius) % (2.0 * area_radius))
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)

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
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)

        elif pattern == 4:  # Stop-and-go - traffic light
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
                    times.append(current_time)
                    x_coords.append(current_x)
                    y_coords.append(base_y)
                    z_coords.append(1.5)
                # Stop phase
                for _ in range(6):
                    if current_time >= sim_time:
                        break
                    current_time += 0.3
                    times.append(current_time)
                    x_coords.append(current_x)
                    y_coords.append(base_y)
                    z_coords.append(1.5)
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
                times.append(t)
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(1.5)

        if times:
            trajectories[node_id] = {
                'times': times,
                'x': x_coords,
                'y': y_coords,
                'z': z_coords,
                'type': 'UE',
                'pattern': pattern
            }

    return trajectories


# ==============================================================================
# UAV Flight Pattern Generator (mirrors C++ implementation)
# ==============================================================================

def generate_uav_trajectories(base_positions, sim_time=30.0, pattern=1, altitude=100.0,
                               max_speed=15.0, orbit_radius=100.0, patrol_length=200.0):
    """
    Generate UAV flight trajectories.

    Patterns:
        0: HOVERING - hover with GPS drift
        1: CIRCULAR_ORBIT - orbit around center
        2: PATROL_LINEAR - back and forth patrol
        3: FIGURE_EIGHT - figure-8 trajectory
        4: GRID_COVERAGE - lawnmower grid sweep
        5: ALTITUDE_VAR - vertical oscillation
    """
    trajectories = {}

    num_waypoints = int(sim_time / 0.5) + 1

    for uav_id, (base_x, base_y) in enumerate(base_positions):
        times = []
        x_coords = []
        y_coords = []
        z_coords = []

        if pattern == 0:  # HOVERING with GPS drift
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                drift_x = 2.0 * math.sin(t * 0.5 + uav_id)
                drift_y = 2.0 * math.cos(t * 0.3 + uav_id)
                times.append(t)
                x_coords.append(base_x + drift_x)
                y_coords.append(base_y + drift_y)
                z_coords.append(altitude)

        elif pattern == 1:  # CIRCULAR_ORBIT
            angular_speed = max_speed / orbit_radius
            phase_offset = uav_id * 2 * math.pi / max(len(base_positions), 1)
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                angle = angular_speed * t + phase_offset
                times.append(t)
                x_coords.append(base_x + orbit_radius * math.cos(angle))
                y_coords.append(base_y + orbit_radius * math.sin(angle))
                z_coords.append(altitude)

        elif pattern == 2:  # PATROL_LINEAR
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                # Calculate position along patrol path
                cycle_time = 2 * patrol_length / max_speed
                phase = (t % cycle_time) / cycle_time
                if phase < 0.5:
                    offset = patrol_length * (phase * 2)
                else:
                    offset = patrol_length * (2 - phase * 2)
                times.append(t)
                x_coords.append(base_x + offset - patrol_length / 2)
                y_coords.append(base_y)
                z_coords.append(altitude)

        elif pattern == 3:  # FIGURE_EIGHT
            angular_speed = max_speed / orbit_radius
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                angle = angular_speed * t
                times.append(t)
                x_coords.append(base_x + orbit_radius * math.sin(angle))
                y_coords.append(base_y + orbit_radius * math.sin(2 * angle) / 2)
                z_coords.append(altitude)

        elif pattern == 4:  # GRID_COVERAGE
            grid_speed = max_speed * 0.7
            sweep_width = orbit_radius * 2
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                row = int(t * grid_speed / sweep_width) % 4
                progress = (t * grid_speed) % sweep_width
                if row % 2 == 1:
                    progress = sweep_width - progress
                times.append(t)
                x_coords.append(base_x - sweep_width/2 + progress)
                y_coords.append(base_y - sweep_width/2 + row * sweep_width/4)
                z_coords.append(altitude)

        elif pattern == 5:  # ALTITUDE_VAR
            alt_amplitude = 20.0
            alt_period = 10.0
            for i in range(num_waypoints):
                t = i * 0.5
                if t > sim_time:
                    break
                z_var = alt_amplitude * math.sin(2 * math.pi * t / alt_period)
                times.append(t)
                x_coords.append(base_x)
                y_coords.append(base_y)
                z_coords.append(altitude + z_var)

        if times:
            trajectories[uav_id] = {
                'times': times,
                'x': x_coords,
                'y': y_coords,
                'z': z_coords,
                'type': 'UAV',
                'pattern': pattern
            }

    return trajectories


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_mobility_trace(filename):
    """
    Load mobility trace data from ns-3 output file.
    Format: Time NodeType NodeID X Y Z
    """
    data = defaultdict(lambda: {'times': [], 'x': [], 'y': [], 'z': [], 'type': None})

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 6:
                # New format: Time NodeType NodeID X Y Z
                time = float(parts[0])
                node_type = parts[1]
                node_id = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            elif len(parts) >= 5:
                # Old format: Time NodeID X Y Z
                time = float(parts[0])
                node_type = 'UE'
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            else:
                continue

            key = (node_type, node_id)
            data[key]['times'].append(time)
            data[key]['x'].append(x)
            data[key]['y'].append(y)
            data[key]['z'].append(z)
            data[key]['type'] = node_type

    return data


def generate_demo_data(n_ues=63, n_uavs=7, sim_time=30.0):
    """
    Generate demonstration data without running ns-3.
    """
    data = {}

    # Scenario parameters (matching C++ defaults)
    center_x, center_y = 2000, 2000
    isd = 500  # Inter-site distance
    area_radius = isd * 1.5

    # LTE eNB at center (static)
    data[('LTE', 0)] = {
        'times': [t * 0.5 for t in range(int(sim_time / 0.5) + 1)],
        'x': [center_x] * (int(sim_time / 0.5) + 1),
        'y': [center_y] * (int(sim_time / 0.5) + 1),
        'z': [3.0] * (int(sim_time / 0.5) + 1),
        'type': 'LTE'
    }

    # UAV gNBs in hexagonal pattern around center
    uav_base_positions = []
    for i in range(n_uavs):
        angle = 2 * math.pi * i / n_uavs
        radius = isd * 0.8
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        uav_base_positions.append((x, y))

    # Generate UAV trajectories (circular orbit pattern)
    uav_trajectories = generate_uav_trajectories(
        uav_base_positions, sim_time=sim_time, pattern=1,
        altitude=100.0, orbit_radius=80.0
    )
    for uav_id, traj in uav_trajectories.items():
        data[('UAV', uav_id + 1)] = traj

    # Generate UE initial positions in grid
    ue_initial = {}
    grid_size = int(math.ceil(math.sqrt(n_ues)))
    spacing = (2.0 * area_radius) / grid_size
    for i in range(n_ues):
        x = center_x - area_radius + (i % grid_size) * spacing + spacing / 2.0
        y = center_y - area_radius + (i // grid_size % grid_size) * spacing + spacing / 2.0
        x = max(center_x - area_radius + 50, min(x, center_x + area_radius - 50))
        y = max(center_y - area_radius + 50, min(y, center_y + area_radius - 50))
        ue_initial[i + n_uavs + 1] = (x, y)

    # Generate UE trajectories
    ue_trajectories = generate_shanghai_trajectories(
        ue_initial, sim_time=sim_time,
        center_x=center_x, center_y=center_y, area_radius=area_radius
    )
    for ue_id, traj in ue_trajectories.items():
        data[('UE', ue_id)] = traj

    return data


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_static_2d(data, output_file=None):
    """Create a static 2D plot showing all trajectories."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color maps
    ue_colors = cm.Greens(np.linspace(0.3, 0.9, 20))
    uav_colors = cm.Reds(np.linspace(0.4, 0.9, 10))

    # Plot each node's trajectory
    for key, node_data in data.items():
        node_type, node_id = key
        x = node_data['x']
        y = node_data['y']

        if node_type == 'LTE':
            ax.scatter(x[0], y[0], c='blue', marker='s', s=300,
                      label='LTE eNB', zorder=10, edgecolors='darkblue', linewidths=2)
            ax.annotate('LTE\n(Centro)', (x[0], y[0]), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=10, fontweight='bold', color='blue')

        elif node_type == 'UAV':
            color = uav_colors[node_id % len(uav_colors)]
            # Plot trajectory
            ax.plot(x, y, '-', color=color, alpha=0.6, linewidth=2)
            # Start position
            ax.scatter(x[0], y[0], c='red', marker='^', s=200,
                      label='UAV gNB' if node_id == min(k[1] for k in data.keys() if k[0]=='UAV') else '',
                      zorder=9, edgecolors='darkred', linewidths=1.5)
            # End position
            ax.scatter(x[-1], y[-1], c='orange', marker='^', s=150, alpha=0.7,
                      zorder=9, edgecolors='darkorange', linewidths=1)
            ax.annotate(f'UAV{node_id}', (x[0], y[0]), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='darkred')

        elif node_type == 'UE':
            color = ue_colors[node_id % len(ue_colors)]
            # Plot trajectory (thin line)
            ax.plot(x, y, '-', color=color, alpha=0.4, linewidth=1)
            # Start position
            ax.scatter(x[0], y[0], c='green', marker='o', s=40, alpha=0.6,
                      label='UE' if node_id == min(k[1] for k in data.keys() if k[0]=='UE') else '',
                      zorder=5, edgecolors='darkgreen', linewidths=0.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('UAV-BS Scenario: Trajectory Overview', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved static plot to {output_file}")

    return fig, ax


def plot_static_3d(data, output_file=None):
    """Create a static 3D plot showing trajectories with altitude."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for key, node_data in data.items():
        node_type, node_id = key
        x = np.array(node_data['x'])
        y = np.array(node_data['y'])
        z = np.array(node_data['z'])

        if node_type == 'LTE':
            ax.scatter(x[0], y[0], z[0], c='blue', marker='s', s=200, label='LTE eNB', zorder=10)
            ax.text(x[0], y[0], z[0]+10, 'LTE', ha='center', fontsize=9, color='blue')

        elif node_type == 'UAV':
            color = plt.cm.Reds(0.5 + 0.5 * (node_id % 7) / 7)
            ax.plot(x, y, z, '-', color=color, alpha=0.7, linewidth=2)
            ax.scatter(x[0], y[0], z[0], c='red', marker='^', s=150,
                      label='UAV gNB' if node_id == min(k[1] for k in data.keys() if k[0]=='UAV') else '',
                      zorder=9)
            # Vertical line to ground
            ax.plot([x[0], x[0]], [y[0], y[0]], [0, z[0]], 'r--', alpha=0.3, linewidth=0.5)

        elif node_type == 'UE':
            color = plt.cm.Greens(0.3 + 0.5 * (node_id % 10) / 10)
            ax.plot(x, y, z, '-', color=color, alpha=0.4, linewidth=0.8)
            ax.scatter(x[0], y[0], z[0], c='green', marker='o', s=20, alpha=0.6,
                      label='UE' if node_id == min(k[1] for k in data.keys() if k[0]=='UE') else '',
                      zorder=5)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Altitude (m)', fontsize=11)
    ax.set_title('UAV-BS Scenario: 3D View', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to {output_file}")

    return fig, ax


def animate_scenario(data, output_file=None, interval=100, sim_time=30.0):
    """Create an animated visualization showing UAV and UE movement."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Check if UEs have insufficient data (less than sim_time worth of data)
    # Also detect if UEs are mostly static (not moving much)
    ue_initial_positions = {}
    incomplete_ues = []

    # Find the max time in the data
    max_data_time = 0
    for node_data in data.values():
        if node_data['times']:
            max_data_time = max(max_data_time, max(node_data['times']))

    # If we have less than 50% of requested sim_time, generate trajectories
    need_trajectory_generation = max_data_time < sim_time * 0.5

    for key, node_data in list(data.items()):
        node_type, node_id = key
        if node_type == 'UE':
            # Check if UE has insufficient time data or is mostly static
            max_ue_time = max(node_data['times']) if node_data['times'] else 0
            if max_ue_time < sim_time * 0.5 or len(node_data['times']) <= 1:
                ue_initial_positions[node_id] = (node_data['x'][0], node_data['y'][0])
                incomplete_ues.append(key)

    # Generate trajectories for UEs with incomplete data
    if ue_initial_positions and need_trajectory_generation:
        print(f"Generating Shanghai trajectories for {len(ue_initial_positions)} UEs (data only covers {max_data_time:.1f}s of {sim_time}s)...")
        # Find center from LTE position or estimate
        center_x, center_y = 2000, 2000
        for key, node_data in data.items():
            if key[0] == 'LTE':
                center_x = node_data['x'][0]
                center_y = node_data['y'][0]
                break

        shanghai_traj = generate_shanghai_trajectories(
            ue_initial_positions, sim_time=sim_time,
            center_x=center_x, center_y=center_y, area_radius=750
        )
        for key in incomplete_ues:
            node_id = key[1]
            if node_id in shanghai_traj:
                data[key] = shanghai_traj[node_id]

    # Also extend UAV trajectories if they're incomplete
    uav_initial_positions = []
    incomplete_uavs = []
    for key, node_data in list(data.items()):
        node_type, node_id = key
        if node_type == 'UAV':
            max_uav_time = max(node_data['times']) if node_data['times'] else 0
            if max_uav_time < sim_time * 0.5:
                uav_initial_positions.append((node_data['x'][0], node_data['y'][0]))
                incomplete_uavs.append(key)

    if uav_initial_positions and need_trajectory_generation:
        print(f"Generating UAV trajectories for {len(uav_initial_positions)} UAVs...")
        uav_traj = generate_uav_trajectories(
            uav_initial_positions, sim_time=sim_time, pattern=1,
            altitude=100.0, orbit_radius=80.0
        )
        for i, key in enumerate(incomplete_uavs):
            if i in uav_traj:
                # Preserve original type
                uav_traj[i]['type'] = 'UAV'
                data[key] = uav_traj[i]

    # Find time range
    all_times = set()
    for node_data in data.values():
        all_times.update(node_data['times'])
    times = sorted(all_times)

    # Limit frames for performance
    max_frames = 200
    if len(times) > max_frames:
        step = len(times) // max_frames
        times = times[::step]

    print(f"Creating animation with {len(times)} frames (time: {times[0]:.1f}s to {times[-1]:.1f}s)")

    # Get coordinate bounds
    all_x = [x for node_data in data.values() for x in node_data['x']]
    all_y = [y for node_data in data.values() for y in node_data['y']]
    margin = 100

    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('UAV-BS Scenario Animation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Initialize plot elements
    lte_scatter = ax.scatter([], [], c='blue', marker='s', s=300, label='LTE eNB',
                             zorder=10, edgecolors='darkblue', linewidths=2)
    uav_scatter = ax.scatter([], [], c='red', marker='^', s=200, label='UAV gNB',
                             zorder=9, edgecolors='darkred', linewidths=1.5)
    ue_scatter = ax.scatter([], [], c='olive', marker='o', s=40, label='UE',
                            zorder=5, alpha=0.8, edgecolors='darkgreen', linewidths=0.5)

    # UAV trails (full trajectory)
    uav_trails = {}
    for key in data.keys():
        if key[0] == 'UAV':
            line, = ax.plot([], [], 'r-', alpha=0.4, linewidth=2)
            uav_trails[key] = line

    # UE trails (last N positions)
    ue_trails = {}
    ue_trail_data = {}
    trail_length = 20
    ue_colors = cm.Greens(np.linspace(0.3, 0.9, 20))
    for key in data.keys():
        if key[0] == 'UE':
            node_id = key[1]
            color = ue_colors[node_id % len(ue_colors)]
            line, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=1.5)
            ue_trails[key] = line
            ue_trail_data[key] = []

    # Time display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Count display
    count_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.legend(loc='upper right', fontsize=10)

    def get_position_at_time(node_data, t):
        """Interpolate position at given time."""
        node_times = node_data['times']
        if t <= node_times[0]:
            return node_data['x'][0], node_data['y'][0]
        if t >= node_times[-1]:
            return node_data['x'][-1], node_data['y'][-1]

        for i in range(len(node_times) - 1):
            if node_times[i] <= t <= node_times[i+1]:
                if node_times[i+1] == node_times[i]:
                    return node_data['x'][i], node_data['y'][i]
                ratio = (t - node_times[i]) / (node_times[i+1] - node_times[i])
                x = node_data['x'][i] + ratio * (node_data['x'][i+1] - node_data['x'][i])
                y = node_data['y'][i] + ratio * (node_data['y'][i+1] - node_data['y'][i])
                return x, y
        return node_data['x'][-1], node_data['y'][-1]

    def init():
        lte_scatter.set_offsets(np.empty((0, 2)))
        uav_scatter.set_offsets(np.empty((0, 2)))
        ue_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        count_text.set_text('')
        for trail in uav_trails.values():
            trail.set_data([], [])
        for trail in ue_trails.values():
            trail.set_data([], [])
        for key in ue_trail_data:
            ue_trail_data[key] = []
        return [lte_scatter, uav_scatter, ue_scatter, time_text, count_text] + \
               list(uav_trails.values()) + list(ue_trails.values())

    def update(frame):
        t = times[frame]

        lte_pos = []
        uav_pos = []
        ue_pos = []

        for key, node_data in data.items():
            node_type, node_id = key
            x, y = get_position_at_time(node_data, t)

            if node_type == 'LTE':
                lte_pos.append([x, y])
            elif node_type == 'UAV':
                uav_pos.append([x, y])
                # Update UAV trail (full trajectory up to current time)
                trail_idx = [i for i, time in enumerate(node_data['times']) if time <= t]
                if trail_idx:
                    trail_x = [node_data['x'][i] for i in trail_idx]
                    trail_y = [node_data['y'][i] for i in trail_idx]
                    uav_trails[key].set_data(trail_x, trail_y)
            elif node_type == 'UE':
                ue_pos.append([x, y])
                # Update UE trail (last N positions)
                if key in ue_trail_data:
                    ue_trail_data[key].append((x, y))
                    if len(ue_trail_data[key]) > trail_length:
                        ue_trail_data[key].pop(0)
                    trail_x = [p[0] for p in ue_trail_data[key]]
                    trail_y = [p[1] for p in ue_trail_data[key]]
                    ue_trails[key].set_data(trail_x, trail_y)

        if lte_pos:
            lte_scatter.set_offsets(np.array(lte_pos))
        if uav_pos:
            uav_scatter.set_offsets(np.array(uav_pos))
        if ue_pos:
            ue_scatter.set_offsets(np.array(ue_pos))

        time_text.set_text(f'Time: {t:.1f} s')
        count_text.set_text(f'LTE: 1 | UAVs: {len(uav_pos)} | UEs: {len(ue_pos)}')

        return [lte_scatter, uav_scatter, ue_scatter, time_text, count_text] + \
               list(uav_trails.values()) + list(ue_trails.values())

    anim = FuncAnimation(fig, update, frames=len(times), init_func=init,
                        blit=True, interval=interval, repeat=True)

    if output_file:
        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=15)
        print(f"Animation saved!")

    return fig, anim


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize UAV-BS scenario mobility traces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot_scenario_uav.py mobility-trace.txt --animate --save
  python3 plot_scenario_uav.py --demo --animate --save
  python3 plot_scenario_uav.py mobility-trace.txt --3d --save
        """
    )
    parser.add_argument('trace_file', nargs='?', default='mobility-trace.txt',
                       help='Path to mobility trace file (default: mobility-trace.txt)')
    parser.add_argument('--animate', action='store_true',
                       help='Show animated visualization')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--3d', dest='plot_3d', action='store_true',
                       help='Show 3D plot with altitudes')
    parser.add_argument('--no-show', dest='no_show', action='store_true',
                       help='Do not display plots (useful with --save)')
    parser.add_argument('--sim-time', type=float, default=30.0,
                       help='Simulation time for generated trajectories (default: 30.0)')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo data without requiring trace file')
    parser.add_argument('--n-ues', type=int, default=63,
                       help='Number of UEs for demo mode (default: 63)')
    parser.add_argument('--n-uavs', type=int, default=7,
                       help='Number of UAVs for demo mode (default: 7)')

    args = parser.parse_args()

    # Load or generate data
    if args.demo:
        print(f"Generating demo data: {args.n_ues} UEs, {args.n_uavs} UAVs, {args.sim_time}s")
        data = generate_demo_data(n_ues=args.n_ues, n_uavs=args.n_uavs, sim_time=args.sim_time)
    else:
        print(f"Loading mobility trace from: {args.trace_file}")
        if not os.path.exists(args.trace_file):
            print(f"Error: File '{args.trace_file}' not found!")
            print("Run the simulation first or use --demo to generate demo data.")
            print("\nExample simulation command:")
            print("  ./ns3 run 'scenario-hierarchical-xangai-UAV --simTime=30 --enableGnbMobilityTrace=true'")
            sys.exit(1)
        data = load_mobility_trace(args.trace_file)

    # Count nodes by type
    node_counts = defaultdict(int)
    for (node_type, _) in data.keys():
        node_counts[node_type] += 1

    print(f"Loaded {len(data)} nodes:")
    for node_type, count in sorted(node_counts.items()):
        print(f"  - {node_type}: {count}")

    # Create plots
    if args.plot_3d:
        fig3d, ax3d = plot_static_3d(data, 'scenario_3d.png' if args.save else None)

    fig2d, ax2d = plot_static_2d(data, 'scenario_trajectories.png' if args.save else None)

    if args.animate:
        fig_anim, anim = animate_scenario(
            data,
            'scenario_animation.gif' if args.save else None,
            sim_time=args.sim_time
        )

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
