#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import rclpy
from rclpy.node import Node
from limo_interfaces.srv import GenerateControlPlot


# ─────────────────────────────────────────────
#  PALETTE — un colore per goal (ciclica)
# ─────────────────────────────────────────────

GOAL_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',
]

def goal_color(i: int) -> str:
    return GOAL_COLORS[i % len(GOAL_COLORS)]

def find_ros2_ws(start: Path):
    for parent in [start] + list(start.parents):
        if parent.name == 'ros2_ws':
            return parent
    return None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def build_time_axis(n_samples: int, goal_reach_times: list) -> np.ndarray:
    """Asse temporale uniforme [0, T] con T = ultimo goal_reach_time."""
    if goal_reach_times:
        T = goal_reach_times[-1]
    else:
        T = float(n_samples)
    return np.linspace(0.0, T, n_samples)


def shade_goals(ax, goal_reach_times: list, n_goals: int, alpha: float = 0.12):
    """Disegna bande verticali colorate per ogni sotto-traiettoria."""
    boundaries = [0.0] + list(goal_reach_times)
    for i in range(n_goals):
        x0 = boundaries[i]
        x1 = boundaries[i + 1] if i + 1 < len(boundaries) else boundaries[-1]
        ax.axvspan(x0, x1, color=goal_color(i), alpha=alpha)
        # linea verticale di separazione
        if i > 0:
            ax.axvline(x=x0, color=goal_color(i), linewidth=1.0,
                       linestyle='--', alpha=0.6)
        # etichetta goal centrata nella banda
        mid = (x0 + x1) / 2.0
        ax.text(mid, 1.01, f'G{i+1}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom',
                fontsize=7, color=goal_color(i), fontweight='bold')


def goal_legend(n_goals: int) -> list:
    return [
        mpatches.Patch(color=goal_color(i), alpha=0.5, label=f'Goal {i+1}')
        for i in range(n_goals)
    ]


def save_fig(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  PLOT FUNCTIONS
# ─────────────────────────────────────────────

def plot_distance_error(t, dist_errors, goal_reach_times, n_goals, output_dir) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))

    shade_goals(ax, goal_reach_times, n_goals)
    ax.plot(t, dist_errors, color='#2c7bb6', linewidth=1.2, label='Distance error [m]')
    ax.axhline(0.0, color='black', linewidth=0.6, linestyle=':')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance error [m]')
    ax.set_title('Distance Error vs Time')
    ax.legend(handles=[ax.lines[0]] + goal_legend(n_goals),
              fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return save_fig(fig, output_dir, '01_distance_error.jpg')


def plot_angular_error(t, ang_errors, goal_reach_times, n_goals, output_dir) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))

    shade_goals(ax, goal_reach_times, n_goals)
    ax.plot(t, np.degrees(ang_errors), color='#d7191c', linewidth=1.2,
            label='Angular error [deg]')
    ax.axhline(0.0, color='black', linewidth=0.6, linestyle=':')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular error [deg]')
    ax.set_title('Angular Error vs Time')
    ax.legend(handles=[ax.lines[0]] + goal_legend(n_goals),
              fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return save_fig(fig, output_dir, '02_angular_error.jpg')


def plot_velocities(t, lin_vels, ang_vels, goal_reach_times, n_goals, output_dir) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    shade_goals(ax1, goal_reach_times, n_goals)

    l1, = ax1.plot(t, lin_vels,  color='#1a9641', linewidth=1.2, label='v [m/s]')
    l2, = ax2.plot(t, np.degrees(ang_vels), color='#f46d43', linewidth=1.2,
                   linestyle='--', label='ω [deg/s]')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Linear velocity [m/s]',  color='#1a9641')
    ax2.set_ylabel('Angular velocity [deg/s]', color='#f46d43')
    ax1.tick_params(axis='y', labelcolor='#1a9641')
    ax2.tick_params(axis='y', labelcolor='#f46d43')

    ax1.set_title('Commanded Velocities vs Time')
    ax1.legend(handles=[l1, l2] + goal_legend(n_goals),
               fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    return save_fig(fig, output_dir, '03_velocities.jpg')


def plot_summary(t, dist_errors, ang_errors, lin_vels, ang_vels,
                 goal_reach_times, n_goals, output_dir) -> str:
    """4-panel summary in un unico jpg."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    titles  = ['Distance Error [m]', 'Angular Error [deg]', 'Velocities']
    colors  = ['#2c7bb6', '#d7191c', '#1a9641']
    data    = [
        dist_errors,
        np.degrees(ang_errors),
        lin_vels,
    ]

    for ax, title, color, y in zip(axes, titles, colors, data):
        shade_goals(ax, goal_reach_times, n_goals)
        ax.plot(t, y, color=color, linewidth=1.0)
        ax.axhline(0.0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.3)

    # sovrapponi omega sull'ultimo pannello (asse destro)
    ax_omega = axes[2].twinx()
    ax_omega.plot(t, np.degrees(ang_vels), color='#f46d43',
                  linewidth=1.0, linestyle='--', alpha=0.8)
    ax_omega.set_ylabel('ω [deg/s]', fontsize=8, color='#f46d43')
    ax_omega.tick_params(axis='y', labelcolor='#f46d43')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Mission Control Summary', fontsize=13, fontweight='bold')

    # legenda goal unica in alto
    fig.legend(handles=goal_legend(n_goals),
               loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 0.98))
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    return save_fig(fig, output_dir, '00_summary.jpg')


# ─────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────

class ControlPlotService(Node):

    def __init__(self):
        super().__init__('control_plot_service')

        ws = find_ros2_ws(Path(__file__).resolve())
        if ws is not None:
            self.output_dir = str(ws / 'src' / 'control_logs')
        else:
            self.output_dir = '/tmp/control_logs'
            self.get_logger().warn('ros2_ws non trovato, uso /tmp/control_logs')

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Plot output dir: {self.output_dir}')

        self.srv = self.create_service(
            GenerateControlPlot,
            '/mission/generate_control_plot',
            self._handle
        )
        self.get_logger().info('ControlPlotService ready on /mission/generate_control_plot')

    def _handle(self, request, response):
        try:
            dist_errors = np.array(request.distance_errors)
            ang_errors  = np.array(request.angular_errors)
            lin_vels    = np.array(request.linear_velocities)
            ang_vels    = np.array(request.angular_velocities)
            reach_times = list(request.goal_reach_times)
            n_goals     = request.n_goals
            out_dir     = request.output_dir or self.output_dir

            n = len(dist_errors)
            if n == 0:
                response.success = False
                response.message = 'Empty data arrays'
                return response

            t = build_time_axis(n, reach_times)

            paths = []
            paths.append(plot_summary(t, dist_errors, ang_errors,
                                       lin_vels, ang_vels,
                                       reach_times, n_goals, out_dir))
            paths.append(plot_distance_error(t, dist_errors,
                                              reach_times, n_goals, out_dir))
            paths.append(plot_angular_error(t, ang_errors,
                                             reach_times, n_goals, out_dir))
            paths.append(plot_velocities(t, lin_vels, ang_vels,
                                          reach_times, n_goals, out_dir))

            response.success      = True
            response.output_paths = paths
            response.message      = f'{len(paths)} plots saved to {out_dir}'
            self.get_logger().info(response.message)

        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Plot generation failed: {e}')

        return response


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ControlPlotService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()