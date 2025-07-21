import matplotlib.pyplot as plt
import numpy as np

from utils import compute_acceleration, quat_to_angle_axis


def plot_3D_trajectory(t_vals, state_vals):
    """

    Args:
        t_vals ():
        state_vals ():

    Returns:

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = state_vals[:, 0] / 1000
    y_vals = state_vals[:, 1] / 1000
    z_vals = state_vals[:, 2] / 1000

    ax.plot3D(
        x_vals,
        y_vals,
        z_vals,
        label="Rocket trajectory",
        linewidth=2,
        color="dodgerblue",
    )
    ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], color="green", label="Launch", s=50)
    ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], color="red", label="Final point", s=50)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("3D Rocket Trajectory")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_1D_position_velocity_acceleration(t_vals, state_vals, axis, environment):

    # Plot altitude vs time
    fig, axs = plt.subplots(3)

    position_dict = {"X": 0, "Y": 1, "Z": 2}
    velocity_dict = {"X": 3, "Y": 4, "Z": 5}

    altitude = state_vals[:, position_dict[axis]] - environment.earth_radius
    axs[0].plot(t_vals, altitude)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Altitude Above Surface (m)")
    axs[0].set_title(f"Altitude vs Time ({axis})")
    axs[0].grid()

    velocity = state_vals[:, velocity_dict[axis]]
    axs[1].plot(t_vals, velocity)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title(f"Velocity vs Time ({axis})")
    axs[1].grid()
    axs[1].sharex(axs[0])

    acceleration = compute_acceleration(t_vals, velocity)
    axs[2].plot(t_vals, acceleration)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s^2)")
    axs[2].set_title(f"Acceleration vs Time ({axis})")
    axs[2].grid()
    axs[2].sharex(axs[0])

    # drag_vals = np.array([
    #     np.linalg.norm(environment.drag_force(p, v, vehicle))
    #     for p, v in zip(state_vals[:, 2], state_vals[:, 5])
    # ])
    # axs[2].plot(t_vals, drag_vals)
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Acceleration (m/s^2)")
    # axs[2].set_title("Acceleration vs Time")
    # axs[2].grid()
    # axs[2].sharex(axs[0])

    plt.tight_layout()
    plt.show()


def plot_3D_trajectory_segments(segments: list[tuple[np.ndarray, np.ndarray]], show_earth: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if show_earth:
        # Add Earth's surface as a spherical mesh
        earth_radius_km = 6371  # Earth radius in km (consistent with trajectory scaling)
        u = np.linspace(0, 2 * np.pi, 50)  # Azimuthal angle; fewer points for performance
        v = np.linspace(0, np.pi, 25)  # Polar angle
        u, v = np.meshgrid(u, v)
        x = earth_radius_km * np.sin(v) * np.cos(u)
        y = earth_radius_km * np.sin(v) * np.sin(u)
        z = earth_radius_km * np.cos(v)
        ax.plot_surface(x, y, z, color="lightblue", alpha=0.4, zorder=0)  # Semi-transparent blue sphere

    colors = ["blue", "green", "orange"]  # Ascent, stage2, stage1
    labels = ["Ascent", "Stage 2 Orbital Insertion", "Stage 1 Return"]

    for i, (t_vals, state_vals) in enumerate(segments):
        x = state_vals[:, 0] / 1000
        y = state_vals[:, 1] / 1000
        z = state_vals[:, 2] / 1000
        ax.plot3D(x, y, z, color=colors[i], label=labels[i])

    # Add markers for separation, etc.
    ax.legend()
    ax.set_title("3D Trajectory - Staged Mission")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    plt.tight_layout()
    plt.show()


def plot_pitch_angle(t_vals, state_vals, guidance):

    fig, ax = plt.subplots()
    quats = state_vals[:, 6:10]
    pitches = []
    for quat in quats:
        angle_axis = quat_to_angle_axis(quat)
        angle = np.degrees(angle_axis[0])
        pitches.append(angle)
    ax.plot(t_vals, pitches)
    ax.axhline(guidance.kick_angle_deg, color="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Degrees)")
    ax.set_title("Pitch vs Time")
    ax.grid()
    plt.show()
