import matplotlib.pyplot as plt

from utils import compute_acceleration


def plot_3D_trajectory(t_vals, state_vals):
    """

    Args:
        t_vals ():
        state_vals ():

    Returns:

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = state_vals[:, 0]
    y_vals = state_vals[:, 1]
    z_vals = state_vals[:, 2]

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

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
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
    axs[0].set_title("Altitude vs Time")
    axs[0].grid()

    velocity = state_vals[:, velocity_dict[axis]]
    axs[1].plot(t_vals, velocity)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity vs Time")
    axs[1].grid()
    axs[1].sharex(axs[0])

    acceleration = compute_acceleration(t_vals, velocity)
    axs[2].plot(t_vals, acceleration)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s^2)")
    axs[2].set_title("Acceleration vs Time")
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
