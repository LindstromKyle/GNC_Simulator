import numpy as np
import matplotlib.pyplot as plt

from vehicle import Vehicle
from environment import Environment
from state import State
from integrator import integrate_rk4
from utils import compute_acceleration

import logging

logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,  # Use DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"  # Overwrite log file each run
)

def run_simulator():

    print(f"Initializing Vehicle...")
    vehicle = Vehicle(dry_mass=50,
                      prop_mass=100,
                      thrust_magnitude=2000,
                      burn_time=10,
                      moment_of_inertia=np.diag([100, 100, 10]),
                      base_drag_coefficient=0.2,
                      drag_scaling_coefficient=0.8,
                      cross_sectional_area=0.1,
                      )

    print(f"Initializing Environment...")
    environment = Environment()

    print(f"Initializing State...")
    # Make sure to initialize with ECI coordinates
    initial_state = State(
        position=[0, 0, environment.earth_radius],
        velocity=[0, 0, 0],
        quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0]
    )

    print(f"Integrating...")
    # Integrate with RK4
    t_vals, state_vals = integrate_rk4(
        vehicle=vehicle,
        environment=environment,
        initial_state=initial_state.as_vector(),
        t_0=0,
        t_final=30,
        delta_t=0.05,
        log_interval=0.5
    )

    # Plot altitude vs time
    fig, axs = plt.subplots(3)

    altitude_z = state_vals[:, 2] - environment.earth_radius
    axs[0].plot(t_vals, altitude_z)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Altitude Above Surface (m)")
    axs[0].set_title("Altitude vs Time")
    axs[0].grid()

    velocity_z = state_vals[:, 5]
    axs[1].plot(t_vals, velocity_z)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity vs Time")
    axs[1].grid()
    axs[1].sharex(axs[0])

    acceleration = compute_acceleration(t_vals, velocity_z)
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


if __name__ == "__main__":
    run_simulator()
