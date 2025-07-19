import numpy as np
import logging
from dataclasses import asdict

from controller import PIDAttitudeController
from plotting import plot_3D_trajectory, plot_1D_position_velocity_acceleration
from utils import quat_to_angle_axis
from vehicle import Vehicle
from environment import Environment
from state import State
from integrator import integrate_rk4

logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,  # Use DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"  # Overwrite log file each run
)


class Simulator:
    def __init__(self, vehicle, environment, initial_state, t_0=0, t_final=2000, delta_t=0.5, log_interval=1):
        self.vehicle = vehicle
        self.environment = environment
        self.initial_state = initial_state
        self.t_0 = t_0
        self.t_final = t_final
        self.delta_t = delta_t
        self.log_interval = log_interval
        self.controller = None  # To be set later

    def add_controller(self, controller):
        self.controller = controller

    def run(self):
        print(f"Integrating...")
        t_vals, state_vals = integrate_rk4(vehicle=self.vehicle,
            environment=self.environment,
            initial_state=self.initial_state.as_vector(),
            t_0=self.t_0,
            t_final=self.t_final,
            delta_t=self.delta_t,
            log_interval=self.log_interval,
            controller=self.controller
        )

        return t_vals, state_vals

    def plot_1D(self, t_vals, state_vals, axis):
        # Plot 1D params
        plot_1D_position_velocity_acceleration(t_vals, state_vals, axis, self.environment)

    def plot_3D(self, t_vals, state_vals):
        # Plot 3D Trajectory
        plot_3D_trajectory(t_vals, state_vals)

if __name__ == "__main__":
    print(f"Initializing Vehicle...")
    vehicle = Vehicle(dry_mass=25600,
                      prop_mass=395700,
                      thrust_magnitude=7200000,
                      burn_time=162,
                      moment_of_inertia=np.diag([470297,470297,705445]),
                      base_drag_coefficient=0.3,
                      drag_scaling_coefficient=2.0,
                      cross_sectional_area=10.5,
                      engine_gimbal_limit=10.0,
                      engine_gimbal_arm=18.0
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
    # sim = Simulator(vehicle=vehicle,
    #                 environment=environment,
    #                 initial_state=initial_state,
    #                 t_0=0,
    #                 t_final=2000,
    #                 delta_t=0.5,
    #                 log_interval=1)
    sim = Simulator(vehicle=vehicle,
                    environment=environment,
                    initial_state=initial_state,
                    t_0=0,
                    t_final=161,
                    delta_t=0.1,
                    log_interval=0.5)

    controller = PIDAttitudeController(
        kp=np.array([5000.0, 5000.0,25000.0]),
        ki=np.zeros(3),
        kd=np.array([1e6, 1e6, 5e5]),
        desired_quaternion=np.array([0.9990, 0.0, 0.0436, 0.0]),  # e.g., ~5° pitch (use Rotation.as_quat)
        vehicle=sim.vehicle
    )

    sim.add_controller(controller)

    t_vals, state_vals = sim.run()

    sim.plot_1D(t_vals, state_vals, "Z")

    sim.plot_3D(t_vals, state_vals)

    # Plot yaw angle
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    quats = state_vals[:, 6:10]
    yaws = []
    for quat in quats:
        angle_axis = quat_to_angle_axis(quat)
        angle = np.degrees(angle_axis[0])
        yaws.append(angle)
    ax.plot(t_vals, yaws)
    ax.axhline(5, color='r')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw (Degrees)")
    ax.set_title("Yaw vs Time")
    ax.grid()
    plt.show()