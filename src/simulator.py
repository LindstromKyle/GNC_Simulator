import numpy as np
import logging

from controller import PIDAttitudeController
from guidance import GravityTurnGuidance, OrbitalInsertionGuidance
from plotting import plot_3D_trajectory, plot_1D_position_velocity_acceleration, plot_3D_trajectory_segments
from vehicle import Vehicle
from environment import Environment
from state import State
from integrator import integrate_rk4

logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,  # Use DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",  # Overwrite log file each run
)


class Simulator:
    def __init__(
        self,
        vehicle,
        environment,
        initial_state,
        t_0=0,
        t_final=2000,
        delta_t=0.5,
        log_interval: float = 1,
    ):
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
        t_vals, state_vals = integrate_rk4(
            vehicle=self.vehicle,
            environment=self.environment,
            initial_state=self.initial_state.as_vector(),
            t_0=self.t_0,
            t_final=self.t_final,
            delta_t=self.delta_t,
            log_interval=self.log_interval,
            controller=self.controller,
        )

        return t_vals, state_vals

    def plot_1D(self, t_vals, state_vals, axis):
        # Plot 1D params
        plot_1D_position_velocity_acceleration(t_vals, state_vals, axis, self.environment)

    def plot_3D(self, t_vals, state_vals):
        # Plot 3D Trajectory
        plot_3D_trajectory(t_vals, state_vals)


if __name__ == "__main__":

    # Stage 1 params
    stage1_dry_mass = 25600
    stage1_ascent_prop = 395700
    stage1_reserve_prop = 30000  # Approx for returns burns
    stage1_thrust = 7200000
    stage1_burn_time = 162
    stage1_moi = np.diag([470297, 470297, 705445])
    stage1_cd_base = 0.3
    stage1_cd_scale = 0.2
    stage1_area = 10.5
    stage1_gimbal_limit = 10.0
    stage1_gimbal_arm = 3.0  # Your code has 18, but typical ~3-5m; adjust

    # Stage 2 params
    stage2_dry_mass = 4000
    stage2_prop = 111500
    stage2_thrust = 934000
    stage2_burn_time = 397
    stage2_moi = np.diag([10000, 10000, 20000])  # Approximate scaled
    stage2_cd_base = 0.3
    stage2_cd_scale = 2.0
    stage2_area = 7.0  # Smaller
    stage2_gimbal_limit = 5.0  # Vacuum engine
    stage2_gimbal_arm = 2.0
    separation_time = stage1_burn_time  # For now; later based on velocity/alt

    # Combined vehicle for ascent
    combined_dry_mass = stage1_dry_mass + stage1_reserve_prop + stage2_dry_mass + stage2_prop
    ascent_combined_stage = Vehicle(
        dry_mass=combined_dry_mass,
        prop_mass=stage1_ascent_prop,
        base_thrust_magnitude=stage1_thrust,
        burn_duration=stage1_burn_time,
        burn_start_time=0.0,
        moment_of_inertia=stage1_moi + stage2_moi,  # Approx sum; improve later
        base_drag_coefficient=stage1_cd_base,
        drag_scaling_coefficient=stage1_cd_scale,
        cross_sectional_area=stage1_area,  # Use stage1 area for stack
        engine_gimbal_limit_deg=stage1_gimbal_limit,
        engine_gimbal_arm_len=stage1_gimbal_arm,
    )

    environment = Environment()

    initial_state = State(
        position=[0, 0, environment.earth_radius],
        velocity=[0, 0, 0],
        quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0],
    )

    ascent_sim = Simulator(
        vehicle=ascent_combined_stage,
        environment=environment,
        initial_state=initial_state,
        t_0=0,
        t_final=separation_time,
        delta_t=0.1,
        log_interval=1,
    )

    ascent_guidance = GravityTurnGuidance(
        kick_start_time=25.0, kick_angle_deg=5.0, prograde_start_time=45.0, kick_direction=np.array([1.0, 0.0, 0.0])
    )

    ascent_controller = PIDAttitudeController(
        kp=np.array([8e4, 8e4, 3e5]),
        ki=np.array([1e-2, 1e-2, 1e-2]),
        kd=np.array([2e6, 2e6, 1e6]),
        guidance=ascent_guidance,
        vehicle=ascent_sim.vehicle,
    )

    ascent_sim.add_controller(ascent_controller)

    print(f"Simulating Ascent...")
    ascent_t_vals, ascent_state_vals = ascent_sim.run()

    # Separation
    burnout_state_vector = ascent_state_vals[-1]
    current_state = State(
        position=burnout_state_vector[:3],
        velocity=burnout_state_vector[3:6],
        quaternion=burnout_state_vector[6:10],
        angular_velocity=burnout_state_vector[10:13],
    )

    # Stage 2 sim
    stage_2 = Vehicle(
        dry_mass=stage2_dry_mass,
        prop_mass=stage2_prop,
        base_thrust_magnitude=stage2_thrust,
        burn_duration=stage2_burn_time,
        moment_of_inertia=stage2_moi,
        base_drag_coefficient=stage2_cd_base,
        drag_scaling_coefficient=stage2_cd_scale,
        cross_sectional_area=stage2_area,
        engine_gimbal_limit_deg=stage2_gimbal_limit,
        engine_gimbal_arm_len=stage2_gimbal_arm,
        burn_start_time=separation_time,  # Ignite immediately
    )

    guidance_stage2 = OrbitalInsertionGuidance(target_apoapsis=200000.0)

    controller_stage2 = PIDAttitudeController(
        kp=np.array([5e4, 5e4, 2e5]),  # Tune for lighter stage
        ki=np.array([1e-3, 1e-3, 1e-3]),
        kd=np.array([1e6, 1e6, 5e5]),
        guidance=guidance_stage2,
        vehicle=stage_2,
    )

    sim_stage2 = Simulator(
        vehicle=stage_2,
        environment=environment,
        initial_state=current_state,
        t_0=separation_time,
        t_final=separation_time + 600,  # Enough for orbit
        delta_t=0.2,  # Larger step ok for vacuum
        log_interval=5,
    )
    sim_stage2.add_controller(controller_stage2)

    print("\nSimulating Stage 2 to Orbit...")
    stage2_t_vals, stage2_state_vals = sim_stage2.run()

    plot_3D_trajectory_segments(
        [(ascent_t_vals, ascent_state_vals), (stage2_t_vals, stage2_state_vals)], show_earth=False
    )
