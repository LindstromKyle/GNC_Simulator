import numpy as np

from controller import PIDAttitudeController
from environment import Environment
from guidance import ModeBasedGuidance
from mission import (
    TimeBasedPhase,
    KickPhase,
    PitchProgramPhase,
    MissionPlanner,
    AscentBurnPhase,
    CoastPhase,
    CircBurnPhase,
)
from plotting import plot_3D_integration_segments
from simulator import Simulator
from state import State
from utils import compute_minimal_quaternion_rotation, rotate_vector_by_quaternion
from vehicle import Falcon9FirstStage, Falcon9SecondStage

# Stage 1
stage1_dry_mass = 25600
stage1_ascent_prop = 395700
stage1_reserve_prop = 30000  # Approx for returns burns
stage1_thrust = 7600000
stage1_avg_isp = 300
stage1_moi = np.diag([470297, 470297, 705445])
stage1_cd_base = 0.3
stage1_cd_scale = 0.2
stage1_area = 10.5
stage1_gimbal_limit = 10.0
stage1_gimbal_arm = 20.0
stage1_dry_com_z = 15
stage1_prop_com_z = 20

# Stage 2 params
stage2_dry_mass = 4000
stage2_prop = 111500
stage2_thrust = 934000
stage2_avg_isp = 348
stage2_moi = np.diag([10000, 10000, 20000])  # Approximate scaled
stage2_cd_base = 0.3
stage2_cd_scale = 2.0
stage2_area = 7.0  # Smaller
stage2_gimbal_limit = 5.0  # Vacuum engine
stage2_gimbal_arm = 2.0
stage2_dry_com_z = 3.0
stage2_prop_com_z = 6.0
separation_time = 162  # For now; later based on velocity/alt

# Combined vehicle for ascent
combined_dry_mass = stage1_dry_mass + stage1_reserve_prop + stage2_dry_mass + stage2_prop

# Vehicle
stage1_combined_vehicle = Falcon9FirstStage(
    dry_mass=combined_dry_mass,
    initial_prop_mass=stage1_ascent_prop,
    base_thrust_magnitude=stage1_thrust,
    average_isp=stage1_avg_isp,
    moment_of_inertia=stage1_moi + stage2_moi,  # Approx sum; improve later
    base_drag_coefficient=stage1_cd_base,
    drag_scaling_coefficient=stage1_cd_scale,
    cross_sectional_area=stage1_area,  # Use stage1 area for stack
    engine_gimbal_limit_deg=stage1_gimbal_limit,
    engine_gimbal_arm_len=stage1_gimbal_arm,
    dry_com_z=stage1_dry_com_z,
    prop_com_z=stage1_prop_com_z,
)

# Environment
environment = Environment()

# Launch site parameters
launch_latitude_deg = 28.5  # Cape Canaveral
launch_latitude_rad = np.deg2rad(launch_latitude_deg)

# Initial position (ECI frame, longitude=0 for simplicity)
cos_lat = np.cos(launch_latitude_rad)
sin_lat = np.sin(launch_latitude_rad)
initial_position = environment.earth_radius * np.array([cos_lat, 0.0, sin_lat])

# Initial velocity: due to Earth's rotation (at rest relative to ground)
omega_cross_r = np.cross(environment.earth_angular_velocity_vector, initial_position)

# Initial quaternion: align body Z with local vertical (radial unit vector)
radial_unit_vector = initial_position / np.linalg.norm(initial_position)
initial_quaternion = compute_minimal_quaternion_rotation(radial_unit_vector)
kick_direction = rotate_vector_by_quaternion(np.array([0, 1, 0]), initial_quaternion)

# State
initial_state = State(
    position=initial_position,
    velocity=omega_cross_r,
    quaternion=initial_quaternion,
    angular_velocity=[0, 0, 0],
    propellant_mass=stage1_ascent_prop,
)

# Set up phase timing
kick_start_time = 10.0
kick_end_time = 30
kick_angle = 5
burnout_time = 162.0

# Phases
stage1_phases = [
    TimeBasedPhase(end_time=kick_start_time, attitude_mode="radial", throttle=1.0, name="Initial Ascent"),
    KickPhase(
        end_time=kick_end_time,
        kick_direction=kick_direction,
        kick_angle_deg=kick_angle,
        throttle=1.0,
        name="Kick",
    ),
    PitchProgramPhase(
        end_time=burnout_time,
        initial_pitch_deg=90 - kick_angle,
        final_pitch_deg=20,
        kick_direction=kick_direction,
        throttle=1.0,
        name="Pitch Program",
    ),
]

# Mission Planner
stage1_planner = MissionPlanner(phases=stage1_phases, environment=environment, start_time=0.0)

# Guidance
stage1_guidance = ModeBasedGuidance()

# Controller
stage1_controller = PIDAttitudeController(
    kp=np.array([1e5, 1e5, 1e5]),
    ki=np.array([0.1, 0.1, 0.1]),
    kd=np.array([1e6, 1.0e6, 1.5e6]),
    guidance=stage1_guidance,
    vehicle=stage1_combined_vehicle,
)

# Simulator
stage1_sim = Simulator(
    vehicle=stage1_combined_vehicle,
    environment=environment,
    initial_state=initial_state,
    mission_planner=stage1_planner,
    t_0=0,
    t_final=162,
    delta_t=0.1,
    log_interval=1,
    log_name="orbit",
)
stage1_sim.add_controller(stage1_controller)

print(f"Simulating Ascent...")
stage1_t_vals, stage1_state_vals, stage1_phase_transitions = stage1_sim.run()

"""
STAGE 2
"""

# Separation
burnout_state_vector = stage1_state_vals[-1]
current_state = State(
    position=burnout_state_vector[:3],
    velocity=burnout_state_vector[3:6],
    quaternion=burnout_state_vector[6:10],
    angular_velocity=burnout_state_vector[10:13],
    propellant_mass=stage2_prop,
)

# Stage 2 Vehicle
stage_2 = Falcon9SecondStage(
    dry_mass=stage2_dry_mass,
    initial_prop_mass=stage2_prop,
    base_thrust_magnitude=stage2_thrust,
    average_isp=stage2_avg_isp,
    moment_of_inertia=stage2_moi,
    base_drag_coefficient=stage2_cd_base,
    drag_scaling_coefficient=stage2_cd_scale,
    cross_sectional_area=stage2_area,
    engine_gimbal_limit_deg=stage2_gimbal_limit,
    engine_gimbal_arm_len=stage2_gimbal_arm,
    dry_com_z=stage2_dry_com_z,
    prop_com_z=stage2_prop_com_z,
)

target_alt = 200000.0
target_apoapsis = target_alt + environment.earth_radius
# simulation_end_time = separation_time + 1200
simulation_end_time = 477
stage2_phases = [
    AscentBurnPhase(target_apoapsis=target_apoapsis, attitude_mode="prograde", throttle=1.0, name="Prograde Stage 2"),
    CoastPhase(time_to_apo_threshold=110.0, attitude_mode="prograde", throttle=0.0, name="Coast"),
    CircBurnPhase(peri_tolerance_factor=0.99, attitude_mode="prograde", throttle=1.0, name="Circularization"),
    TimeBasedPhase(end_time=simulation_end_time, attitude_mode="prograde", throttle=0.0, name="Orbit"),
]
stage2_planner = MissionPlanner(phases=stage2_phases, environment=environment, start_time=separation_time)

stage2_guidance = ModeBasedGuidance()

controller_stage2 = PIDAttitudeController(
    kp=np.array([5e3, 5e3, 5e3]),  # Tune for lighter stage
    ki=np.array([5, 5, 0]),
    kd=np.array([5e4, 5e4, 5e4]),
    guidance=stage2_guidance,
    vehicle=stage_2,
)

sim_stage2 = Simulator(
    vehicle=stage_2,
    environment=environment,
    initial_state=current_state,
    mission_planner=stage2_planner,
    t_0=separation_time,
    t_final=simulation_end_time,  # Enough for orbit
    delta_t=0.05,  # Larger step ok for vacuum
    log_interval=0.5,
    log_name="orbit",
)
sim_stage2.add_controller(controller_stage2)

print("\nSimulating Stage 2 to Orbit...")
stage2_t_vals, stage2_state_vals, stage2_phase_transitions = sim_stage2.run()

# Combine phase transitions for plotting
all_phase_transitions = [(t, f"{name}") for t, name in stage1_phase_transitions] + [
    (t, f"{name}") for t, name in stage2_phase_transitions
]

# Combine t and state vals for plotting
all_t_vals = np.append(stage1_t_vals, stage2_t_vals)
all_state_vals = np.vstack((stage1_state_vals, stage2_state_vals))

plot_3D_integration_segments(
    t_vals=all_t_vals,
    state_vals=all_state_vals,
    phase_transitions=all_phase_transitions,
    show_earth=False,
)
