import numpy as np

from controller import PIDAttitudeController
from environment import Environment
from guidance import ModeBasedGuidance
from mission import TimeBasedPhase, KickPhase, MissionPlanner, PitchProgramPhase
from plotting import plot_3D_integration_segments
from simulator import Simulator
from state import State
from utils import compute_minimal_quaternion_rotation, rotate_vector_by_quaternion
from vehicle import Falcon9FirstStage

"""Simulate short ascent of rocket carrying first and second stage""" ""

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

# Stage 2 params
stage2_dry_mass = 4000
stage2_prop = 111500
stage2_moi = np.diag([10000, 10000, 20000])  # Approximate scaled
separation_time = 162  # For now; later based on velocity/alt

# Combined vehicle for ascent
combined_dry_mass = stage1_dry_mass + stage1_reserve_prop + stage2_dry_mass + stage2_prop

# Vehicle
ascent_combined_stage = Falcon9FirstStage(
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
    dry_com_z=15,
    prop_com_z=20,
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
ascent_phases = [
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
        final_pitch_deg=45,
        kick_direction=kick_direction,
        throttle=1.0,
        name="Pitch Program",
    ),
]

# Mission Planner
ascent_planner = MissionPlanner(phases=ascent_phases, environment=environment, start_time=0.0)

# Guidance
ascent_guidance = ModeBasedGuidance()

# Controller
ascent_controller = PIDAttitudeController(
    kp=np.array([1e5, 1e5, 1e5]),
    ki=np.array([0.1, 0.1, 0.1]),
    kd=np.array([1e6, 1.0e6, 1.5e6]),
    guidance=ascent_guidance,
    vehicle=ascent_combined_stage,
)

# Simulator
ascent_sim = Simulator(
    vehicle=ascent_combined_stage,
    environment=environment,
    initial_state=initial_state,
    mission_planner=ascent_planner,
    t_0=0,
    t_final=162,
    delta_t=0.1,
    log_interval=0.1,
    log_name="short_ascent",
)
ascent_sim.add_controller(ascent_controller)

print(f"Simulating Ascent...")
ascent_t_vals, ascent_state_vals, ascent_phase_transitions = ascent_sim.run()

plot_3D_integration_segments(
    t_vals=ascent_t_vals,
    state_vals=ascent_state_vals,
    phase_transitions=ascent_phase_transitions,
    show_earth=False,
)
