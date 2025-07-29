import logging
from abc import ABC, abstractmethod
import numpy as np

from utils import compute_orbital_elements, compute_time_to_apoapsis


class Phase(ABC):
    @abstractmethod
    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        pass

    @abstractmethod
    def get_setpoints(self) -> dict:
        pass


class TimeBasedPhase(Phase):
    def __init__(self, end_time: float, attitude_mode: str, throttle: float = 1.0, name: str = "Unnamed"):
        self.end_time = end_time
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class KickPhase(Phase):
    def __init__(
        self,
        end_time: float,
        kick_direction: np.ndarray,
        kick_angle_deg: float,
        throttle: float = 1.0,
        name: str = "Unnamed",
    ):
        self.end_time = end_time
        self.attitude_mode = "kick"
        self.throttle = throttle
        self.kick_direction = kick_direction
        self.kick_angle_deg = kick_angle_deg
        self.kick_angle_rad = np.deg2rad(self.kick_angle_deg)
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "kick_direction": self.kick_direction,
            "kick_angle_rad": self.kick_angle_rad,
        }


class AscentBurnPhase(Phase):
    def __init__(
        self, target_apoapsis: float, attitude_mode: str = "prograde", throttle: float = 1.0, name: str = "Unnamed"
    ):
        self.target_apoapsis = target_apoapsis
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return (elements is not None) and (elements["apoapsis_radius"] >= self.target_apoapsis)

    def get_setpoints(self) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class CoastPhase(Phase):
    def __init__(
        self,
        time_to_apo_threshold: float,
        attitude_mode: str = "prograde",
        throttle: float = 0.0,
        name: str = "Unnamed",
    ):
        self.time_to_apo_threshold = time_to_apo_threshold
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        if elements is None:
            return False
        time_to_apo = compute_time_to_apoapsis(
            state_vector[:3], state_vector[3:6], elements, self.mu
        )  # mu from planner
        return time_to_apo <= self.time_to_apo_threshold

    def get_setpoints(self) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class CircBurnPhase(Phase):
    def __init__(
        self,
        peri_tolerance_factor: float,
        attitude_mode: str = "prograde",
        throttle: float = 1.0,
        name: str = "Unnamed",
    ):
        self.peri_tolerance_factor = peri_tolerance_factor
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:

        return (elements is not None) and (
            elements["periapsis_radius"] >= elements["apoapsis_radius"] * self.peri_tolerance_factor
        )

    def get_setpoints(self) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class PitchProgramPhase(Phase):
    def __init__(
        self,
        end_time: float,
        initial_pitch_deg: float,
        final_pitch_deg: float,
        kick_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),  # Default eastward
        throttle: float = 1.0,
        name: str = "Pitch Program",
    ):
        self.end_time = end_time
        self.initial_pitch_deg = initial_pitch_deg
        self.final_pitch_deg = final_pitch_deg
        self.kick_direction = kick_direction
        self.throttle = throttle
        self.name = name
        self.attitude_mode = "programmed_pitch"  # Custom mode for guidance

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return time >= self.end_time

    def get_setpoints(self) -> dict:
        return {
            "throttle": self.throttle,
            "attitude_mode": self.attitude_mode,
            "initial_pitch_deg": self.initial_pitch_deg,
            "final_pitch_deg": self.final_pitch_deg,
            "kick_direction": self.kick_direction,
            # Note: start_time and duration are added dynamically by MissionPlanner
        }


class MissionPlanner:
    def __init__(self, phases: list[Phase], environment, start_time: float = 0.0):
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase = phases[0]
        self.mu = environment.gravitational_constant * environment.earth_mass
        self.environment = environment
        self.phase_transitions = [(start_time, phases[0].name)]
        self.phase_start_times = [start_time] * len(phases)  # Initialize list for each phase's start time
        self.phase_start_times[0] = start_time  # Set initial
        # Inject mu to phases if needed (for CoastPhase)
        for phase in self.phases:
            if hasattr(phase, "time_to_apo_threshold"):  # Hack for CoastPhase
                phase.mu = self.mu

    def update(self, time: float, state_vector: np.ndarray, log_flag: bool) -> dict:
        position = state_vector[0:3]
        velocity = state_vector[3:6]
        elements = compute_orbital_elements(position, velocity, self.mu)
        altitude = (np.linalg.norm(position) - self.environment.earth_radius) / 1000

        if log_flag:
            # Full orbital velocity (magnitude)
            r_unit_vector = position / np.linalg.norm(position)
            orbital_velocity = np.linalg.norm(velocity)
            # Radial velocity
            radial_velocity = np.dot(velocity, r_unit_vector)
            # Tangential velocity
            tangential_velocity = (
                np.sqrt(orbital_velocity**2 - radial_velocity**2)
                if orbital_velocity**2 - radial_velocity**2 > 0.0
                else 0.0
            )
            logging.info(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logging.info(f"---------------------------------[MISSION PLANNER]----------------------------------------")
            logging.info(f"time (s): {time:.2f} | phase: {self.current_phase.name}")
            logging.info(
                f"current altitude (km): {altitude:.4f} | "
                f"apoapsis altitude (km): {((elements["apoapsis_radius"] - self.environment.earth_radius) / 1000):.4f} | "
                f"periapsis altitude (km): {((elements["periapsis_radius"] - self.environment.earth_radius) / 1000):.4f}"
            )
            logging.info(
                f"orbital vel (km/s): {orbital_velocity/1000:.4f} | tangential vel (km/s): {tangential_velocity/1000:.4f} | radial vel (km/s): {radial_velocity/1000:.4f}"
            )

        self.current_phase = self.phases[self.current_phase_idx]
        if self.current_phase.is_complete(time, state_vector, elements):
            self.current_phase_idx += 1
            if self.current_phase_idx < len(self.phases):
                logging.info(f"Phase transition to {self.phases[self.current_phase_idx].name} at t={time:.2f}")
                self.phase_transitions.append((time, self.phases[self.current_phase_idx].name))
                # Update start time for the new phase
                self.phase_start_times[self.current_phase_idx] = time
            else:
                logging.info(f"Integration segment complete at t={time:.2f}")
                return {"throttle": 0.0, "attitude_mode": "prograde"}
            self.current_phase = self.phases[self.current_phase_idx]

        # Get base setpoints from phase
        setpoints = self.current_phase.get_setpoints()

        # Dynamically add start_time and duration if the phase supports it (e.g., for PitchProgramPhase)
        if setpoints.get("attitude_mode") == "programmed_pitch":
            start_time = self.phase_start_times[self.current_phase_idx]
            duration = self.current_phase.end_time - start_time
            setpoints["start_time"] = start_time
            setpoints["duration"] = duration

        return setpoints

    def get_phase_transitions(self) -> list[tuple[float, str]]:
        return self.phase_transitions
