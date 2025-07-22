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
        self, target_ra: float, attitude_mode: str = "prograde", throttle: float = 1.0, name: str = "Unnamed"
    ):
        self.target_ra = target_ra
        self.attitude_mode = attitude_mode
        self.throttle = throttle
        self.name = name

    def is_complete(self, time: float, state_vector: np.ndarray, elements: dict | None) -> bool:
        return elements is not None and elements["ra"] >= self.target_ra

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
        return elements is not None and elements["rp"] >= elements["ra"] * self.peri_tolerance_factor

    def get_setpoints(self) -> dict:
        return {"throttle": self.throttle, "attitude_mode": self.attitude_mode}


class MissionPlanner:
    def __init__(self, phases: list[Phase], environment, start_time: float = 0.0):
        self.phases = phases
        self.current_phase_idx = 0
        self.mu = environment.gravitational_constant * environment.earth_mass
        self.phase_transitions = [(start_time, phases[0].name)]
        # Inject mu to phases if needed (for CoastPhase)
        for phase in self.phases:
            if hasattr(phase, "time_to_apo_threshold"):  # Hack for CoastPhase
                phase.mu = self.mu

    def update(self, time: float, state_vector: np.ndarray) -> dict:
        elements = compute_orbital_elements(state_vector[:3], state_vector[3:6], self.mu)
        current_phase = self.phases[self.current_phase_idx]
        if current_phase.is_complete(time, state_vector, elements):
            self.current_phase_idx += 1
            if self.current_phase_idx < len(self.phases):
                logging.info(f"Phase transition to {self.phases[self.current_phase_idx].name} at t={time:.2f}")
                self.phase_transitions.append((time, self.phases[self.current_phase_idx].name))
            else:
                logging.info(f"Mission complete at t={time:.2f}")
                return {"throttle": 0.0, "attitude_mode": "prograde"}
            current_phase = self.phases[self.current_phase_idx]
        return current_phase.get_setpoints()

    def get_phase_transitions(self) -> list[tuple[float, str]]:
        return self.phase_transitions
