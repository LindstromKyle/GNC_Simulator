from dynamics import calculate_dynamics
import numpy as np

def integrate_rk4(vehicle, environment, initial_state, t_0, t_final, delta_t, log_interval):

    # Initialize starting time and position
    t_values = [t_0]
    state_values = [initial_state]
    current_time = t_0
    current_state = initial_state.copy()
    last_logged_time = None


    while current_time < t_final:

        # Log dynamic variables every [log_interval] seconds of simulation time
        log_flag = False
        if last_logged_time is None or round(current_time - last_logged_time, 12) >= log_interval:
            log_flag = True
            last_logged_time = current_time

        # Calculate step size, ensure we don’t overshoot
        h = min(delta_t, t_final - current_time)

        # Calculate intermediate slopes, logging k_1 every [log_interval] seconds
        k_1 = calculate_dynamics(current_time, current_state, vehicle, environment, log=log_flag)
        k_2 = calculate_dynamics(current_time + h / 2, current_state + k_1 / 2, vehicle, environment, log=False)
        k_3 = calculate_dynamics(current_time + h / 2, current_state + k_2 / 2, vehicle, environment, log=False)
        k_4 = calculate_dynamics(current_time + h, current_state + k_3, vehicle, environment, log=False)

        # Update state and time
        weighted_average = (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
        current_state += h * weighted_average
        current_time += h

        # Append state and time to results
        t_values.append(current_time)
        state_values.append(current_state.copy())

    return np.array(t_values), np.array(state_values)
