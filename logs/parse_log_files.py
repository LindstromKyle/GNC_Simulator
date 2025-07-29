import numpy as np
import re
import matplotlib.pyplot as plt


def parse_log_to_structured_array(filename):
    data = []
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("[INFO] time (s):"):
            record = {}

            # Parse MISSION PLANNER
            mission_line = lines[i][6:].strip()  # Skip [INFO]
            parts = [p.strip() for p in mission_line.split("|")]
            record["time"] = float(re.search(r"time \(s\): ([-.\deE]+)", parts[0]).group(1))
            record["phase"] = re.search(r"phase: (.*)", parts[1]).group(1).strip()

            i += 1
            alt_line = lines[i][6:].strip()
            parts = [p.strip() for p in alt_line.split("|")]
            record["current_altitude"] = float(re.search(r"current altitude \(km\): ([-.\deE]+)", parts[0]).group(1))
            record["apoapsis_altitude"] = float(re.search(r"apoapsis altitude \(km\): ([-.\deE]+)", parts[1]).group(1))
            record["periapsis_altitude"] = float(
                re.search(r"periapsis altitude \(km\): ([-.\deE]+)", parts[2]).group(1)
            )

            i += 1
            vel_line = lines[i][6:].strip()
            parts = [p.strip() for p in vel_line.split("|")]
            record["orbital_vel"] = float(re.search(r"orbital vel \(km/s\): ([-.\deE]+)", parts[0]).group(1))
            record["tangential_vel"] = float(re.search(r"tangential vel \(km/s\): ([-.\deE]+)", parts[1]).group(1))
            record["radial_vel"] = float(re.search(r"radial vel \(km/s\): ([-.\deE]+)", parts[2]).group(1))

            # Skip to GUIDANCE
            while i < len(lines) and "GUIDANCE" not in lines[i]:
                i += 1
            i += 1  # Now at current quat line

            if i >= len(lines):
                break
            quat_line = lines[i][6:].strip()
            parts = [p.strip() for p in quat_line.split("|")]
            quat_str = re.search(r"current quat: \[(.*?)\]", parts[0]).group(1)
            record["current_quat"] = np.fromstring(quat_str, sep=" ")
            att_str = re.search(r"attitude: \[(.*?)\]", parts[1]).group(1)
            record["attitude"] = np.fromstring(att_str, sep=" ")

            i += 1
            des_quat_line = lines[i][6:].strip()
            des_quat_str = re.search(r"desired quat: \[(.*?)\]", des_quat_line).group(1)
            record["desired_quat"] = np.fromstring(des_quat_str, sep=" ")

            i += 1
            err_line = lines[i][6:].strip()
            parts = [p.strip() for p in err_line.split("|")]
            err_quat_str = re.search(r"error quat: \[(.*?)\]", parts[0]).group(1)
            record["error_quat"] = np.fromstring(err_quat_str, sep=" ")
            record["error_angle"] = float(re.search(r"error angle \(deg\): ([-.\deE]+)", parts[1]).group(1))

            # Skip to CONTROLLER
            while i < len(lines) and "CONTROLLER" not in lines[i]:
                i += 1
            i += 1  # Now at desired torque line

            if i >= len(lines):
                break
            torque_line = lines[i][6:].strip()
            parts = [p.strip() for p in torque_line.split("|")]
            des_torque_str = re.search(r"desired torque \(N\*m\): \[(.*?)\]", parts[0]).group(1)
            record["desired_torque"] = np.fromstring(des_torque_str, sep=" ")
            gimbal_str = re.search(r"engine gimbal angles: \[(.*?)\]", parts[1]).group(1)
            record["engine_gimbal_angles"] = np.fromstring(gimbal_str, sep=" ")
            record["throttle"] = float(re.search(r"throttle: ([-.\deE]+)", parts[2]).group(1))

            i += 1
            app_line = lines[i][6:].strip()
            parts = [p.strip() for p in app_line.split("|")]
            app_torque_str = re.search(r"applied torque \(N\*m\): \[(.*?)\]", parts[0]).group(1)
            record["applied_torque"] = np.fromstring(app_torque_str, sep=" ")
            ang_vel_str = re.search(r"ang vel \(rad/s\): \[(.*?)\]", parts[1]).group(1)
            record["ang_vel"] = np.fromstring(ang_vel_str, sep=" ")
            ang_acc_str = re.search(r"ang acc \(rad/s/s\): \[(.*?)\]", parts[2]).group(1)
            record["ang_acc"] = np.fromstring(ang_acc_str, sep=" ")

            # Skip to DYNAMICS
            while i < len(lines) and "DYNAMICS" not in lines[i]:
                i += 1
            i += 1  # Now at pos line

            if i >= len(lines):
                break
            pos_line = lines[i][6:].strip()
            parts = [p.strip() for p in pos_line.split("|")]
            pos_str = re.search(r"pos \(m\): \[(.*?)\]", parts[0]).group(1)
            record["pos"] = np.fromstring(pos_str, sep=" ")
            vel_str = re.search(r"vel \(m/s\): \[(.*?)\]", parts[1]).group(1)
            record["vel"] = np.fromstring(vel_str, sep=" ")
            acc_str = re.search(r"acc \(m/s/s\): \[(.*?)\]", parts[2]).group(1)
            record["acc"] = np.fromstring(acc_str, sep=" ")

            i += 1
            force_line = lines[i][6:].strip()
            parts = [p.strip() for p in force_line.split("|")]
            thrust_str = re.search(r"thrust \(N\): \[(.*?)\]", parts[0]).group(1)
            record["thrust"] = np.fromstring(thrust_str, sep=" ")
            drag_str = re.search(r"drag \(N\): \[(.*?)\]", parts[1]).group(1)
            record["drag"] = np.fromstring(drag_str, sep=" ")
            gravity_str = re.search(r"gravity \(N\): \[(.*?)\]", parts[2]).group(1)
            record["gravity"] = np.fromstring(gravity_str, sep=" ")
            net_str = re.search(r"net force \(N\): \[(.*?)\]", parts[3]).group(1)
            record["net_force"] = np.fromstring(net_str, sep=" ")

            i += 1
            mass_line = lines[i][6:].strip()
            parts = [p.strip() for p in mass_line.split("|")]
            record["total_mass"] = float(re.search(r"total mass \(kg\): ([-.\deE]+)", parts[0]).group(1))
            record["propellant_mass"] = float(re.search(r"propellant mass \(kg\): ([-.\deE]+)", parts[1]).group(1))
            record["mass_flow"] = float(re.search(r"mass flow \(kg/s\): ([-.\deE]+)", parts[2]).group(1))

            data.append(record)

        i += 1

    # Define dtype
    dtype = [
        ("time", np.float64),
        ("phase", "U50"),
        ("current_altitude", np.float64),
        ("apoapsis_altitude", np.float64),
        ("periapsis_altitude", np.float64),
        ("orbital_vel", np.float64),
        ("tangential_vel", np.float64),
        ("radial_vel", np.float64),
        ("current_quat", np.float64, (4,)),
        ("attitude", np.float64, (3,)),
        ("desired_quat", np.float64, (4,)),
        ("error_quat", np.float64, (4,)),
        ("error_angle", np.float64),
        ("desired_torque", np.float64, (3,)),
        ("engine_gimbal_angles", np.float64, (2,)),
        ("throttle", np.float64),
        ("applied_torque", np.float64, (3,)),
        ("ang_vel", np.float64, (3,)),
        ("ang_acc", np.float64, (3,)),
        ("pos", np.float64, (3,)),
        ("vel", np.float64, (3,)),
        ("acc", np.float64, (3,)),
        ("thrust", np.float64, (3,)),
        ("drag", np.float64, (3,)),
        ("gravity", np.float64, (3,)),
        ("net_force", np.float64, (3,)),
        ("total_mass", np.float64),
        ("propellant_mass", np.float64),
        ("mass_flow", np.float64),
    ]

    # Create structured array
    structured_data = np.zeros(len(data), dtype=dtype)
    for idx, rec in enumerate(data):
        for field in dtype:
            name = field[0]
            structured_data[name][idx] = rec[name]

    return structured_data


def plot_six(time, y_list, labels=None):
    """
    Plots six arrays in a 3x2 grid, with time on x-axis and y values on y-axis.
    All subplots share the x-axis.

    Parameters:
    - time: numpy array, the common x-axis values
    - y_list: list of 6 numpy arrays, the y-values for each plot
    - labels: optional list of 6 strings, labels for each subplot
    """
    if len(y_list) != 6:
        raise ValueError("y_list must contain exactly 6 arrays")
    if labels is None:
        labels = [f"Plot {i+1}" for i in range(6)]
    elif len(labels) != 6:
        raise ValueError("labels must contain exactly 6 strings")

    fig, axs = plt.subplots(3, 2, sharex=True)

    # Left column: first 3
    for i in range(3):
        axs[i, 0].plot(time, y_list[i])
        axs[i, 0].set_title(labels[i])
        axs[i, 0].set_xlabel("Time")
        axs[i, 0].tick_params(labelbottom=True)

    # Right column: next 3
    for i in range(3):
        axs[i, 1].plot(time, y_list[i + 3])
        axs[i, 1].set_title(labels[i + 3])
        axs[i, 1].set_xlabel("Time")
        axs[i, 1].tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    array = parse_log_to_structured_array("short_ascent.log")

    time = array["time"]
    y_list = [
        array["desired_torque"][:, 0],
        array["applied_torque"][:, 0],
        array["engine_gimbal_angles"][:, 0],
        array["error_angle"],
        array["ang_vel"][:, 0],
        array["ang_vel"][:, 1],
    ]
    labels = [
        "desired_torque (X)",
        "applied_torque (X)",
        "engine_gimbal_angles (X)",
        "error_angle",
        "ang_vel (X)",
        "ang_vel (Y)",
    ]
    plot_six(time, y_list, labels)
