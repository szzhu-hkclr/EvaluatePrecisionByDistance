import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    r = R.from_quat([q[0], q[1], q[2], q[3]])  # Scalar-last order
    return r.as_matrix()


def calculate_distance(pose1, pose2):
    """Calculate Euclidean distance between two poses."""
    return np.linalg.norm(np.array(pose1) - np.array(pose2))


def calculate_aT6(aT3, link_transform):
    """Calculate aT6 using aT3 and the link transform."""
    r = quaternion_to_rotation_matrix(link_transform["Rotation"])
    t = np.array(link_transform["Translation"]).reshape(3, 1)
    link3ToLink6 = np.vstack((np.hstack((r, t)), [0, 0, 0, 1]))
    return np.dot(aT3, link3ToLink6)


def load_aT3_results(aT3_json_file):
    """Load aT3 results from the provided JSON file."""
    with open(aT3_json_file, 'r') as file:
        data = json.load(file)

    aT3_results = []
    for entry in data["aT3_mats"]:
        aT3_results.append(np.array(entry["aT3"]))
    return aT3_results


def main(data_file, aT3_json_file):
    # Load data
    with open(data_file, 'r') as file:
        data = json.load(file)

    tracker_points = {p["name"]: [p["X"], p["Y"], p["Z"]] for p in data["tracker_points"]}
    link_transforms = {t["name"]: t for t in data["link_transforms"]}
    wrist3_Link_poses = {p["name"]: p["pose"][:3] for p in data["wrist3_Link_poses"]}
    sensor_poses = {p["name"]: p["pose"][:3] for p in data["sensor_poses"]}

    # Load aT3 results from JSON
    aT3_results = load_aT3_results(aT3_json_file)

    # Ground truth distances (Method 1)
    a1 = calculate_distance(tracker_points["P1"], tracker_points["P6"])
    a2 = calculate_distance(tracker_points["P6"], tracker_points["P11"])
    a3 = calculate_distance(tracker_points["P1"], tracker_points["P11"])

    # Method 2: Calculate aT6 for P1, P6, P11 and compute distances
    aT3_P1, aT3_P6, aT3_P11 = aT3_results
    aT6_P1 = calculate_aT6(aT3_P1, link_transforms["P1"])
    aT6_P6 = calculate_aT6(aT3_P6, link_transforms["P6"])
    aT6_P11 = calculate_aT6(aT3_P11, link_transforms["P11"])

    b1 = calculate_distance(aT6_P1[:3, 3], aT6_P6[:3, 3])
    b2 = calculate_distance(aT6_P6[:3, 3], aT6_P11[:3, 3])
    b3 = calculate_distance(aT6_P1[:3, 3], aT6_P11[:3, 3])

    # Method 3: Distances from wrist3_Link poses
    c1 = calculate_distance(wrist3_Link_poses["P1"], wrist3_Link_poses["P6"])
    c2 = calculate_distance(wrist3_Link_poses["P6"], wrist3_Link_poses["P11"])
    c3 = calculate_distance(wrist3_Link_poses["P1"], wrist3_Link_poses["P11"])

    # Method 4: Distances from sensor poses
    d1 = calculate_distance(sensor_poses["P1"], sensor_poses["P6"])
    d2 = calculate_distance(sensor_poses["P6"], sensor_poses["P11"])
    d3 = calculate_distance(sensor_poses["P1"], sensor_poses["P11"])

    # Print results
    print(f"Ground Truth (Method 1): a1={a1}, a2={a2}, a3={a3}")
    print(f"Method 2 (aT6): b1={b1}, b2={b2}, b3={b3}")
    print(f"Method 3 (wrist3_Link): c1={c1}, c2={c2}, c3={c3}")
    print(f"Method 4 (sensor): d1={d1}, d2={d2}, d3={d3}")

    # Error Calculations
    errors_method2 = [abs(a1 - b1), abs(a2 - b2), abs(a3 - b3)]
    errors_method3 = [abs(a1 - c1), abs(a2 - c2), abs(a3 - c3)]
    errors_method4 = [abs(a1 - d1), abs(a2 - d2), abs(a3 - d3)]

    # Mean Errors
    mean_error_method2 = np.mean(errors_method2)
    mean_error_method3 = np.mean(errors_method3)
    mean_error_method4 = np.mean(errors_method4)

    # Print Mean Errors
    print("\nMean Errors:")
    print(f"Method 2 (aT6): {mean_error_method2}")
    print(f"Method 3 (wrist3_Link): {mean_error_method3}")
    print(f"Method 4 (sensor): {mean_error_method4}")

    # Create Table of Mean Errors
    methods = ["Method 2 (aT6)", "Method 3 (wrist3_Link)", "Method 4 (sensor)"]
    mean_errors = [mean_error_method2, mean_error_method3, mean_error_method4]

    print("\nSummary Table:")
    print(f"{'Method':<25} {'Mean Error':<10}")
    for method, mean_error in zip(methods, mean_errors):
        print(f"{method:<25} {mean_error:<10.5f}")

    # Plot Errors
    labels = ['d1', 'd2', 'd3']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, errors_method2, width, label="Method 2 (aT6)")
    ax.bar(x, errors_method3, width, label="Method 3 (wrist3_Link)")
    ax.bar(x + width, errors_method4, width, label="Method 4 (sensor)")

    # Add labels, title, legend
    ax.set_xlabel("Distances")
    ax.set_ylabel("Error")
    ax.set_title("Comparison of Errors Across Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage:
    # Pass the `aT3_results.json` file and the main data file.
    main("data.json", "aT3_results.json")