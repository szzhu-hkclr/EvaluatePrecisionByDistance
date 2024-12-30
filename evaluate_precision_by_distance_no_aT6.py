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


def main(data_file):
    # Load data
    with open(data_file, 'r') as file:
        data = json.load(file)

    tracker_points = {p["name"]: [p["X"], p["Y"], p["Z"]] for p in data["tracker_points"]}
    ee_poses = {p["name"]: p["pose"][:3] for p in data["ee_poses"]}
    sensor_poses = {p["name"]: p["pose"][:3] for p in data["sensor_poses"]}

    # Ground truth distances (Method 1)
    a1 = calculate_distance(tracker_points["P1"], tracker_points["P2"])
    a2 = calculate_distance(tracker_points["P2"], tracker_points["P3"])
    a3 = calculate_distance(tracker_points["P3"], tracker_points["P4"])
    a4 = calculate_distance(tracker_points["P1"], tracker_points["P3"])
    a5 = calculate_distance(tracker_points["P2"], tracker_points["P4"])
    a6 = calculate_distance(tracker_points["P1"], tracker_points["P4"])

    # Method 3: Distances from wrist3_Link poses
    c1 = calculate_distance(ee_poses["P1"], ee_poses["P2"])
    c2 = calculate_distance(ee_poses["P2"], ee_poses["P3"])
    c3 = calculate_distance(ee_poses["P3"], ee_poses["P4"])
    c4 = calculate_distance(ee_poses["P1"], ee_poses["P3"])
    c5 = calculate_distance(ee_poses["P2"], ee_poses["P4"])
    c6 = calculate_distance(ee_poses["P1"], ee_poses["P4"])

    # Method 4: Distances from sensor poses
    d1 = calculate_distance(sensor_poses["P1"], sensor_poses["P2"])
    d2 = calculate_distance(sensor_poses["P2"], sensor_poses["P3"])
    d3 = calculate_distance(sensor_poses["P3"], sensor_poses["P4"])
    d4 = calculate_distance(sensor_poses["P1"], sensor_poses["P3"])
    d5 = calculate_distance(sensor_poses["P2"], sensor_poses["P4"])
    d6 = calculate_distance(sensor_poses["P1"], sensor_poses["P4"])

    # Print results
    method_label_3="Method 3 (nachi_kinematics)"
    method_label_4="Method 4 (handeye)"
    
    print(f"Ground Truth (Method 1): a1={a1}, a2={a2}, a3={a3}, a4={a4}, a5={a5}, a6={a6}")
    print(method_label_3 + f": c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}, c6={c6}")
    print(method_label_4 + f": d1={d1}, d2={d2}, d3={d3}, d4={d4}, d5={d5}, d6={d6}")

    # Error Calculations
    errors_method3 = [abs(a1 - c1), abs(a2 - c2), abs(a3 - c3), abs(a4 - c4), abs(a5 - c5), abs(a6 - c6)]
    errors_method4 = [abs(a1 - d1), abs(a2 - d2), abs(a3 - d3), abs(a4 - d4), abs(a5 - d5), abs(a6 - d6)]

    # Mean Errors
    mean_error_method3 = np.mean(errors_method3)
    mean_error_method4 = np.mean(errors_method4)

    # Print Mean Errors
    print("\nMean Errors:")
    print(method_label_3 + f": {mean_error_method3}")
    print(method_label_4 + f": {mean_error_method4}")

    # Create Table of Mean Errors
    methods = [method_label_3, method_label_4]
    mean_errors = [mean_error_method3, mean_error_method4]

    print("\nSummary Table:")
    print(f"{'Method':<25} {'Mean Error':<10}")
    for method, mean_error in zip(methods, mean_errors):
        print(f"{method:<25} {mean_error:<10.5f}")

    # Plot Errors
    labels = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x, errors_method3, width, label=method_label_3)
    ax.bar(x + width, errors_method4, width, label=method_label_4)

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
    main("data_16w.json")