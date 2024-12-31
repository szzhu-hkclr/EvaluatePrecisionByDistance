import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt


def calculate_distance(pose1, pose2):
    """Calculate Euclidean distance between two poses."""
    return np.linalg.norm(np.array(pose1) - np.array(pose2))


def main(data_file):
    # Load data
    with open(data_file, 'r') as file:
        data = json.load(file)

    tracker_points = {p["name"]: p["pose"][:3] for p in data["tracker_points"]}
    # ee_poses = {p["name"]: p["pose"][:3] for p in data["ee_poses"]}
    # sensor_poses = {p["name"]: p["pose"][:3] for p in data["sensor_poses"]}

    # The distance of 2 nearby points in the robot task plan is 500mm
    # Evaluate precision by comparison robot planned distance and Euclidean distance of 2 nearby points
    p1 = calculate_distance(tracker_points["P1"], tracker_points["P2"])
    p2 = calculate_distance(tracker_points["P2"], tracker_points["P3"])
    p3 = calculate_distance(tracker_points["P3"], tracker_points["P4"])
    p4 = calculate_distance(tracker_points["P4"], tracker_points["P5"])
    p5 = calculate_distance(tracker_points["P5"], tracker_points["P6"])
    p6 = calculate_distance(tracker_points["P6"], tracker_points["P7"])
    p7 = calculate_distance(tracker_points["P7"], tracker_points["P8"])
    p8 = calculate_distance(tracker_points["P8"], tracker_points["P9"])
    p9 = calculate_distance(tracker_points["P9"], tracker_points["P10"])
    p10 = calculate_distance(tracker_points["P10"], tracker_points["P11"])

    # Evaluate repeatability by comparison 2 odd & even points position differences
    o1 = calculate_distance(tracker_points["P1"], tracker_points["P3"])
    o2 = calculate_distance(tracker_points["P3"], tracker_points["P5"])
    o3 = calculate_distance(tracker_points["P5"], tracker_points["P7"])
    o4 = calculate_distance(tracker_points["P7"], tracker_points["P9"])
    o5 = calculate_distance(tracker_points["P9"], tracker_points["P11"])
    e1 = calculate_distance(tracker_points["P2"], tracker_points["P4"])
    e2 = calculate_distance(tracker_points["P4"], tracker_points["P6"])
    e3 = calculate_distance(tracker_points["P6"], tracker_points["P8"])
    e4 = calculate_distance(tracker_points["P8"], tracker_points["P10"])
    e5 = calculate_distance(tracker_points["P10"], tracker_points["P12"])

    # Print results
    precision_label="Calculate Precision"
    repeatability_label="Compute Repeatability"
    
    print(f"Robot Task Planned Distance = 500mm")
    print(precision_label + f"[Unit/mm]:\n p1={p1-500}\n p2={p2-500}\n p3={p3-500}\n p4={p4-500}\n p5={p5-500}\n p6={p6-500}\n p7={p7-500}\n p8={p8-500}\n p9={p9-500}\n p10={p10-500}")
    print(repeatability_label + f"[Unit/mm]:\n o1={o1}\n o2={o2}\n o3={o3}\n o4={o4}\n o5={o5}\n e1={e1}\n e2={e2}\n e3={e3}\n e4={e4}\n e5={e5}")

    # Error Calculations& Struct
    errors_precision = [abs(p1 - 500), abs(p2 - 500), abs(p3 - 500), abs(p4 - 500), abs(p5 - 500), abs(p6 - 500), abs(p7 - 500), abs(p8 - 500), abs(p9 - 500), abs(p10 - 500)]
    errors_repeatability = [abs(o1), abs(o2), abs(o3), abs(o4), abs(o5), abs(e1), abs(e2), abs(e3), abs(e4), abs(e5)]

    # Mean Errors
    mean_error_precision = np.mean(errors_precision)
    mean_error_repeatability = np.mean(errors_repeatability)

    # Print Mean Errors
    print("\nMean Errors:")
    print(precision_label + f": {mean_error_precision} mm")
    print(repeatability_label + f": {mean_error_repeatability} mm")


if __name__ == '__main__':        
    main("data_repeatability_16w.json")