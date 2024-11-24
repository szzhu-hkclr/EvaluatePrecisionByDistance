import numpy as np
import json
import sys

def calculate_distance(pose1, pose2):
    """Calculate Euclidean distance between two poses."""
    return np.linalg.norm(np.array(pose1) - np.array(pose2))

def load_6p_results(json_file):
    """Load 6p results from the provided JSON file."""
    with open(json_file, 'r') as file:
        data = json.load(file)

    return [np.array(entry) for entry in data["6p_vecs"]]

def main(json_file):
    # Load 6p results from JSON
    res = load_6p_results(json_file)

    # Ground truth distances (Method 1)
    dist1 = calculate_distance(res[0], res[1])
    dist2 = calculate_distance(res[0], res[2])
    dist3 = calculate_distance(res[1], res[2])

    # Print results
    print(f"dist1={dist1}, dist2={dist2}, dist3={dist3}")
    abs_dists = [abs(dist1), abs(dist2), abs(dist3)]

    # Mean Errors
    mean_error = np.mean(abs_dists)

    # Print Mean Errors
    print(f"\nMean Errors: {mean_error*1000} mm")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python calculate_6p_dist.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    main(json_file)