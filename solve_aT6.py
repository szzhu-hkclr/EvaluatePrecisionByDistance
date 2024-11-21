import numpy as np
from solve_aT3_6p_json_refactor import solve_aT3_6p  # Import the solve_aT3_6p function


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat([q[0], q[1], q[2], q[3]])  # scalar-last order (x, y, z, w)
    return r.as_matrix()


def calculate_link6_transform(json_data):
    """
    Solve for aT6 using aT3 results from solve_aT3_6p and link transforms.
    """
    # Extract necessary data from JSON
    tracker_points = json_data["tracker_points"]
    link_transforms = json_data["link_transforms"]

    # Solve for aT3 using solve_aT3_6p
    aT3_results = solve_aT3_6p(tracker_points, link_transforms)

    # Compute aT6 for each group
    aT6_results = []
    for idx, (aT3, _) in enumerate(aT3_results):
        group_name = f"P{idx * 5 + 1}"  # Starting point for each group (P1, P6, P11, etc.)
        link_transform = next(item for item in link_transforms if item["name"] == group_name)

        # Extract rotation and translation for 3T6
        r = quaternion_to_rotation_matrix(link_transform["Rotation"])
        t = np.array(link_transform["Translation"]).reshape(3, 1)

        # Construct 3T6 matrix
        link3ToLink6 = np.vstack((np.hstack((r, t)), [0, 0, 0, 1]))

        # Calculate aT6 = aT3 * 3T6
        aT6 = np.dot(aT3, link3ToLink6)
        aT6_results.append(aT6)

    return aT6_results


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python solve_aT6.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    with open(json_file, "r") as file:
        data = json.load(file)

    # Calculate aT6 for all groups
    aT6_results = calculate_link6_transform(data)

    # Print the results
    for idx, aT6 in enumerate(aT6_results):
        print(f"Group {idx + 1}: aT6 =\n{aT6}\n")