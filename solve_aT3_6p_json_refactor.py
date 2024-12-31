import json
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    # Normalize quaternion and convert to rotation matrix
    r = R.from_quat([q[0], q[1], q[2], q[3]])  # default scalar-last order – (x, y, z, w)
    return r.as_matrix()

def solve_aT3_6p(tracker_points, link_transforms):

    def parse_json_data(points, transforms, group):
        marker_points = []
        Link3TEnds = []
        for p in group:
            marker_point = next(item for item in points if item["name"] == p)
            link_transform = next(item for item in transforms if item["name"] == p)

            marker_points.append(marker_point["pose"])
            translation = link_transform["Translation"]
            rotation = link_transform["Rotation"]
            # Convert quaternion to rotation matrix
            r = quaternion_to_rotation_matrix(rotation)
            t = np.array(translation).reshape(3, 1)
            Link3TEnds.append(np.hstack((r, t)))

        return marker_points, Link3TEnds

    def solve_transformation(marker_points, Link3TEnds):
        n = len(marker_points)
        assert n >= 5

        A = np.zeros((3*n, 15))
        b = np.zeros((3*n, 1))

        for i in range(n):
            marker_point = [x for x in marker_points[i]]
            Link3TEnd_i = Link3TEnds[i]

            b[3*i, 0] = -Link3TEnd_i[0][3]
            b[3*i+1, 0] = -Link3TEnd_i[1][3]
            b[3*i+2, 0] = -Link3TEnd_i[2][3]

            A[3*i][:3] = [ Link3TEnd_i[0][0], Link3TEnd_i[0][1], Link3TEnd_i[0][2] ]
            A[3*i][3:6] = [ -marker_point[0], -marker_point[1], -marker_point[2]]
            A[3*i][6] = -1 

            A[3*i + 1][:3] = [ Link3TEnd_i[1][0], Link3TEnd_i[1][1], Link3TEnd_i[1][2] ]
            A[3*i + 1][7:10] = [ -marker_point[0], -marker_point[1], -marker_point[2]]
            A[3*i + 1][10] = -1 

            A[3*i + 2][:3] =  [ Link3TEnd_i[2][0], Link3TEnd_i[2][1], Link3TEnd_i[2][2] ]
            A[3*i + 2][11:14] = [ -marker_point[0], -marker_point[1], -marker_point[2]]
            A[3*i + 2][14] = -1

        x = np.linalg.lstsq(A, b, rcond=None)[0]

        est_3Ta = np.array([x[3:7, 0],
                            x[7:11, 0],
                            x[11:, 0],
                            [0, 0, 0, 1]])
        
        est_6p = np.array([x[:3, 0]])

        return np.linalg.inv(est_3Ta), est_6p

    # Solve for each group
    groups = [["P1", "P2", "P3", "P4", "P5"], ["P6", "P7", "P8", "P9", "P10"], ["P11", "P12", "P13", "P14", "P15"]]
    results = []
    for group in groups:
        marker_points, Link3TEnds = parse_json_data(tracker_points, link_transforms, group)
        est_aT3, est_6p = solve_transformation(marker_points, Link3TEnds)
        results.append((est_aT3, est_6p))

    return results

def main(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    tracker_points = data['tracker_points']
    link_transforms = data['link_transforms']

    results = solve_aT3_6p(tracker_points, link_transforms)

    for idx, (est_aT3, est_6p) in enumerate(results):
        print(f"Group {idx+1}: est_aT3 =\n{est_aT3}\n est_6p =\n{est_6p}\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python solve_aT3_6p_json_refactor.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    main(json_file)