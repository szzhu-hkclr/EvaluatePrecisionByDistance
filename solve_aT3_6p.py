import os
import numpy as np

def solve_aT3_6p(Link3TEnds, marker_points):

    ## solve aT3 = [r1, r2, r3, t1]     6^p = [p1, p2, p3]
    ##             [r4, r5, r6, t2]
    ##             [r7, r8, r9, t3]
    ##             [0, 0, 0, 1]
    ##  Formulate  Ax = b   A is 3n*15, x is 15*1, b is 3n*1 
    ##    where x = [p1, p2, p3, r1, r2, r3, t1, r4, r5, r6, t2, r7, r8, r9, t3]^T

    assert len(marker_points) == len(Link3TEnds)
    n = len(marker_points)
    assert n >= 5

    A = np.zeros((3*n, 15))
    b = np.zeros((3*n, 1))

    for i in range(n):
        marker_point = [x for x in marker_points[i]]
        
        Link3TEnd_i = Link3TEnds[i]

        ## [a1, a2, a3, b1]    
        ## [a4, a5, a6, b2] 
        ## [a7, a8, a9, b3] 
        ## [0, 0, 0, 1] 

        b[3*i, 0] = -Link3TEnd_i[0][3]        # -b1
        b[3*i+1, 0] = -Link3TEnd_i[1][3]      # -b2
        b[3*i+2, 0] = -Link3TEnd_i[2][3]      # -b3
        
        
        A[3*i][:3] = [ Link3TEnd_i[0][0], Link3TEnd_i[0][1], Link3TEnd_i[0][2] ]             # a1  a2  a3
        A[3*i][3:6] = [ -marker_point[0], -marker_point[1], -marker_point[2]]                # -q1, -q2, -q3
        A[3*i][6] = -1 


        A[3*i + 1][:3] = [ Link3TEnd_i[1][0], Link3TEnd_i[1][1], Link3TEnd_i[1][2] ]         # a4  a5  a6
        A[3*i + 1][7:10] = [ -marker_point[0], -marker_point[1], -marker_point[2]]           # -q1, -q2, -q3
        A[3*i + 1][10] = -1 


        A[3*i + 2][:3] =  [ Link3TEnd_i[2][0], Link3TEnd_i[2][1], Link3TEnd_i[2][2] ]         # a7  a8  a9
        A[3*i + 2][11:14] = [ -marker_point[0], -marker_point[1], -marker_point[2]]           # -q1, -q2, -q3
        A[3*i + 2][14] = -1

    

    
    x = np.linalg.lstsq(A, b, rcond=0)[0]

    est_3Ta = np.array([x[3:7, 0],
                        x[7:11, 0],
                        x[11:, 0],
                        [0, 0, 0, 1]])
    
    est_6p = np.array([x[:3, 0]])

    return np.linalg.inv(est_3Ta), est_6p



if __name__ == '__main__':

    
    ######## Link3TEnds: 3_T_6
    ######## marker_points: a_p
    Link3TEnds = []
    marker_points = []


    est_aT3, est_6p = solve_aT3_6p(Link3TEnds, marker_points)