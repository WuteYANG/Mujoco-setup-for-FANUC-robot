import os
import time
import math
import numpy as np
from math import sin, cos, pi, atan2, asin, acos, sqrt
from numpy import cross
# import trajectory_cubic as traj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Path3DCollection
import matplotlib.animation as animation

# 0.4 -0.2 0.35 180 0 0
# 0.5416  0.0955  0.8050 90.0000 -70.0000  100.0000

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def main_FK(q, no_sol):
    """main function (Forward Kinematics)"""
    if no_sol is False:
        command = q
        vec = np.array(command, np.float)
        points, axis_rotation = all_points_(vec)
        p_end_eff, r_end_eff = FK(vec)
        #end_effector = np.array(points[-1,0], points[-1,1], points[-1,2], r_end_eff[0], r_end_eff[1], r_end_eff[2])
        #print("end effector: ", end_effector)
        print("end effector: ", p_end_eff, r_end_eff)
                
        t1 = time.time()
        #bounding_box = b_box(points, axis_rotation)

        plt_title = 'FANUC Robot Arm Simulation'
        draw_arm(points, axis_rotation, plt_title)
    else:
        pass

def main_IK():
    """main function (Inverse Kinematics)"""
    # input = x, y, z
    command = input("command: ")

    command = command.split()
    if len(command) != 6:
        print("Error!")
    else:
        vec = np.array(command, np.float)
        q, no_sol = IK(vec)

    if no_sol is True:
        print()
        print("No Solution! (Singular)")
        print()
    else:
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print("q_sols: \n", q)
    
    return q, no_sol

def main_FK2(q, no_sol):
    """main function (Forward Kinematics)"""
    if no_sol is False:
        command = q
        vec = np.array(command, np.float)
        points, axis_rotation = all_points_(vec)
        p_end_eff, r_end_eff = FK(vec)
        #end_effector = np.array(points[-1,0], points[-1,1], points[-1,2], r_end_eff[0], r_end_eff[1], r_end_eff[2])
        #print("end effector: ", end_effector)
        print("end effector: ", p_end_eff, r_end_eff)
                
        t1 = time.time()
        #bounding_box = b_box(points, axis_rotation)

        plt_title = 'FANUC Robot Arm Simulation'
        draw_arm(points, axis_rotation, plt_title)
    else:
        pass

def main_IK2():
    """main function (Inverse Kinematics)"""
    # input = x, y, z
    command = input("command: ")

    command = command.split()
    if len(command) != 6:
        print("Error!")
    else:
        vec = np.array(command, np.float)
        q, no_sol = IK(vec)

    if no_sol is True:
        print()
        print("No Solution! (Singular)")
        print()
    else:
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print("q_sols: \n", q)
    
    return q, no_sol

def IK(vec):
    """
    6-DOF,
    original mechanism,
    Geometric Solution
    Inverse Kinematics function
    """
    no_sol = False
    (px, py, pz, rx, ry, rz) = (vec[0], vec[1], vec[2], vec[3]/180*math.pi, vec[4]/180*math.pi, vec[5]/180*math.pi)

    Rx = np.array([ [1, 0, 0],
                    [0, cos(rx), -sin(rx)],
                    [0, sin(rx), cos(rx)] ])
    Ry = np.array([ [cos(ry), 0, sin(ry)],
                    [0, 1, 0],
                    [-sin(ry), 0, cos(ry)] ])
    Rz = np.array([ [cos(rz), -sin(rz), 0],
                    [sin(rz), cos(rz), 0],
                    [0, 0, 1] ])
    rot = np.dot(Rz, np.dot(Ry,Rx))
    #print('rot:', rot)

    # DH-parameters
    d = np.array([0.3300, 0.0000, 0.0000, -0.4200, 0.0000, -0.0800]) 
    al = np.array([-np.pi/2, np.pi, np.pi/2, -np.pi/2, np.pi/2, np.pi])
    a = np.array([0.0500, 0.4400, -0.035, 0, 0, 0])
    q = np.zeros(6)
    """
    Calculate theta1
    """
    t1 = np.zeros(2)
    T04_p = np.array([ px - np.abs(d[5])*rot[0,2], py - np.abs(d[5])*rot[1,2], pz - np.abs(d[5])*rot[2,2] ])
    #print('T04_p:', T04_p)
    t1 = np.array([ atan2(T04_p[1], T04_p[0]), atan2(T04_p[1], T04_p[0]) + math.pi ])
    #print('theta1',t1*180/math.pi)
    """
    Calculate theta3
    """
    q[0] = t1[0]
    T01 = np.array([[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0]),  a[0]*math.cos(q[0])],
                    [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0]),  a[0]*math.sin(q[0])],
                    [              0,                  math.sin(al[0]),                 math.cos(al[0]),                 d[0]],
                    [              0,                                0,                               0,                   1]] )
    R01 = np.array([[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0])],
                    [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0])],
                    [              0,                  math.sin(al[0]),                 math.cos(al[0])]] )
    P01 = np.array([T01[0,3],T01[1,3],T01[2,3]])
    #print('T01_p:',T01[0,3], T01[1,3], T01[2,3])
    T14_p = np.array([T04_p[0]-T01[0,3],T04_p[1]-T01[1,3],T04_p[2]-T01[2,3]])
    l2 = sqrt(T14_p[0]*T14_p[0]+T14_p[1]*T14_p[1]+T14_p[2]*T14_p[2])
    l1 = sqrt(a[2]*a[2]+d[3]*d[3])
    phi = acos((l1*l1+a[1]*a[1]-l2*l2)/(2*l1*a[1]))
    alpha = acos((-a[2])/l1)
    t3 = np.array([-np.pi+phi+alpha, -np.pi-phi+alpha])
    #t3 = np.array([np.pi-phi-alpha, -np.pi-phi+alpha])
    #print('theta3:',t3*180/np.pi)
    """
    Calculate theta2
    """
    beta1 = atan2(T14_p[2],sqrt(T14_p[0]*T14_p[0]+T14_p[1]*T14_p[1]))
    beta2 = asin((l1*l1+a[1]*a[1]-l2*l2)/(2*a[1]*l1)) + asin((l2*l2+l1*l1-a[1]*a[1])/(2*l1*l2))
    if T04_p[2] >= T01[2,3]:
        t2 = np.array([np.pi/2-(np.abs(beta1)+beta2), np.pi/2+(np.abs(beta1)-beta2)])
    else:
        t2 = np.array([np.pi/2+(np.abs(beta1)-beta2), np.pi/2-(np.abs(beta1)+beta2)])
    #print('theta2',t2*180/np.pi)
    """
    Calculate theta5
    """
    q[0] = t1[0]
    q[1] = t2[0]-math.pi/2
    q[2] = t3[0]+math.pi
    q[3] = 0
    T12 = np.array( [[ math.cos(q[1]),  -math.sin(q[1])*math.cos(al[1]),  math.sin(q[1])*math.sin(al[1]),  a[1]*math.cos(q[1])],
                     [ math.sin(q[1]),   math.cos(q[1])*math.cos(al[1]), -math.cos(q[1])*math.sin(al[1]),  a[1]*math.sin(q[1])],
                     [              0,                  math.sin(al[1]),                 math.cos(al[1]),                 d[1]],
                     [              0,                                0,                               0,                   1]] )
    T23 = np.array( [[ math.cos(q[2]),  -math.sin(q[2])*math.cos(al[2]),  math.sin(q[2])*math.sin(al[2]),  a[2]*math.cos(q[2])],
                     [ math.sin(q[2]),   math.cos(q[2])*math.cos(al[2]), -math.cos(q[2])*math.sin(al[2]),  a[2]*math.sin(q[2])],
                     [              0,                  math.sin(al[2]),                 math.cos(al[2]),                 d[2]],
                     [              0,                                0,                               0,                   1]] )
    T34 = np.array( [[ math.cos(q[3]),  -math.sin(q[3])*math.cos(al[3]),  math.sin(q[3])*math.sin(al[3]),  a[3]*math.cos(q[3])],
                     [ math.sin(q[3]),   math.cos(q[3])*math.cos(al[3]), -math.cos(q[3])*math.sin(al[3]),  a[3]*math.sin(q[3])],
                     [              0,                  math.sin(al[3]),                 math.cos(al[3]),                 d[3]],
                     [              0,                                0,                               0,                   1]] )

    T03 = np.array(np.dot(T01, np.dot(T12, T23)))
    t5 = acos(rot[0,2]*(-T03[0,2])+rot[1,2]*(-T03[1,2])+rot[2,2]*(-T03[2,2]))
    if rot[2,2] <= 0-0.000001 and rot[2,2] >= 0+0.000001:
        t5 = 0
    elif rot[2,2] < 0-0.000001:
        t5 = -t5 # end-effector point down
    #t5 = math.pi/2-acos(rot[0,2]*T04[0,1]+rot[1,2]*T04[1,1]+rot[2,2]*T04[2,1])
    q[4] = t5
    #print('theta5:',t5*180/math.pi)
    """
    Calculate theta 4 6
    """
    R03 = np.array(T03[0:3, 0:3])
    R30 = np.transpose(R03)
    rott = np.array(rot)
    
    for i in range(3):
        for j in range(3):
            if rot[i,j] <= 0+0.00016 and rot[i,j] >= 0-0.00016:
                rot[i,j] = 0.0
            if R30[i,j] <= 0+0.00016 and R30[i,j] >= 0-0.00016:
                R30[i,j] = 0.0
    rott[:,0] = -rot[:,0]
    rott[:,2] = -rot[:,2]
    for i in range(3):
        for j in range(3):
            if rott[i,j] <= 0+0.0001 and rott[i,j] >= 0-0.0001:
                rott[i,j] = 0.0
    R36 = np.dot(R30, rott)

    for i in range(3):
        for j in range(3):
            if R36[i,j] <= 0+0.0001 and R36[i,j] >= 0-0.0001:
                R36[i,j] = 0.0
    t4 = atan2(R36[1,2]/math.sin(t5), R36[0,2]/math.sin(t5))
    if R36[2,0] == 0:
        t6 = atan2(R36[2,1]/math.sin(t5), R36[2,0]/math.sin(t5))
    else:    
        t6 = atan2(R36[2,1]/math.sin(t5), -R36[2,0]/math.sin(t5))
    """
    Reorganize q
    """
    q[1] = t2[0]
    q[2] = t3[0]
    q[3] = t4
    q[5] = t6
    for i in range(6):      ## There is singular point
        if math.isnan(q[i]):
            no_sol = True
            break
    
    return q*180/math.pi, no_sol

def FK(q_vec):
    """
    6-DOF,
    original mechanism,
    Forward Kinematics function,
    return [x,y,z] of end-effector
    """
    dof = 6
    # DH-parameters
    d = np.array([0.3300, 0.0000, 0.0000, -0.4200, 0.0000, -0.0800])        
    al = np.array([-np.pi/2, np.pi, np.pi/2, -np.pi/2, np.pi/2, np.pi])
    a = np.array([0.0500, 0.4400, -0.035, 0, 0, 0])
    theta = np.array([0, -90, 180, 0, 0, 180]) # From DH table
    q_vec = (q_vec + theta)
    # each joint angle
    q = np.zeros(dof)
    for i in range(dof):
        q[i] = q_vec[i]*math.pi/180

    T01 = np.array( [[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0]),  a[0]*math.cos(q[0])],
                     [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0]),  a[0]*math.sin(q[0])],
                     [              0,                  math.sin(al[0]),                 math.cos(al[0]),                 d[0]],
                     [              0,                                0,                               0,                   1]] )
    T12 = np.array( [[ math.cos(q[1]),  -math.sin(q[1])*math.cos(al[1]),  math.sin(q[1])*math.sin(al[1]),  a[1]*math.cos(q[1])],
                     [ math.sin(q[1]),   math.cos(q[1])*math.cos(al[1]), -math.cos(q[1])*math.sin(al[1]),  a[1]*math.sin(q[1])],
                     [              0,                  math.sin(al[1]),                 math.cos(al[1]),                 d[1]],
                     [              0,                                0,                               0,                   1]] )
    T23 = np.array( [[ math.cos(q[2]),  -math.sin(q[2])*math.cos(al[2]),  math.sin(q[2])*math.sin(al[2]),  a[2]*math.cos(q[2])],
                     [ math.sin(q[2]),   math.cos(q[2])*math.cos(al[2]), -math.cos(q[2])*math.sin(al[2]),  a[2]*math.sin(q[2])],
                     [              0,                  math.sin(al[2]),                 math.cos(al[2]),                 d[2]],
                     [              0,                                0,                               0,                   1]] )
    T34 = np.array( [[ math.cos(q[3]),  -math.sin(q[3])*math.cos(al[3]),  math.sin(q[3])*math.sin(al[3]),  a[3]*math.cos(q[3])],
                     [ math.sin(q[3]),   math.cos(q[3])*math.cos(al[3]), -math.cos(q[3])*math.sin(al[3]),  a[3]*math.sin(q[3])],
                     [              0,                  math.sin(al[3]),                 math.cos(al[3]),                 d[3]],
                     [              0,                                0,                               0,                   1]] )
    T45 = np.array( [[ math.cos(q[4]),  -math.sin(q[4])*math.cos(al[4]),  math.sin(q[4])*math.sin(al[4]),  a[4]*math.cos(q[4])],
                     [ math.sin(q[4]),   math.cos(q[4])*math.cos(al[4]), -math.cos(q[4])*math.sin(al[4]),  a[4]*math.sin(q[4])],
                     [              0,                  math.sin(al[4]),                 math.cos(al[4]),                 d[4]],
                     [              0,                                0,                               0,                   1]] )
    T56 = np.array( [[ math.cos(q[5]),  -math.sin(q[5])*math.cos(al[5]),  math.sin(q[5])*math.sin(al[5]),  a[5]*math.cos(q[5])],
                     [ math.sin(q[5]),   math.cos(q[5])*math.cos(al[5]), -math.cos(q[5])*math.sin(al[5]),  a[5]*math.sin(q[5])],
                     [              0,                  math.sin(al[5]),                 math.cos(al[5]),                 d[5]],
                     [              0,                                0,                               0,                   1]] )
    T_each = np.array([T01, T12, T23, T34, T45, T56])
    # after for loop, T_each = [T01, T02, T03, T04, T05, T06]
    for i in range(6):
        if i > 0:
            T_each[i] = np.dot(T_each[i-1],T_each[i])

    ## The position of end-effector
    points = np.zeros(3)
    points = np.array([ T_each[5][0, 3], T_each[5][1, 3], T_each[5][2, 3] ])

    ## Orientation
    ry = math.atan2(-T_each[5][2,0], math.sqrt(T_each[5][0,0]*T_each[5][0,0]+T_each[5][1,0]*T_each[5][1,0]))
    if ry <= -np.pi/2+0.00000001 and ry >= -np.pi/2-0.00000001:
        rz = 0
        rx = -math.atan2(T_each[5][0,1], T_each[5][1,1])
    elif ry <= np.pi/2+0.00000001 and ry >= np.pi/2-0.00000001:
        rz = 0
        rx = math.atan2(T_each[5][0,1], T_each[5][1,1])
    else:
        rz = math.atan2(T_each[5][1,0]/math.cos(ry), T_each[5][0,0]/math.cos(ry))
        rx = math.atan2(T_each[5][2,1]/math.cos(ry), T_each[5][2,2]/math.cos(ry))
    r_end_effector = np.array([rx, ry, rz]) / np.pi * 180

    return (points, r_end_effector) 

def all_points_(q_vec):
    """
    6-DOF
    original mechanism,
    calculate forward kinematic and the points we need
    """
    dof = 6
    # DH-parameters
    d = np.array([0.3300, 0.0000, 0.0000, -0.4200, 0.0000, -0.0800]) 
    al = np.array([-np.pi/2, np.pi, -np.pi/2, np.pi/2, -np.pi/2, np.pi])
    a = np.array([0.0500, 0.4400, 0.035, 0, 0, 0])
    theta = np.array([0, -90, 0, 0, 0, 0]) # From DH table
    q_vec = q_vec + theta
    # each joint angle
    q = np.zeros(dof)
    for i in range(dof):
        q[i] = q_vec[i]*math.pi/180

    T01 = np.array( [[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0]),  a[0]*math.cos(q[0])],
                     [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0]),  a[0]*math.sin(q[0])],
                     [              0,                  math.sin(al[0]),                 math.cos(al[0]),                 d[0]],
                     [              0,                                0,                               0,                   1]] )
    T12 = np.array( [[ math.cos(q[1]),  -math.sin(q[1])*math.cos(al[1]),  math.sin(q[1])*math.sin(al[1]),  a[1]*math.cos(q[1])],
                     [ math.sin(q[1]),   math.cos(q[1])*math.cos(al[1]), -math.cos(q[1])*math.sin(al[1]),  a[1]*math.sin(q[1])],
                     [              0,                  math.sin(al[1]),                 math.cos(al[1]),                 d[1]],
                     [              0,                                0,                               0,                   1]] )
    T23 = np.array( [[ math.cos(q[2]),  -math.sin(q[2])*math.cos(al[2]),  math.sin(q[2])*math.sin(al[2]),  a[2]*math.cos(q[2])],
                     [ math.sin(q[2]),   math.cos(q[2])*math.cos(al[2]), -math.cos(q[2])*math.sin(al[2]),  a[2]*math.sin(q[2])],
                     [              0,                  math.sin(al[2]),                 math.cos(al[2]),                 d[2]],
                     [              0,                                0,                               0,                   1]] )
    T34 = np.array( [[ math.cos(q[3]),  -math.sin(q[3])*math.cos(al[3]),  math.sin(q[3])*math.sin(al[3]),  a[3]*math.cos(q[3])],
                     [ math.sin(q[3]),   math.cos(q[3])*math.cos(al[3]), -math.cos(q[3])*math.sin(al[3]),  a[3]*math.sin(q[3])],
                     [              0,                  math.sin(al[3]),                 math.cos(al[3]),                 d[3]],
                     [              0,                                0,                               0,                   1]] )
    T45 = np.array( [[ math.cos(q[4]),  -math.sin(q[4])*math.cos(al[4]),  math.sin(q[4])*math.sin(al[4]),  a[4]*math.cos(q[4])],
                     [ math.sin(q[4]),   math.cos(q[4])*math.cos(al[4]), -math.cos(q[4])*math.sin(al[4]),  a[4]*math.sin(q[4])],
                     [              0,                  math.sin(al[4]),                 math.cos(al[4]),                 d[4]],
                     [              0,                                0,                               0,                   1]] )
    T56 = np.array( [[ math.cos(q[5]),  -math.sin(q[5])*math.cos(al[5]),  math.sin(q[5])*math.sin(al[5]),  a[5]*math.cos(q[5])],
                     [ math.sin(q[5]),   math.cos(q[5])*math.cos(al[5]), -math.cos(q[5])*math.sin(al[5]),  a[5]*math.sin(q[5])],
                     [              0,                  math.sin(al[5]),                 math.cos(al[5]),                 d[5]],
                     [              0,                                0,                               0,                   1]] )
    T_each = np.array([T01, T12, T23, T34, T45, T56])
    # after for loop, T_each = [T01, T02, T03, T04, T05, T06]
    for i in range(6):
        if i > 0:
            T_each[i] = np.dot(T_each[i-1],T_each[i])
    #all axis are unit vectors [[x,y,z],[x,y,z],[x,y,z]]
    axis_rotation = np.zeros((6,3,3))
    for i in range(6):
        for j in range(3):
            axis_rotation[i][j] = np.array([T_each[i][j,0], T_each[i][j,1], T_each[i][j,2]])

    # there are base & point1~7 & end point
    points = np.zeros((9, 3))
    points[1] = points[0] - d[0]*axis_rotation[0][:,1]
    points[2] = np.array([ T_each[0][0, 3], T_each[0][1, 3], T_each[0][2, 3] ])
    points[3] = np.array([ T_each[1][0, 3], T_each[1][1, 3], T_each[1][2, 3] ])
    points[4] = np.array([ T_each[2][0, 3], T_each[2][1, 3], T_each[2][2, 3] ])
    points[5] = np.array([ T_each[3][0, 3], T_each[3][1, 3], T_each[3][2, 3] ])
    points[6] = np.array([ T_each[4][0, 3], T_each[4][1, 3], T_each[4][2, 3] ])
    points[7] = np.array([ T_each[5][0, 3], T_each[5][1, 3], T_each[5][2, 3] ])
    points[8] = np.array([ T_each[5][0, 3], T_each[5][1, 3], T_each[5][2, 3] ])
    #rot = np.zeros((3,3))

    return (points, axis_rotation)

def b_box(points, axis_rotation):
    """
    only 3-DOF, q4,q5,q6 is 0 deg
    calculate all bounding box with known parameters,
    each information includes mass center(x,y,z) & length(x,y,z)/2
    """
    bounding_box = [None]*6
    # index:0->mass center, index:1~8->vertices
    # there are box1 ~ box6
    box_center = np.zeros((6,3))
    box_center[0] = points[0] + np.dot(axis_rotation[0], np.array([0, -0.1650, 0]))
    box_center[1] = (points[2] + points[3])/2
    box_center[2] = points[4] # point
    box_center[3] = (points[4] + points[5])/2
    box_center[4] = (points[5] + points[6])/2 # point
    box_center[5] = (points[6] + points[7])/2

    # (x, y, z)/2 of each bounding box. *different reference frame
    length = 0.5 * np.array( [[0.1900, 0.3300, 0.1900],
                              [0.4750, 0.1200, 0.2350],
                              [0.0000, 0.0000, 0.0000],
                              [0.1100, 0.4200, 0.1100],
                              [0.0000, 0.0000, 0.0000],
                              [0.0800, 0.0900, 0.0800]] )

    # calculate the vetices of bounding boxes
    box_vertices = np.zeros((8, 3))
    v = np.array( [[-1,-1,-1],
                   [-1,1,-1],
                   [1,1,-1],
                   [1,-1,-1],
                   [-1,-1,1],
                   [-1,1,1],
                   [1,1,1],
                   [1,-1,1]] )

    for i in range(6):
        c = len(box_vertices)
        for j in range(c):
            box_vertices[j] = box_center[i] + np.dot( axis_rotation[i], length[i]*v[j] )
        bounding_box[i] = [box_center[i], np.copy(box_vertices)]

    return bounding_box


def draw_arm(points, axis_rotation, plt_title=False):
    """draw manipulator & environment"""

    # set the properties of x, y, z axis
    fig = plt.figure(figsize=(8, 6), dpi=100)

    ax = Axes3D(fig)
    limit = 1.2
    # ax.set_xlim(-limit, limit)
    # ax.set_ylim(-limit, limit)
    # ax.set_zlim(-limit, limit)
    
    # CH5 exp
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.9, 0.1)
    # ax.set_zlim(-0.4, 0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-0.5, limit)

    ax.set_xlabel('x[m]', fontsize=12)
    ax.set_ylabel('y[m]', fontsize=12)
    ax.set_zlabel('z[m]', fontsize=12)
    ticks = np.linspace(-limit ,limit ,num=7)
    ticks2 = np.linspace(-0.4, limit, num=5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks2)

    # CH5 exp
    # ax.set_xticks(np.linspace(-0.5, 0.5 ,num=6))
    # ax.set_yticks(np.linspace(-0.9 ,0.1 ,num=6))
    # ax.set_zticks(np.linspace(-0.4 ,0.6 ,num=6))

    ax.tick_params(labelsize=12)
    ax.view_init(elev=0.1,azim=45)
    ## Point of end-effector
    xs = points[7, 0]
    ys = points[7, 1]
    zs = points[7, 2]
    ax.scatter(xs, ys, zs, color="red",marker="o", s=20)
    
    # CH5 exp
    # ax.view_init(elev=20,azim=-100)
    
    # # draw obstacles
    # obstacles_box, obstacles_axis, length_obs = obstacles()
    # c = len(obstacles_box)
    
    # for j in range(c):
    #     x = obstacles_box[j][1][:, 0]
    #     y = obstacles_box[j][1][:, 1]
    #     z = obstacles_box[j][1][:, 2]
    #     vertices = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
    #     tupleList = list(zip(x, y, z))
    #     poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
    #     # ax.scatter(x,y,z, color="red",marker="*", s=5)

    #     if j == 0:
    #         obs_color = 'red'; lw=0.2; ls=':'
    #     else:
    #         obs_color = 'gray'; lw=1.5; ls=':'
    #     ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=lw, linestyles=ls))
    #     if j == 0:
    #         ax.add_collection3d(Poly3DCollection(poly3d, facecolors=obs_color, linewidths=1, alpha=0.1))
    
    if plt_title is not False:
        ax.set_title(plt_title, fontsize=14)
    
    # draw manipulator and joints
    '''
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    ax.scatter(xs, ys, zs, color='blue',marker='o', s=10)
    ax.scatter(0, 0, 0, color='black',marker='^',s=20, label='base')
    ax.plot(xs, ys, zs, label='manipulator')
    ax.legend()
    '''
    # draw 6 bounding boxes
    bounding_box = b_box(points, axis_rotation)
    for i in range(len(bounding_box)):
        x = bounding_box[i][1][:, 0]
        y = bounding_box[i][1][:, 1]
        z = bounding_box[i][1][:, 2]
        vertices = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
        tupleList = list(zip(x, y, z))
        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        # ax.scatter(x,y,z, color="red",marker="*", s=5)
        # color = ['#0072e3', '#00db00', '#ad5a5a', 'w', 'w', 'w']
        #color = [(0.2,0.5,0.5), 'skyblue', (0.8,0.5,0.5), 'silver', 'silver', 'silver']
        color = [(0.3,0.3,0.3), (1,1,0.3),(1,1,0.3),(1,1,0.3),(1,1,0.3),'k']
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color[i], linewidths=1, alpha=0.5))
        ax.add_collection3d(Line3DCollection(poly3d, colors='#5b5b5b', linewidths=0.2, linestyles=':'))
        '''
        # draw box center
        x_ = bounding_box[i][0][0]
        y_ = bounding_box[i][0][1]
        z_ = bounding_box[i][0][2]
        ax.scatter(x_,y_,z_, color='red',marker='*', s=20)
        '''
    
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # mode = input("mode: ")
    # mode = int(mode)
    # # mode = 0
    # if mode == 0:
    #     print("mode: FK")
    #     main_FK()
    # if mode == 1:
    #     print("mode: IK")
    keepgoing = True
    while keepgoing:
        q, no_sol = main_IK2()
        main_FK2(q, no_sol)
        keepgoing = input("continue: ")
        if keepgoing == "0":
            keepgoing = False
        else:
            pass