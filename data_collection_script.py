import numpy as np
import zmq
import lcm
import time, copy

from drake import lcmt_robot_state
from drake import lcmt_robot_plan

from plan_runner_client.zmq_client import PlanManagerZmqClient, SchunkManager
from plan_runner_client.calc_plan_msg import (
    calc_joint_space_plan_msg,
    calc_task_space_plan_msg,
)

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.common.eigen_geometry import Quaternion, AngleAxis
from pydrake.trajectories import PiecewisePolynomial


def normalized(x):
    return x / np.linalg.norm(x)

zmq_client = PlanManagerZmqClient()
schunk = SchunkManager()
frame_E = zmq_client.plant.GetFrameByName('iiwa_link_7')
X_ET = RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0, 0, 0.114])

time.sleep(1.0)

schunk.send_schunk_position_command(100)
schunk.wait_for_command_to_finish()
time.sleep(1.0)

# 1. Go to default joint.
duration = 15
t_knots = np.linspace(0, duration, 3)
q_knots = np.zeros((3, 7))

q_knots[0,:] = zmq_client.get_current_joint_angles()
q_knots[1,:] = [0, 0.5, 0.0, -0.75, 0.0, 1.5, 0.0]
q_knots[2,:] = [0, 0.5, 0.0, -0.75, 0.0, 1.5, 0.0]
plan_msg = calc_joint_space_plan_msg(t_knots, q_knots)
zmq_client.send_plan(plan_msg)
time.sleep(1.0)
zmq_client.wait_for_plan_to_finish()
time.sleep(2.0)

print(zmq_client.get_current_ee_pose(frame_E))
print(zmq_client.get_current_ee_pose(frame_E).multiply(X_ET))

# 2. 
X_W = RigidTransform([0.5, 0.0, 0.0])
default_rot = RotationMatrix.MakeXRotation(-np.pi/2).multiply(
    RotationMatrix.MakeYRotation(-np.pi/2))

#=============================================================================
# Arc Frames
#=============================================================================

r = 0.3
phi_range = np.linspace(-np.pi/2, np.pi/2, 10)

# Divide into stages.
for phi in phi_range:
    max_theta = 0.5 * np.abs(phi) + np.pi/12
    phi_num = int(round(max_theta * 10.0))

    theta_range = np.linspace(-max_theta, max_theta, phi_num)

    duration = 2.0 * phi_num
    t_knots = np.zeros(phi_num + 1)
    t_knots[0] = 0
    t_knots[1:] = 10 + np.linspace(0, duration, phi_num)

    print(t_knots)

    X_WT_lst = []
    X_WT_lst.append(zmq_client.get_current_ee_pose(frame_E).multiply(X_ET))

    print(max_theta)
    print(phi_num)
    print(duration)

    for theta in theta_range:
        # Compute theta rotation.
        theta_rot = default_rot.multiply(RotationMatrix.MakeXRotation(theta))

        normal_W = np.array([0,0,1])
        normal_T = theta_rot.inverse().multiply(normal_W)

        AA = AngleAxis(phi, normal_T)
        rotation_now = RotationMatrix(theta_rot.multiply(AA.rotation()))

        # Compute rotation in a manner consistent with.
        ynormal = AA.rotation()[:,1]

        # Find a rotation consistent with projection.
        search_dim = 100
        y_range = np.linspace(0, 2 * np.pi, search_dim)
        AA2_storage = np.zeros((search_dim, 3, 3))
        cost_storage = np.zeros(search_dim)

        for i in range(len(y_range)):
            AA2 = rotation_now.multiply(AngleAxis(y_range[i], [0,1,0]).rotation())
            #AA2 = RotationMatrix(AngleAxis(y_range[i], [0,1,0])).multiply(rotation_now)
            #AA2 = AA2.matrix()
            # Extract x component
            cost_storage[i] = -AA2[1,0]
            AA2_storage[i] = AA2

        AA2 = AA2_storage[np.argmin(cost_storage)]

        # Comptue translation.
        translation = [
            0.55 + r * np.cos(phi) * np.sin(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(theta)
        ]

        X_WT = RigidTransform(RotationMatrix(AA2), translation)
        X_WT_lst.append(X_WT)

    plan_msg = calc_task_space_plan_msg(X_ET, X_WT_lst, t_knots)
    zmq_client.send_plan(plan_msg)
    time.sleep(1.0)
    zmq_client.wait_for_plan_to_finish()
    time.sleep(2.0)
