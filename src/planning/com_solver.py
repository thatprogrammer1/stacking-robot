import numpy as np
from pydrake.all import (
    RigidTransform, MathematicalProgram, Solve, RevoluteJoint)
from manipulation.meshcat_utils import AddMeshcatTriad
from typing import List


class COMProblem:
    def __init__(self):
        self.program = MathematicalProgram()
        self.displacement = self.program.NewContinuousVariables(3, "d")
        self.force = self.program.NewContinuousVariables(1, "f")

    def solve(self):
        self.program.AddLinearConstraint(self.force[0] <= 0)
        self.program.SetInitialGuess(self.displacement, np.array([0, 0, 0]))
        self.program.SetInitialGuess(self.force, np.array([-0.3]))
        result = Solve(self.program)
        if result.is_success():
            return result.GetSolution(self.displacement)

    def add_com_costs(self, joints: List[RevoluteJoint], par_poses: List[RigidTransform],  external_torques: List[float], X_G: RigidTransform):
        for (joint, X_par, external_torque) in zip(joints, par_poses, external_torques):
            X_par_joint = joint.frame_on_parent().GetFixedPoseInBodyFrame()
            X_joint = X_par @ X_par_joint
            axis = X_joint.rotation() @ joint.revolute_axis()
            joint_position = X_joint.translation()
            lever_arm = X_G.GetAsMatrix34() @ np.concatenate((self.displacement,
                                                              np.array([1]))) - joint_position
            expected_torque = axis.dot(
                np.cross(lever_arm, np.concatenate((np.array([0, 0]), self.force))))
            torque_error = expected_torque - external_torque
            self.program.AddCost(torque_error ** 2)
