import numpy as np
from pydrake.all import (AbstractValue, LeafSystem,
                         PointCloud, RigidTransform, Fields, BaseField, MathematicalProgram, Solve, Sphere, Rgba)
import typing
from manipulation.meshcat_utils import AddMeshcatTriad


class COMSolver(LeafSystem):
    """
    Outputs the highest point on the stack cylinder to determine 
    where to place next block   

    InputPorts:
    - external_torque : 7-array = external torques on iiwa joints
    OutputPorts:
    - com : 3-array = center of mass of held block relative to end of iiwa
    """

    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)
        self._plant = plant
        self._meshcat = meshcat
        self._external_torque_index = self.DeclareVectorInputPort(
            "external_torque", 7).get_index()
        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])).get_index()

        # output is NaN if it could not be solved (probably means arm is moving)
        self.DeclareVectorOutputPort(
            "com", 3, self.CalcCOM)

    def external_torque_from(self, body_poses, displacement, force):
        res = []
        # reversed_joints = [2, 3, 4, 6]
        for idx in range(7):
            joint_name = f"iiwa_joint_{idx + 1}"
            joint = self._plant.GetJointByName(joint_name)
            X_par = body_poses[int(
                self._plant.GetBodyByName(f"iiwa_link_{idx}").index())]
            X_par_joint = joint.frame_on_parent().GetFixedPoseInBodyFrame()
            X_joint = X_par @ X_par_joint
            joint_axis = X_joint.rotation() @ joint.revolute_axis()
            # if idx in reversed_joints:
            #     joint_axis = -joint_axis
            joint_position = X_joint.translation()
            AddMeshcatTriad(
                self._meshcat, f"joint{idx}", X_PT=X_joint)
            lever_arm = displacement - joint_position
            expected_torque = joint_axis.dot(
                np.cross(lever_arm, np.concatenate((np.array([0, 0]), force))))
            res.append(expected_torque)
        return np.array(res)

    def CalcCOM(self, context, output):
        program = MathematicalProgram()
        displacement = program.NewContinuousVariables(3, "d")
        force = program.NewContinuousVariables(1, "f")
        external_torques = self.get_input_port(
            self._external_torque_index).Eval(context)
        body_poses = self.get_input_port(self._body_poses_index).Eval(context)
        # reversed_joints = [3, 4, 5, 6]
        for idx in range(7):
            joint_name = f"iiwa_joint_{idx + 1}"
            joint = self._plant.GetJointByName(joint_name)
            X_par = body_poses[int(
                self._plant.GetBodyByName(f"iiwa_link_{idx}").index())]
            X_par_joint = joint.frame_on_parent().GetFixedPoseInBodyFrame()
            X_joint = X_par @ X_par_joint
            joint_axis = X_joint.rotation() @ joint.revolute_axis()
            # if idx in reversed_joints:
            #     joint_axis = -joint_axis
            joint_position = X_joint.translation()
            AddMeshcatTriad(
                self._meshcat, f"joint{idx}", X_PT=X_joint)
            lever_arm = displacement - joint_position
            expected_torque = joint_axis.dot(
                np.cross(lever_arm, np.concatenate((np.array([0, 0]), force))))
            torque_error = expected_torque - external_torques[idx]
            program.AddCost(torque_error ** 2)
        program.AddLinearConstraint(force[0] <= 0)
        program.SetInitialGuess(displacement, body_poses[int(
            self._plant.GetBodyByName(f"iiwa_link_7").index())].translation())
        program.SetInitialGuess(force, np.array([-0.3]))
        result = Solve(program)
        if result.is_success():
            com = result.GetSolution(displacement)
            output.set_value(com)
            expected = self.external_torque_from(
                body_poses, result.GetSolution(displacement), result.GetSolution(force))
            normalized_t = external_torques / np.sum(np.abs(external_torques))
            brick_ts = []
            for idx in range(17, 20):
                brick_ts.append(self.external_torque_from(
                    body_poses, body_poses[idx].translation(), [-0.028]))
            normalized_bricks = []
            for brick_t in brick_ts:
                normalized_bricks.append(
                    brick_t / np.sum(np.abs(brick_t)))
            self._meshcat.SetObject(
                path="/com", shape=Sphere(0.01), rgba=Rgba(0.19, 0.72, 0.27, 1.0))
            self._meshcat.SetTransform(
                path="/com", X_ParentPath=RigidTransform(com))
        else:
            output.set_value(np.full((3,), np.nan))
