from copy import copy
from dataclasses import dataclass

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from planning.pick import (
    MakePickFrames, MakePlaceFrames, MakeTwistFrames, MakeGripperTrajectories)
from pydrake.all import (AbstractValue,
                         InputPortIndex,
                         LeafSystem, PiecewisePolynomial, PiecewisePose, RigidTransform,
                         RollPitchYaw,
                         PointCloud, Sphere, Rgba)
from planning.com_solver import COMProblem
from typing import (List, Union)


@dataclass
class InitialState:
    pass


@dataclass
class SettleState:
    held_pose: RigidTransform
    start_time: float
    prev_stack_com: Union[np.ndarray, None]


@dataclass
class ClearSpaceState:
    pose_trajectory: PiecewisePose
    wsg_trajectory: PiecewisePolynomial


@dataclass
class PickState:
    pose_trajectory: PiecewisePose
    wsg_trajectory: PiecewisePolynomial
    target_stack_point: np.ndarray
    prev_stack_com: Union[np.ndarray, None]


@dataclass
class COMPhase1State:
    held_pose: RigidTransform
    start_time: float
    problem: COMProblem
    target_stack_point: np.ndarray
    prev_stack_com: Union[np.ndarray, None]


@dataclass
class TwistState:
    pose_trajectory: PiecewisePose
    wsg_trajectory: PiecewisePolynomial
    problem: COMProblem
    target_stack_point: np.ndarray
    prev_stack_com: Union[np.ndarray, None]


@dataclass
class COMPhase2State:
    held_pose: RigidTransform
    start_time: float
    problem: COMProblem
    target_stack_point: np.ndarray
    prev_stack_com: Union[np.ndarray, None]


@dataclass
class PlaceState:
    pose_trajectory: PiecewisePose
    wsg_trajectory: PiecewisePolynomial
    target_stack_point: np.ndarray
    placed_com: Union[np.ndarray, None]


@dataclass
class GoHomeState:
    joint_trajectory: PiecewisePolynomial
    prev_stack_com: Union[np.ndarray, None]


class StackingPlanner(LeafSystem):
    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._stack_position_index = self.DeclareVectorInputPort(
            "stack_position", 3).get_index()

        self._gripper_body_index = plant.GetBodyByName("body").index()
        self._iiwa_joints = [plant.GetJointByName(
            f"iiwa_joint_{idx + 1}") for idx in range(7)]
        self._iiwa_link_indices = [plant.GetBodyByName(
            f"iiwa_link_{idx}").index() for idx in range(7)]
        num_positions = 7

        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])).get_index()
        self._grasp_index = self.DeclareAbstractInputPort(
            "grasp", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state",
                                                            2).get_index()
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions).get_index()
        self._external_torque_index = self.DeclareVectorInputPort(
            "external_torque", 7).get_index()

        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        self.DeclareAbstractOutputPort(
            "control_mode", lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode)
        self.DeclareAbstractOutputPort(
            "reset_diff_ik", lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset)
        self.DeclareVectorOutputPort("iiwa_position_command", num_positions,
                                     self.CalcIiwaPosition)

        self._home_index = self.DeclareDiscreteState(num_positions)
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(InitialState()))

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        # for debugging
        self.meshcat = meshcat

    def Update(self, context, state):
        mode = state.get_mutable_abstract_state(int(self._mode_index))
        mode_val = mode.get_value()

        current_time = context.get_time()
        wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
        body_poses = self.get_input_port(self._body_poses_index).Eval(context)
        X_G = body_poses[int(self._gripper_body_index)]

        if isinstance(mode_val, InitialState):
            mode.set_value(SettleState(X_G, current_time, []))
        elif isinstance(mode_val, SettleState):
            start_time = mode_val.start_time
            prev_stack_com = mode_val.prev_stack_com

            if current_time > start_time + 1.0:
                self.StartPicking(context, mode, prev_stack_com)
        elif isinstance(mode_val, ClearSpaceState):
            pass
        elif (isinstance(mode_val, PickState) or isinstance(mode_val, TwistState) or isinstance(mode_val, PlaceState)) and np.linalg.norm(mode_val.pose_trajectory.GetPose(current_time).translation() - X_G.translation()) > 0.2:
            print("Replanning due to large tracking error.")
            q = self.get_input_port(
                self._iiwa_position_index).Eval(context)
            home = copy(context.get_discrete_state(
                self._home_index).get_value())
            home[0] = q[0]  # Safer to not reset the first joint.
            mode.set_value(GoHomeState(
                PiecewisePolynomial.FirstOrderHold(
                    [current_time, current_time + 5.0], np.vstack((q, home)).T), prev_stack_com
            ))
        elif isinstance(mode_val, PickState):
            pose_trajectory = mode_val.pose_trajectory
            target_stack_point = mode_val.target_stack_point
            prev_stack_com = mode_val.prev_stack_com

            if not pose_trajectory.is_time_in_range(current_time):
                mode.set_value(COMPhase1State(
                    X_G, current_time, COMProblem(), target_stack_point, prev_stack_com))
        elif isinstance(mode_val, COMPhase1State):
            start_time = mode_val.start_time
            problem = mode_val.problem
            target_stack_point = mode_val.target_stack_point
            prev_stack_com = mode_val.prev_stack_com

            if wsg_state[0] < 0.01:
                mode.set_value(SettleState(
                    X_G,
                    current_time
                ))
            elif current_time > start_time + 2.0:
                external_torques = self.get_input_port(
                    self._external_torque_index).Eval(context)
                problem.add_com_costs(self._iiwa_joints, [
                    body_poses[idx] for idx in self._iiwa_link_indices], external_torques, X_G)
                mode.set_value(TwistState(
                    *MakeGripperTrajectories(MakeTwistFrames(X_G, current_time)),
                    problem=problem,
                    target_stack_point=target_stack_point,
                    prev_stack_com=prev_stack_com
                ))
        elif isinstance(mode_val, TwistState):
            pose_trajectory = mode_val.pose_trajectory
            problem = mode_val.problem
            target_stack_point = mode_val.target_stack_point
            prev_stack_com = mode_val.prev_stack_com

            if not pose_trajectory.is_time_in_range(current_time):
                mode.set_value(COMPhase2State(
                    X_G, current_time, problem, target_stack_point, prev_stack_com))
        elif isinstance(mode_val, COMPhase2State):
            start_time = mode_val.start_time
            problem = mode_val.problem
            target_stack_point = mode_val.target_stack_point

            if current_time > start_time + 2.0:
                external_torques = self.get_input_port(
                    self._external_torque_index).Eval(context)
                problem.add_com_costs(self._iiwa_joints, [
                    body_poses[idx] for idx in self._iiwa_link_indices], external_torques, X_G)
                com = problem.solve()
                self._meshcat.SetObject(
                    path="/com", shape=Sphere(0.01), rgba=Rgba(0.19, 0.72, 0.27, 1.0))
                self._meshcat.SetTransform(
                    path="/com", X_ParentPath=RigidTransform(X_G) @ RigidTransform(com))
                height_offset = np.array([0, 0, 0.1])
                R_G = RollPitchYaw(-np.pi / 2, 0, np.pi / 2)
                place_pose = RigidTransform(
                    R_G,
                    target_stack_point + height_offset - R_G.ToRotationMatrix() @ com)
                AddMeshcatTriad(self._meshcat, "place", X_PT=place_pose)
                mode.set_value(PlaceState(
                    *MakeGripperTrajectories(MakePlaceFrames(X_G, place_pose, current_time)),
                    target_stack_point=target_stack_point,
                    stack_coms=place_pose @ com
                ))
        elif isinstance(mode_val, PlaceState):
            pose_trajectory = mode_val.pose_trajectory
            stack_coms = mode_val.stack_coms
            if not pose_trajectory.is_time_in_range(current_time):
                self.StartPicking(context, mode, stack_coms)
        elif isinstance(mode_val, GoHomeState):
            joint_trajectory = mode_val.joint_trajectory
            stack_coms = mode_val.stack_coms
            if not joint_trajectory.is_time_in_range(current_time):
                self.StartPicking(context, mode, stack_coms)

    def StartPicking(self, context, mode, prev_com_stack):
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[
            int(self._gripper_body_index)]
        pick_pose = None
        for _ in range(5):
            cost, pick_pose = self.get_input_port(
                self._grasp_index).Eval(context)
            if not np.isinf(cost):
                break
        else:
            raise RuntimeError("Could not find a valid grasp after 5 attempts")

        height_offset = np.array([0, 0, 0.3])
        stack_position = self.get_input_port(
            self._stack_position_index).Eval(context)
        clearance_pose = RigidTransform(
            RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
            stack_position + height_offset)

        frames = MakePickFrames(
            initial_pose, pick_pose, clearance_pose, context.get_time())
        AddMeshcatTriad(self._meshcat, "initial", X_PT=frames[0][1])
        AddMeshcatTriad(self._meshcat, "prepick", X_PT=frames[2][1])
        AddMeshcatTriad(self._meshcat, "pick", X_PT=pick_pose)
        AddMeshcatTriad(self._meshcat, "clearance", X_PT=clearance_pose)
        self._meshcat.SetObject(
            path="/stack", shape=Sphere(0.01), rgba=Rgba(0.21, 0.38, 0.79, 1.0))
        self._meshcat.SetTransform(
            path="/stack", X_ParentPath=RigidTransform(stack_position))
        height_offset = np.array([0, 0, 0.1])

        print(f"Planned a pick at time {context.get_time()}.")
        mode.set_value(PickState(*MakeGripperTrajectories(frames),
                       target_stack_point=stack_position, stack_coms=prev_com_stack))

    def CalcGripperPose(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        if isinstance(mode, InitialState) or isinstance(mode, GoHomeState):
            output.set_value(self.get_input_port(self._body_poses_index).Eval(context)
                             [int(self._gripper_body_index)])
        elif isinstance(mode, PickState) or isinstance(mode, TwistState) or isinstance(mode, PlaceState):
            pose_trajectory = mode.pose_trajectory
            if pose_trajectory.get_number_of_segments() > 0 and pose_trajectory.is_time_in_range(current_time):
                output.set_value(pose_trajectory.GetPose(current_time))
            else:
                output.set_value(self.get_input_port(self._body_poses_index).Eval(context)
                                 [int(self._gripper_body_index)])
        elif isinstance(mode, SettleState) or isinstance(mode, COMPhase1State) or isinstance(mode, COMPhase2State):
            held_pose = mode.held_pose
            output.set_value(held_pose)
        else:
            raise RuntimeError(f"Unknown state: {mode}")

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
        current_time = context.get_time()

        opened = np.array([0.107])
        closed = np.array([0.0])

        if isinstance(mode, PickState) or isinstance(mode, TwistState) or isinstance(mode, PlaceState):
            wsg_trajectory = mode.wsg_trajectory
            if wsg_trajectory.get_number_of_segments() > 0 and wsg_trajectory.is_time_in_range(current_time):
                output.SetFromVector(wsg_trajectory.value(current_time))
            else:
                output.SetFromVector(closed if isinstance(
                    mode, PickState) or isinstance(mode, TwistState) else opened)
        elif isinstance(mode, GoHomeState) or isinstance(mode, InitialState) or isinstance(mode, SettleState):
            output.SetFromVector([opened])
        elif isinstance(mode, COMPhase1State) or isinstance(mode, COMPhase2State):
            output.SetFromVector([closed])
        else:
            raise RuntimeError(f"Unknown state: {mode}")

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if isinstance(mode, GoHomeState):
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if isinstance(mode, GoHomeState):
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._home_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context))

    def CalcIiwaPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        current_time = context.get_time()

        if isinstance(mode, GoHomeState):
            output.SetFromVector(mode.joint_trajectory.value(current_time))
        else:
            output.SetFromVector(q)
