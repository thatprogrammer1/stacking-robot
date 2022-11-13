from copy import copy
from enum import Enum

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.pick import (MakeGripperCommandTrajectory, MakeGripperFrames,
                               MakeGripperPoseTrajectory)
from pydrake.all import (AbstractValue,
                         InputPortIndex,
                         LeafSystem, PiecewisePolynomial, PiecewisePose, RigidTransform,
                         RollPitchYaw)


class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING = 2
    GO_HOME = 3


class StackingPlanner(LeafSystem):
    def __init__(self, plant, meshcat, stacking_zone_center, stacking_zone_radius):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._stacking_zone_center = stacking_zone_center
        self._stacking_zone_radius = stacking_zone_radius
        self._gripper_body_index = plant.GetBodyByName("body").index()
        num_positions = 7

        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])).get_index()
        self._grasp_index = self.DeclareAbstractInputPort(
            "grasp", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state",
                                                            2).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE))
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self._times_index = self.DeclareAbstractState(AbstractValue.Make(
            {"initial": 0.0}))

        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions).get_index()

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

        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        # for debugging
        self.meshcat = meshcat

    def Update(self, context, state):
        mode = state.get_mutable_abstract_state(int(self._mode_index))
        times = state.get_mutable_abstract_state(int(
            self._times_index))
        traj_X_G = state.get_mutable_abstract_state(
            int(self._traj_X_G_index))
        traj_q = state.get_mutable_abstract_state(int(
            self._traj_q_index))
        traj_wsg_command = state.get_mutable_abstract_state(int(
            self._traj_wsg_index))

        current_time = context.get_time()
        wsg_state = self.get_input_port(
            self._wsg_state_index).Eval(context)
        X_G = self.get_input_port(self._body_poses_index).Eval(context)[int(
            self._gripper_body_index)]

        if mode.get_value() == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times.get_value()["initial"] > 1.0:
                self.StartPicking(context, mode, times,
                                  traj_X_G, traj_wsg_command)
        elif mode.get_value() == PlannerState.GO_HOME:
            if not traj_q.get_value().is_time_in_range(current_time):
                self.StartPicking(context, mode, times,
                                  traj_X_G, traj_wsg_command)
        else:
            # If we are between pick and place and the gripper is closed, then
            # we've missed or dropped the object.  Time to replan.
            if times.get_value()["postpick"] < current_time and current_time < times.get_value()["preplace"] and wsg_state[0] < 0.01:
                self.StartWaitForObjectToSettle(
                    current_time, X_G, mode, times, traj_X_G)
            elif not traj_X_G.get_value().is_time_in_range(current_time):
                self.StartPicking(context, mode, times,
                                  traj_X_G, traj_wsg_command)
            elif np.linalg.norm(traj_X_G.get_value().GetPose(current_time).translation() - X_G.translation()) > 0.2:
                # If my trajectory tracking has gone this wrong, then I'd better
                # stop and replan.  TODO(russt): Go home, in joint coordinates,
                # instead.
                self.StartGoHome(context, mode, traj_q)

    def StartWaitForObjectToSettle(self, current_time, X_G, mode, times, traj_X_G):
        mode.set_value(
            PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        times.set_value({"initial": current_time})
        traj_X_G.set_value(PiecewisePose.MakeLinear(
            [current_time, np.inf], [X_G, X_G]))

    def StartGoHome(self, context, mode, traj_q):
        print("Replanning due to large tracking error.")
        mode.set_value(PlannerState.GO_HOME)
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T)
        traj_q.set_value(q_traj)

    def StartPicking(self, context, mode, times, traj_X_G, traj_wsg_command):
        pick_pose = None
        for _ in range(5):
            cost, pick_pose = self.get_input_port(
                self._grasp_index).Eval(context)
            if not np.isinf(cost):
                break
        else:
            raise RuntimeError("Could not find a valid grasp after 5 attempts")

        mode.set_value(PlannerState.PICKING)

        X_G, planned_times = MakeGripperFrames({
            "initial":
                self.get_input_port(self._body_poses_index).Eval(context)
                [int(self._gripper_body_index)],
            "pick": pick_pose,
            "place": RigidTransform(
                    RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                    [*self._stacking_zone_center, .20])
        }, t0=context.get_time())
        print(
            f"Planned {planned_times['postplace'] - planned_times['initial']} second trajectory in picking mode at time {context.get_time()}."
        )
        times.set_value(planned_times)

        if True:  # Useful for debugging
            AddMeshcatTriad(self._meshcat, "X_Oinitial", X_PT=X_G["initial"])
            AddMeshcatTriad(self._meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(self._meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(self._meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G.set_value(MakeGripperPoseTrajectory(X_G, planned_times))
        traj_wsg_command.set_value(MakeGripperCommandTrajectory(planned_times))

    def start_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()

    def CalcGripperPose(self, context, output):
        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index)).get_value()
        if (traj_X_G.get_number_of_segments() > 0 and
                traj_X_G.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(
                    self._traj_X_G_index)).get_value().GetPose(
                        context.get_time()))
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(self.get_input_port(self._body_poses_index).Eval(context)
                         [int(self._gripper_body_index)])

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(
            self._traj_wsg_index)).get_value()
        if (traj_wsg.get_number_of_segments() > 0 and
                traj_wsg.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context))

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_abstract_state(int(
            self._traj_q_index)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))
