import numpy as np
from pydrake.all import (AngleAxis, PiecewisePolynomial, PiecewisePose,
                         RigidTransform, RotationMatrix)
from typing import (List, Tuple)

Frame = Tuple[float, RigidTransform, bool]


def MakePickFrames(initial_pose: RigidTransform, pick_pose: RigidTransform, clearance_pose: RigidTransform, t0: float) -> List[Frame]:
    """
    initial_pose: the initial pose of the gripper
    pick_pose: the pose the gripper should be in to pick the object
    t0: time the trajectory starts at.

    Returns sequence of frames, which are tuples of:
    - time
    - gripper pose
    - is gripper open (Boolean)
    "initial" -> "prepare" -> "pre_pick" -> "pick_start" -> "pick_end" -> "postpick" -> "clearance"
    """
    frames = []

    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])

    prepick_pose = pick_pose @ X_GgraspGpregrasp

    # I'll interpolate a halfway orientation by converting to axis angle and
    # halving the angle.
    X_GinitialGprepick = initial_pose.inverse() @ prepick_pose
    angle_axis = X_GinitialGprepick.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGprepick.translation() / 2.0)

    prepare_pose = initial_pose @ X_GinitialGprepare
    p_G = np.array(prepare_pose.translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < .5:
        scale = .5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    prepare_pose.set_translation(p_G)
    prepare_time = 4.0 * np.linalg.norm(X_GinitialGprepare.translation())
    clearance_time = 5.0 * \
        np.linalg.norm(prepick_pose.translation() -
                       clearance_pose.translation())

    frames.append((t0, initial_pose, True))
    frames.append((frames[-1][0] + prepare_time, prepare_pose, True))
    frames.append((frames[-1][0] + prepare_time, prepick_pose, True))
    frames.append((frames[-1][0] + 1.5, pick_pose, True))
    frames.append((frames[-1][0] + 2.0, pick_pose, False))
    frames.append((frames[-1][0] + 2.0, prepick_pose, False))
    frames.append((frames[-1][0] + clearance_time, clearance_pose, False))

    return frames


def MakeTwistFrames(initial_pose: RigidTransform, t0: float) -> List[Frame]:
    X_GclearanceGtwist = RigidTransform(
        AngleAxis(angle=0.2, axis=np.array([1, 1, 1]) / np.sqrt(3)),
        np.zeros((3,)))
    twist_pose = initial_pose @ X_GclearanceGtwist

    frames = []
    frames.append((t0, initial_pose, False))
    frames.append((t0 + 4.0, twist_pose, False))
    return frames


def MakePlaceFrames(initial_pose: RigidTransform, place_pose: RigidTransform, t0: float) -> List[Frame]:
    """
    initial_pose: pose of gripper after clearance

    Returns sequence of frames.
    "initial" -> "place_start" -> "place_end" -> "postplace"
    """
    frames = []
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])

    time_to_grasp = 7.0 * \
        np.linalg.norm(place_pose.translation() -
                       initial_pose.translation())

    preplace_pose = place_pose @ X_GgraspGpregrasp

    frames.append((t0, initial_pose, False))
    frames.append((frames[-1][0] + time_to_grasp, preplace_pose, False))
    frames.append((frames[-1][0] + 2.0, place_pose, False))
    frames.append((frames[-1][0] + 2.0, place_pose, True))
    frames.append((frames[-1][0] + 2.0, preplace_pose, True))

    return frames


def MakeGripperTrajectories(frames: List[Frame]) -> Tuple[PiecewisePose, PiecewisePolynomial]:
    """
    Constructs a gripper position and WSG trajectory from the planned frames.
    """
    sample_times = []
    poses = []

    opened = np.array([0.107])
    closed = np.array([0.0])
    wsg_samples = []

    for (time, pose, is_open) in frames:
        sample_times.append(time)
        poses.append(pose)

        wsg_samples.append(opened if is_open else closed)

    return PiecewisePose.MakeLinear(sample_times, poses), PiecewisePolynomial.FirstOrderHold(sample_times, np.array(wsg_samples).T)
