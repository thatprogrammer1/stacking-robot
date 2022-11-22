import numpy as np
from pydrake.all import (AbstractValue, LeafSystem,
                         PointCloud, RigidTransform, Fields, BaseField)
import typing


class StackDetector(LeafSystem):
    """
    Outputs the highest point on the stack cylinder to determine 
    where to place next block   

    InputPorts:
    - merged_pcd : PointCloud = merged point cloud from all cameras
    OutputPorts:
    - next_stack_position : 3-array = highest point within stack cylinder
    """

    def __init__(self, stacking_zone_center: np.array, stacking_zone_radius: float, meshcat):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._stacking_zone_center = stacking_zone_center
        self._stacking_zone_radius = stacking_zone_radius
        self._merged_pcd_index = self.DeclareAbstractInputPort(
            "merged_pcd", AbstractValue.Make(PointCloud(0))).get_index()
        self.DeclareVectorOutputPort(
            "next_stack_position", 3, self.CalcNextStackPosition)

    def CalcNextStackPosition(self, context, output):
        merged_pcd = self.get_input_port(
            self._merged_pcd_index).Eval(context)
        points = merged_pcd.xyzs()
        # stack points are points in cylinder around center
        stack_points = points[:, np.linalg.norm(
            points[:2, :] - self._stacking_zone_center[..., np.newaxis], axis=0) <= self._stacking_zone_radius]
        # next stack position = at height of highest point and at center of stack cylinder laterally

        if True:
            cloud = PointCloud(stack_points.shape[1],
                               Fields(BaseField.kXYZs | BaseField.kRGBs))
            cloud.mutable_xyzs()[:] = stack_points
            cloud.mutable_rgbs()[:] = np.array(
                [[255, 0, 0]]*stack_points.shape[1]).T
            self._meshcat.SetObject("/stack_points", cloud, point_size=0.005)

        if stack_points.shape[1] > 0:
            pos = np.hstack((self._stacking_zone_center,
                            np.max(stack_points[2, :])))
        else:
            pos = [*self._stacking_zone_center, 0]
        print("Highest point on stack: ", pos)
        output.set_value(pos)
