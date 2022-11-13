import numpy as np
from pydrake.all import (AbstractValue, LeafSystem, PointCloud, RigidTransform)


class StackDetector(LeafSystem):
    def __init__(self, stacking_zone_center, stacking_zone_radius):
        LeafSystem.__init__(self)
        self._stacking_zone_center = stacking_zone_center
        self._stacking_zone_radius = stacking_zone_radius
        self._merged_pcd_index = self.DeclareAbstractInputPort(
            "merged_pcd", AbstractValue.Make(PointCloud(0))).get_index()
        self.DeclareVectorOutputPort(
            "next_stack_position", 3, self.CalcNextStackPosition)

    def CalcNextStackPosition(self, context, output):
        merged_pcd = context.get_abstract_state(
            int(self._merged_pcd_index)).get_value()
        points = merged_pcd.xyzs()
        # stack points are points in cylinder around center
        stack_points = points[np.linalg.norm(
            points[:2, :] - self._stacking_zone_center, axis=0) <= self._stacking_zone_radius]
        # next stack position = at heigh of highest point and at center of stack cylinder laterally
        output.set_value(np.hstack((self._stacking_zone_center,
                         np.max(stack_points[2, :]))))
