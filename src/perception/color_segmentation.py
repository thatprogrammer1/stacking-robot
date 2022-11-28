
import numpy as np
from pydrake.all import (AbstractValue, LeafSystem, PointCloud,
                         Fields, BaseField)


class ColorSegmentation(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0, fields=Fields(
            BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals)))

        self._point_cloud_index = self.DeclareAbstractInputPort(
            "point_cloud", model_point_cloud).get_index()

        self.DeclareAbstractOutputPort(
            "segmented_clouds", lambda: AbstractValue.Make([]), self.SegmentPoints)

    def SegmentPoints(self, context, output):
        pcd = self.get_input_port(
            self._point_cloud_index).Eval(context)
        segmented_points = []
        label_map = {color[0]: ([], [], []) for color in pcd.rgbs().T}

        for point, color, normal in zip(pcd.xyzs().T, pcd.rgbs().T, pcd.normals().T):
            label_map[color[0]][0].append(point)
            label_map[color[0]][1].append(color)
            label_map[color[0]][2].append(normal)
        for color in label_map:
            val = [np.array(a).T for a in label_map[color]]
            new_pcd = PointCloud(val[0].shape[1], Fields(
                BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))
            new_pcd.mutable_xyzs()[:] = val[0]
            # to make visualizing colors distinct, just generate random color
            new_pcd.mutable_rgbs()[:] = np.array(
                [np.random.randint(0, 256, 3)]*val[0].shape[1]).T
            new_pcd.mutable_normals()[:] = val[2]
            segmented_points.append(new_pcd)
        print("Segmentation group keys:", label_map.keys())
        output.set_value(segmented_points)
