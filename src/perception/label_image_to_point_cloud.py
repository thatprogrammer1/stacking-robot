

from typing import List
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, Concatenate, LeafSystem, PointCloud,
                         RigidTransform,
                         ImageLabel16I, BaseField, Fields, ImageDepth32F, ImageTraits, ImageRgba8U)

CameraPorts = namedtuple('CameraPorts', 'cloud_index, label_index')


class LabelImageToPointCloud(LeafSystem):
    """
    Input ports:
    - label_image
    - depth_image
    - color_iamge

    Output ports:
    - PointCloud, filtering out iiwa labels
    - Segmentation could be done at this point, such as returning point clouds for each 
        object
    """

    def __init__(self, camera_info, model_labels):
        LeafSystem.__init__(self)
        label_image = AbstractValue.Make(
            ImageLabel16I(camera_info.width(), camera_info.height()))
        depth_image = AbstractValue.Make(
            ImageDepth32F(camera_info.width(), camera_info.height()))
        color_image = AbstractValue.Make(
            ImageRgba8U(camera_info.width(), camera_info.height()))
        camera_pose = AbstractValue.Make(RigidTransform())
        self._label_port = self.DeclareAbstractInputPort(
            "label_image", label_image).get_index()
        self._depth_port = self.DeclareAbstractInputPort(
            "depth_image", depth_image).get_index()
        self._color_port = self.DeclareAbstractInputPort(
            "color_image", color_image).get_index()
        self._camera_pose_port = self.DeclareAbstractInputPort(
            "camera_pose", camera_pose).get_index()

        self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud(
                new_size=0, fields=Fields(BaseField.kXYZs | BaseField.kRGBs))),
            self.GetPointCloud)
        self._camera_info = camera_info
        self._model_labels = model_labels

    def GetPointCloud(self, context, output):
        depth_image = self.get_input_port(self._depth_port).Eval(context)
        # label_image is 480 x 640 x 1
        label_image = self.get_input_port(self._label_port).Eval(context)
        color_image = self.get_input_port(self._color_port).Eval(context)
        camera_pose = self.get_input_port(self._camera_pose_port).Eval(context)
        camera_info = self._camera_info
        forbidden = []
        if self._model_labels != None:
            forbidden.append(self._model_labels['iiwa'])
            forbidden.append(self._model_labels['wsg'])

        # Copied from depth_image_to_point_cloud.cc::DoConvert
        height = depth_image.height()
        width = depth_image.width()
        cx = camera_info.center_x()
        cy = camera_info.center_y()
        fx_inv = 1 / camera_info.focal_x()
        fy_inv = 1 / camera_info.focal_y()
        X_PC = camera_pose
        tmp_res = []
        # this loop is slow, maybe we can vectorize it?
        for v in range(height):
            for u in range(width):
                if label_image.at(u, v)[0] in forbidden:
                    continue
                z = depth_image.at(u, v)[0]
                # we may need to do a check for kTooClose or kTooFar, but i can't find it
                x = z * (u - cx) * fx_inv
                y = z * (v - cy) * fy_inv
                pt = X_PC @ np.array([x, y, z])
                tmp_res.append((pt, color_image.at(u, v)[0]))
        res = PointCloud(len(tmp_res), Fields(
            BaseField.kXYZs | BaseField.kRGBs))
        xyzs = res.mutable_xyzs()
        colors = res.mutable_rgbs()
        for i in range(len(tmp_res)):
            xyzs[:, i] = tmp_res[i][0]
            colors[:, i] = tmp_res[i][1]
        output.set_value(res)