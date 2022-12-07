

from typing import List
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, Concatenate, LeafSystem, PointCloud,
                         RigidTransform,
                         ImageLabel16I, BaseField, Fields, ImageDepth32F, ImageTraits, ImageRgba8U)

CameraPorts = namedtuple('CameraPorts', 'cloud_index, label_index')


class LabelImageToPointCloud(LeafSystem):
    """
    Generate PointCloud from camera image. 
    This code also filters out the iiwa by cheating and using labels.

    Input ports:
    - label_image
    - depth_image
    - color_image
    - camera_pose

    Output ports:
    - point_cloud
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

        # Copied from depth_image_to_point_cloud.cc::DoConvert
        height = depth_image.height()
        width = depth_image.width()
        cx = camera_info.center_x()
        cy = camera_info.center_y()
        fx_inv = 1 / camera_info.focal_x()
        fy_inv = 1 / camera_info.focal_y()
        X_PC = camera_pose
        # Nompy magic to make indexing arrays
        u = np.tile(np.arange(width), (height, 1)).flatten()
        v = np.tile(np.arange(height), (width, 1)).T.flatten()

        z = depth_image.data[:, :, 0].flatten()
        x = z * (u - cx) * fx_inv
        y = z * (v - cy) * fy_inv

        filtered_colors = color_image.data.reshape(height*width, -1)[:, 0:3]

        res = PointCloud(len(x), Fields(
            BaseField.kXYZs | BaseField.kRGBs))
        xyzs = res.mutable_xyzs()
        colors = res.mutable_rgbs()
        xyzs[:] = X_PC @ np.array([x, y, z])
        colors[:] = filtered_colors.T

        output.set_value(res)
