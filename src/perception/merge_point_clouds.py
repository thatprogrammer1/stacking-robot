
import numpy as np
from pydrake.all import (AbstractValue, Concatenate, LeafSystem, PointCloud,
                         RigidTransform,
                         BaseField, Fields)


class MergePointClouds(LeafSystem):
    def __init__(self, plant, bin_instance, camera_body_indices, meshcat):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        model_point_cloud = AbstractValue.Make(PointCloud(0))

        self._num_cameras = len(camera_body_indices)
        self._camera_ports = []
        for i in range(self._num_cameras):
            point_cloud_port = f"camera{i}_point_cloud"
            self._camera_ports.append(self.DeclareAbstractInputPort(
                point_cloud_port, model_point_cloud).get_index())

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud(new_size=0, fields=Fields(
                BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))),
            self.GetPointCloud)

        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        # TODO: change if we change bin size/location
        a = X_B.multiply([-.3+margin, -.3+margin, 0.015+margin])
        b = X_B.multiply([.3-margin, .3-0.025-margin, 0.5])
        print(a, b)
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        # Draws space diagonal of cropping box for visualization
        # meshcat.SetLineSegments(
        #     "/cropping_box",  self._crop_lower[:, None],
        #     self._crop_upper[:, None])

        self._camera_body_indices = camera_body_indices

    def GetPointCloud(self, context, output):
        body_poses = self.get_input_port(
            self.GetInputPort("body_poses").get_index()).Eval(context)
        pcd = []
        for i in range(self._num_cameras):
            port = self._camera_ports[i]
            cloud = self.get_input_port(
                port).Eval(context)

            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())

        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        if False:
            self._meshcat.SetObject(
                "/down_sampled_pcd", down_sampled_pcd, point_size=0.005)
        output.set_value(down_sampled_pcd)
