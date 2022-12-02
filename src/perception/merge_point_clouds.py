
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
        a = X_B.multiply([0, 0, margin])
        b = X_B.multiply([10, 10, 10])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        self._camera_body_indices = camera_body_indices

    def filterPointCloudByColor(self, cloud):
        print(cloud, cloud.size())
        rgbs = cloud.rgbs()
        print("RBG", rgbs.shape)
        pos = cloud.mutable_xyzs()
        print(pos.shape)
        mask = pos[2, :] < 0.001
        print(mask.shape)
        pos[0, :] += mask * 1e3
        pos[1, :] += mask * 1e3
        pos[2, :] += mask * 1e3
        # for i in range(cloud.size()):
        #     if cloud.rgb(i)[0]==104:
        #         pos = cloud.mutable_xyz(i)
        #         # print(pos)
        #         pos[0] = 1000.
        #         pos[1] = 1000.
        #         pos[2] = 1000.
        #         # = np.array([100, 100, 100])
        #         # print(cloud.rgb(i))
            
    def GetPointCloud(self, context, output):
        body_poses = self.get_input_port(
            self.GetInputPort("body_poses").get_index()).Eval(context)
        pcd = []
        for i in range(self._num_cameras):
            port = self._camera_ports[i]
            cloud = self.get_input_port(
                port).Eval(context)

            self.filterPointCloudByColor(cloud)
            # pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            # print(pcd[-1].fields())
            pcd.append(cloud)
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())

        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        if True:
            self._meshcat.SetObject(
                "/down_sampled_pcd", down_sampled_pcd, point_size=0.005)
        output.set_value(down_sampled_pcd)
