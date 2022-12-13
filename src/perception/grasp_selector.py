
from typing import List
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, AddMultibodyPlantSceneGraph, Concatenate, DiagramBuilder, LeafSystem, Parser,
                         PointCloud,
                         RigidTransform,
                         RollPitchYaw, ImageLabel16I, Fields, BaseField, Cylinder, Rgba)

from manipulation import FindResource, running_as_notebook
from perception.grasping_utils import GenerateAntipodalGraspCandidate
from manipulation.scenarios import (AddPackagePaths)


# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.AddAllModelsFromFile(
        FindResource("models/clutter_planning.dmd.yaml"))
    plant.Finalize()
    return builder.Build()


class GraspSelector(LeafSystem):
    def __init__(self, stacking_zone_center: np.array, stacking_zone_radius: float, meshcat, random_seed=None):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._stacking_zone_center = stacking_zone_center
        self._stacking_zone_radius = stacking_zone_radius
        model_point_cloud = AbstractValue.Make(PointCloud(0, fields=Fields(
            BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals)))

        self._point_cloud_index = self.DeclareAbstractInputPort(
            "point_cloud", model_point_cloud).get_index()
        self._segmented_clouds_index = self.DeclareAbstractInputPort(
            "segmented_clouds", AbstractValue.Make([])).get_index()

        port = self.DeclareAbstractOutputPort(
            "grasp_selection", lambda: AbstractValue.Make(
                (np.inf, RigidTransform(), np.inf)), self.SelectGrasp)
        port.disable_caching_by_default()
        port = self.DeclareAbstractOutputPort(
            "clutter_grasp_selection", lambda: AbstractValue.Make(
                (np.inf, RigidTransform(), np.inf)), self.SelectClutterGrasp)
        port.disable_caching_by_default()

        self._internal_model = make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng(random_seed)

    def SelectGrasp(self, context, output):
        # TODO: use the segmented clouds to do smarter grasp selection
        segmented_clouds = self.get_input_port(
            self._segmented_clouds_index).Eval(context)
        down_sampled_pcd = self.get_input_port(
            self._point_cloud_index).Eval(context)

        # print("Num of segmented clouds", len(segmented_clouds))

        if True:
            # Visualize how the points are segmented
            for i in range(len(segmented_clouds)):
                self._meshcat.SetObject(
                    f"/segmented_cloud_{i}", segmented_clouds[i], point_size=0.005)

            self._meshcat.SetObject(
                f"/stack_cylinder", Cylinder(self._stacking_zone_radius, 10), rgba=Rgba(0.79, 0.32, 0.32, 0.5))
            self._meshcat.SetTransform(
                path="/stack_cylinder", X_ParentPath=RigidTransform(np.hstack((self._stacking_zone_center, np.array([0])))))

        for segmented_cloud in segmented_clouds:
            if np.all(np.linalg.norm(segmented_cloud.xyzs()[:2, :] - self._stacking_zone_center[..., np.newaxis], axis=0) > self._stacking_zone_radius):
                costs = []
                X_Gs = []
                points = []
                # TODO(russt): Take the randomness from an input port, and re-enable
                # caching.
                for i in range(100):
                    cost, X_G, target_point = GenerateAntipodalGraspCandidate(
                        self._internal_model, self._internal_model_context,
                        segmented_cloud, self._rng)
                    if np.isfinite(cost):
                        costs.append(cost)
                        X_Gs.append(X_G)
                        points.append(target_point)

                if len(costs) > 0:
                    best = np.argmin(costs)
                    p = points[best]
                    # print(p)
                    height_from_ground = p[2]
                    output.set_value(
                        (costs[best], X_Gs[best], height_from_ground))
                    return
        # failed to find grasp
        X_WG = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                              [0.5, 0, 0.22])
        output.set_value((np.inf, X_WG, 0))

    def SelectClutterGrasp(self, context, output):
        segmented_clouds = self.get_input_port(
            self._segmented_clouds_index).Eval(context)
        down_sampled_pcd = self.get_input_port(
            self._point_cloud_index).Eval(context)

        # remove points from stack cylinder
        # hack to keep normals of points that remain
        points = np.array(
            [down_sampled_pcd.xyzs(), down_sampled_pcd.normals()])
        grasp_points = points[:, :, np.linalg.norm(
            points[0, :2, :] - self._stacking_zone_center[..., np.newaxis], axis=0) > self._stacking_zone_radius]
        num_points = grasp_points.shape[2]

        # print("Num of segmented clouds", len(segmented_clouds))

        if True:
            # Visualize how the points are segmented
            for i in range(len(segmented_clouds)):
                self._meshcat.SetObject(
                    f"/segmented_cloud_{i}", segmented_clouds[i], point_size=0.005)
            self._meshcat.SetObject(
                f"/stack_cylinder", Cylinder(self._stacking_zone_radius, 10), rgba=Rgba(0.79, 0.32, 0.32, 0.5))
            self._meshcat.SetTransform(
                path="/stack_cylinder", X_ParentPath=RigidTransform(np.hstack((self._stacking_zone_center, np.array([0])))))

        for segmented_cloud in segmented_clouds:
            if np.any(np.linalg.norm(segmented_cloud.xyzs()[:2, :] - self._stacking_zone_center[..., np.newaxis], axis=0) <= self._stacking_zone_radius):
                costs = []
                X_Gs = []
                points = []
                # TODO(russt): Take the randomness from an input port, and re-enable
                # caching.
                for i in range(1000):
                    cost, X_G, target_point = GenerateAntipodalGraspCandidate(
                        self._internal_model, self._internal_model_context,
                        segmented_cloud, self._rng)
                    if np.isfinite(cost):
                        costs.append(cost)
                        X_Gs.append(X_G)
                        points.append(target_point)

                if len(costs) > 0:
                    best = np.argmin(costs)
                    p = points[best]
                    print(p)
                    height_from_ground = p[2]
                    output.set_value(
                        (costs[best], X_Gs[best], height_from_ground))
                    return
        # failed to find grasp
        X_WG = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                              [0.5, 0, 0.22])
        output.set_value((np.inf, X_WG, 0))
