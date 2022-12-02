
from typing import List
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, AddMultibodyPlantSceneGraph, Concatenate, DiagramBuilder, LeafSystem, Parser, RotationMatrix,
                         PointCloud, Rgba,
                         RigidTransform,
                         RollPitchYaw, ImageLabel16I, Fields, BaseField)

from manipulation import FindResource, running_as_notebook
# from manipulation.clutter import GraspCandidateCost
from manipulation.scenarios import (AddPackagePaths)

def GraspCandidateCosta(diagram,
                       context,
                       cloud,
                       wsg_body_index=None,
                       plant_system_name="plant",
                       scene_graph_system_name="scene_graph",
                       adjust_X_G=False,
                       verbose=False,
                       meshcat=None):
    """
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains 
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        wsg_body_index: The body index of the gripper in plant.  If None, then 
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    X_G = plant.GetFreeBodyPose(plant_context, wsg)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW @ cloud.xyzs()

    # Crop to a region inside of the finger box.
    crop_min = [-.05, 0.1, -0.00625]
    crop_max = [.05, 0.1125, 0.00625]
    indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                      crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                      crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]),
                     axis=0)

    if meshcat:
        pc = PointCloud(np.sum(indices))
        pc.mutable_xyzs()[:] = cloud.xyzs()[:, indices]
        meshcat.SetObject("planning/points", pc, rgba=Rgba(1., 0, 0), point_size=0.01)

    if adjust_X_G and np.sum(indices)>0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max())/2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(
        scene_graph_context)

    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        if verbose:
            print("Gripper is colliding with the sink!\n")
            print(f"cost: {cost}")
        return cost

    # Check collisions between the gripper and the point cloud
    margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(cloud.xyz(i),
                                                              threshold=0.01)
        if distances:
            print(distances)
            cost = np.inf
            if verbose:
                print("Gripper is colliding with the point cloud!\n")
                print(f"cost: {cost}")
            return cost

    n_GC = X_GW.rotation().multiply(cloud.normals()[:,indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0*X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0,:]**2)
    if verbose:
        print(f"cost: {cost}")
        print(f"normal terms: {n_GC[0,:]**2}")
    return cost


def GenerateAntipodalGraspCandidate(diagram,
                                    context,
                                    cloud,
                                    rng,
                                    wsg_body_index=None,
                                    plant_system_name="plant",
                                    scene_graph_system_name="scene_graph", meshcat=None):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel. 
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains 
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        rng: a np.random.default_rng()
        wsg_body_index: The body index of the gripper in plant.  If None, then 
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(np.linalg.norm(n_WS),
                      1.0), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y,Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        print("Normal was pointing down rejected")
        return np.inf, None

    Gy = y - np.dot(y,Gx)*Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll=-np.pi/3.0
    max_roll=np.pi/3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in (min_roll + (max_roll - min_roll)*alpha):
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCosta(diagram, context, cloud, adjust_X_G=True, verbose=True, meshcat=meshcat)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G

        #draw_grasp_candidate(X_G, f"collision/{theta:.1f}")
    print("No Good costs found")
    return np.inf, None


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
                (np.inf, RigidTransform())), self.SelectGrasp)
        port.disable_caching_by_default()

        self._internal_model = make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng(random_seed)

    def SelectGrasp(self, context, output):
        # TODO: use the segmented clouds to do smarter grasp selection
        segmented_clouds = self.get_input_port(
            self._segmented_clouds_index).Eval(context)
        
        print("Segmented", segmented_clouds)
        
        down_sampled_pcd = self.get_input_port(
            self._point_cloud_index).Eval(context)

        # remove points from stack cylinder
        # hack to keep normals of points that remain
        # points = np.array(
        #     [down_sampled_pcd.xyzs(), down_sampled_pcd.normals()])
        # print(points.shape)
        # grasp_points = points[:, :, np.linalg.norm(
        #     points[0, :2, :] - self._stacking_zone_center[..., np.newaxis], axis=0) > self._stacking_zone_radius]
        # num_points = grasp_points.shape[2]
        # print("Cylinder filtered", num_points)
        # cloud = PointCloud(num_points,
        #                    Fields(BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))
        # cloud.mutable_xyzs()[:] = grasp_points[0]
        # cloud.mutable_rgbs()[:] = np.array(
        #     [[0, 255, 0]]*num_points).T
        # cloud.mutable_normals()[:] = grasp_points[1]
        cloud = down_sampled_pcd
        
        if True:
            # Visualize how the points are segmented
            # for i in range(len(segmented_clouds)):
            #     self._meshcat.SetObject(
            #         f"/segmented_cloud_{i}", segmented_clouds[i], point_size=0.005)

            # Visualize what points are candidates for grasp selection
            self._meshcat.SetObject(
                "/grasp_selection_points", cloud, point_size=0.005)
            
            self._meshcat.SetLineSegments(
            "down_sampled_normals", cloud.xyzs(),
            cloud.xyzs() + 0.01 * cloud.normals())
            
        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(100 if running_as_notebook else 2):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model, self._internal_model_context,
                cloud, self._rng, meshcat=self._meshcat)
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            X_WG = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                                  [0.5, 0, 0.22])
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))
