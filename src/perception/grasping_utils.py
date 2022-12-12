import numpy as np
from pydrake.all import (PointCloud, RotationMatrix, RigidTransform)

def GraspCandidateCost(diagram,
                       context,
                       cloud,
                       target_xyz,
                       target_normal,
                       wsg_body_index=None,
                       plant_system_name="plant",
                       scene_graph_system_name="scene_graph",
                       adjust_X_G=False,
                       verbose=False,
                       meshcat_path=None):
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
    # print(np.sum(indices))
    if meshcat_path:
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
    
    # (ANI) For some reason this part messes up the grasps so ditch it
    # Check collisions between the gripper and the sink
    # if query_object.HasCollisions():
    #     cost = np.inf
    #     if verbose:
    #         print("Gripper is colliding with the sink!\n")
    #         print(f"cost: {cost}")
    #     return cost

    # Check collisions between the gripper and the point cloud
    margin = 0.001   # must be smaller than the margin used in the point cloud preprocessing.
    # count = 0
    cost = 0
    for i in range(cloud.size()):
        # # Only check points close to the target bc points from the other boxes dont matter
        # print(np.dot(cloud.xyz(i) - target_xyz, target_normal)/(np.linalg.norm(cloud.xyz(i) - target_xyz)*np.linalg.norm(target_normal)))
        if np.dot(cloud.xyz(i) - target_xyz, target_normal)/(np.linalg.norm(cloud.xyz(i) - target_xyz)*np.linalg.norm(target_normal)) < -0.5:
            # print("Example skipped", cloud.xyz(i), target_xyz, target_normal)
            # print(np.dot(cloud.xyz(i) - target_xyz, target_normal))
            continue
        distances = query_object.ComputeSignedDistanceToPoint(cloud.xyz(i),
                                                              threshold=margin)
        if distances:
            cost += 100

    # print(cost)        
    # if cost == np.inf:
    #     if verbose:
    #             print("Gripper is colliding with the point cloud!", count, "times", "out of", cloud.size(), "points")
    #             print(f"cost: {cost}")
    #     return cost
    
    n_GC = X_GW.rotation().multiply(cloud.normals()[:,indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0*X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= 5*np.sum(n_GC[0,:]**2)
    if verbose:
        print(f"cost: {cost}")
        print(f"normal terms: {n_GC[0,:]**2}")

    # print("Point", target_xyz, "NormaL", target_normal)
    # print(cost)
    return cost

def GenerateAntipodalGraspCandidate(diagram,
                                    context,
                                    cloud,
                                    rng,
                                    wsg_body_index=None,
                                    plant_system_name="plant",
                                    scene_graph_system_name="scene_graph"):
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
        point: The xyz of the targeted point in the cloud
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

    def normal_up(n_WS):
        return abs(n_WS[2]) > abs(n_WS[0]) and abs(n_WS[2]) > abs(n_WS[1])
    
    def isEdge(p_WS):
        l = []
        # print(cloud.xyzs().shape, p_WS[:, None].shape)
        arr = cloud.xyzs() - p_WS[:, None]
        # print("Arr", arr.shape)
        arr = np.linalg.norm(arr, axis=0)
        # print("Normed", arr.shape)
        arr = np.argsort(arr)
        prod = 1
        for i in range(25):
            for j in range(i+1, 25):
                norm1 = cloud.normal(arr[i]) 
                norm2 = cloud.normal(arr[j])
                prod = min(prod, np.dot(norm1, norm2))
        return prod < 0.9
    
    # Don't select top of boxes
    while normal_up(n_WS) or isEdge(p_WS):
        index = rng.integers(0, cloud.size() - 1)
        p_WS = cloud.xyz(index)
        n_WS = cloud.normal(index)

    assert np.isclose(np.linalg.norm(n_WS),
                      1.0), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y,Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None, p_WS

    Gy = y - np.dot(y,Gx)*Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll=-np.pi * 0.8
    max_roll=np.pi * 0.8
    alpha = np.array([0.5, 0.6, 0.4, 0.65, 0.35, 0.7, 0.3, 0.75, 0.25, 0.8, 0.2, 0.95, 0.05, 1.0, 0.0])
    for theta in (min_roll + (max_roll - min_roll)*alpha):
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        # Cost
        cost = GraspCandidateCost(diagram, context, cloud, p_WS, n_WS, adjust_X_G=True, verbose=False)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G, p_WS

    return np.inf, None, p_WS
