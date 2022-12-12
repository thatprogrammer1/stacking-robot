import os
import numpy as np
from perception.color_segmentation import ColorSegmentation
from perception.merge_point_clouds import MergePointClouds
from planning.stack_detector import StackDetector
from planning.stacking_planner import StackingPlanner
from planning.com_solver import COMProblem
from pydrake.all import (DiagramBuilder,
                         MeshcatVisualizer, MeshcatVisualizerParams, Role, PortSwitch, Box, PolygonSurfaceMesh, RigidTransform, RotationMatrix, AddMultibodyPlantSceneGraph, SpatialInertia, UnitInertia, CoulombFriction)
from manipulation.scenarios import (
    AddIiwaDifferentialIK)
from scenarios import MakeManipulationStation
from perception.grasp_selector import GraspSelector
from pydrake.all import LeafSystem
import random

# Simulation tuning pset code


def AddBox(plant, shape, name, mass=1, mu=10, color=[.5, .5, .9, 1.0], pose=RigidTransform()):
    instance = plant.AddModelInstance(name)
    inertia = UnitInertia.SolidBox(shape.width(), shape.depth(),
                                   shape.height())
    body = plant.AddRigidBody(
        name, instance,
        SpatialInertia(mass=mass,
                       p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=inertia))
    plant.RegisterCollisionGeometry(body, pose, shape, name,
                                    CoulombFriction(mu, mu))
    plant.RegisterVisualGeometry(body, pose, shape, name, color)


def AddPrism(plant, points_2d, height, name, mass=1, mu=10, color=[.5, .5, .9, 1.0], pose=RigidTransform()):
    # this code is wrong
    # polygonsurfacemesh is not a shape
    instance = plant.AddModelInstance(name)
    inertia = UnitInertia.SolidBox(1., 1., 1.)
    N = points_2d.shape[0]
    points_3d = np.concatenate((np.hstack((points_2d, np.zeros((N, 1)))), np.hstack(
        (points_2d, np.ones((N, 1)) * height))))
    face_data = []
    face_data += [N] + list(range(N))
    face_data += [N] + list(range(N, 2 * N))
    for v in range(N):
        next_v = (v + 1) % N
        face_data += [4] + [v, next_v, v + N, next_v + N]
    shape = PolygonSurfaceMesh(np.array(face_data), points_3d)
    body = plant.AddRigidBody(name, instance, SpatialInertia(
        mass=mass, p_PScm_E=np.array([0., 0., 0.]), G_SP_E=inertia))
    plant.RegisterCollisionGeometry(
        body, pose, shape, name, CoulombFriction(mu, mu))
    plant.RegisterVisualGeometry(body, pose, shape, name, color)


def GetStation():
    """
    Create a manipulation station with our own model package loaded.
    Modify model directives to change what models we include.
    """
    model_directives = """
directives:
- add_directives:
    file: package://stacking/clutter_w_cameras.dmd.yaml
"""

    colors = ["blue", "green", "cyan", "red", "yellow"]

    for i in range(2):
        model_directives += f"""
- add_model:
    name: brick{i}
    file: package://drake/examples/manipulation_station/models/061_foam_brick.sdf
"""
    
    for i in range(2):
            model_directives += f"""
- add_model:
    name: pentagon{i}
    file: package://stacking/pent/pent_{colors[i]}.sdf
"""


    def callback(plant):
        pass
        # box = Box(0.06, 0.06, 0.1)
        # for i in range(0):
        #     AddBox(plant, box, f"box{i}", color=[0.6, 0.3, 0.2, 1.0])
        # for i in range(1):
        #     x, y = random.uniform(xStart, xEnd), random.uniform(yStart, yEnd)
        #     print("Placing prism at", x, y)
        #     AddPrism(plant, np.array([[0, 1], [1, 0], [0, 0]]),
        #              1, f"prism{i}", color=[0.6, 0.3, 0.2, 1.0])

    return MakeManipulationStation(callback,
                                   model_directives=model_directives,
                                   time_step=0.001,
                                   package_xmls=[os.path.join(os.path.dirname(
                                       os.path.realpath(__file__)), "models/package.xml")],
                                   disable_cheat_segmentation=True
                                   )


class StaticController(LeafSystem):
    """
    For some reason WSG input port is required, so this system sends a 0 
    to the wsg. For testing other systems.
    """

    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareVectorOutputPort(
            "wsg_position", 1, self.CalcWsgPosition)

    def CalcWsgPosition(self, context, output):
        output.SetFromVector([np.array([0.0])])


def BuildStaticDiagram(meshcat):
    """
    Builds a diagram with no planning (there's probably a better way to test things, but idk)
    """
    builder = DiagramBuilder()

    station = builder.AddSystem(GetStation())
    plant = station.GetSubsystemByName("plant")
    planner = builder.AddSystem(StaticController(plant))
    builder.Connect(planner.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    meshcat_param = MeshcatVisualizerParams()
    """ kProximity for collision geometry and kIllustration for visual geometry """
    meshcat_param.role = Role.kIllustration
    meshcat_param.role = Role.kProximity

    MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat, meshcat_param)

    return builder.Build(), plant


def BuildStackingDiagram(meshcat, seed):
    builder = DiagramBuilder()
    manip_station = GetStation()
    station = builder.AddSystem(manip_station)
    plant = station.GetSubsystemByName("plant")

    merge_point_clouds = builder.AddSystem(
        MergePointClouds(plant,
                         plant.GetModelInstanceByName("bin0"),
                         camera_body_indices=[
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera0"))[0],
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera1"))[0],
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera2"))[0]
                         ],
                         meshcat=meshcat))
    for i in range(3):
        point_cloud_port = f"camera{i}_point_cloud"
        # label_image_port = f"camera{i}_label_image"
        builder.Connect(station.GetOutputPort(point_cloud_port),
                        merge_point_clouds.GetInputPort(point_cloud_port))
        # builder.Connect(station.GetOutputPort(label_image_port), merge_point_clouds.GetInputPort(label_image_port))
    builder.Connect(station.GetOutputPort("body_poses"),
                    merge_point_clouds.GetInputPort("body_poses"))

    color_segmentation = builder.AddSystem(ColorSegmentation())
    builder.Connect(merge_point_clouds.GetOutputPort("point_cloud"),
                    color_segmentation.GetInputPort("point_cloud"))
    grasp_selector = builder.AddSystem(
        GraspSelector(stacking_zone_center=np.array(
            [.6, .2]), stacking_zone_radius=.07, meshcat=meshcat, random_seed=seed)
    )
    builder.Connect(merge_point_clouds.GetOutputPort("point_cloud"),
                    grasp_selector.GetInputPort("point_cloud"))
    builder.Connect(color_segmentation.GetOutputPort("segmented_clouds"),
                    grasp_selector.GetInputPort("segmented_clouds"))

    # TODO (khm): add stack detector, wire planner to use its output to figure out where to place next
    detector = builder.AddSystem(StackDetector(
        stacking_zone_center=np.array([.6, .2]), stacking_zone_radius=.07, meshcat=meshcat))
    builder.Connect(merge_point_clouds.GetOutputPort("point_cloud"),
                    detector.GetInputPort("merged_pcd"))
    planner = builder.AddSystem(StackingPlanner(
        plant, meshcat, np.array([0.3, 0.0, 0.2])))
    builder.Connect(detector.GetOutputPort("next_stack_position"),
                    planner.GetInputPort("stack_position"))
    builder.Connect(station.GetOutputPort("body_poses"),
                    planner.GetInputPort("body_poses"))
    builder.Connect(grasp_selector.GetOutputPort("grasp_selection"),
                    planner.GetInputPort("grasp"))
    builder.Connect(grasp_selector.GetOutputPort("clutter_grasp_selection"),
                    planner.GetInputPort("grasp_clutter"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    planner.GetInputPort("wsg_state"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    planner.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                    planner.GetInputPort("external_torque"))

    robot = station.GetSubsystemByName(
        "iiwa_controller").get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                    diff_ik.GetInputPort("robot_state"))
    builder.Connect(planner.GetOutputPort("reset_diff_ik"),
                    diff_ik.GetInputPort("use_robot_state"))

    builder.Connect(planner.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(),
                    switch.DeclareInputPort("diff_ik"))
    builder.Connect(planner.GetOutputPort("iiwa_position_command"),
                    switch.DeclareInputPort("position"))
    builder.Connect(switch.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(planner.GetOutputPort("control_mode"),
                    switch.get_port_selector_input_port())

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    return builder.Build(), plant, visualizer


def visualize_diagram(diagram):
    """
    Util to visualize the system diagram
    """
    from IPython.display import SVG, display
    import pydot
    display(SVG(pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2))[0].create_svg()))
