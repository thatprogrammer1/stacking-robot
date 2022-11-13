import os
import numpy as np
from grasp.segmentation import Segmentation
from planning.stacking_planner import StackingPlanner
from pydrake.all import (DiagramBuilder,
                         MeshcatVisualizer, MeshcatVisualizerParams, Role, PortSwitch, Box, RigidTransform, RotationMatrix, AddMultibodyPlantSceneGraph, SpatialInertia, UnitInertia, CoulombFriction)
from manipulation.scenarios import (
    AddIiwaDifferentialIK)
from scenarios import MakeManipulationStation
from grasp.grasp_selector import GraspSelector
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

    for i in range(2):
        model_directives += f"""
- add_model:
    name: brick{i}
    file: package://drake/examples/manipulation_station/models/061_foam_brick.sdf
"""

    def callback(plant):
        xSize, ySize = 0.3, 0.3
        xStart, yStart = 0.2, -0.2
        xEnd, yEnd = xStart+xSize, yStart+ySize

        box = Box(0.06, 0.06, 0.1)
        for i in range(1):
            x, y = random.uniform(xStart, xEnd), random.uniform(yStart, yEnd)
            print("Placing box at", x, y)
            X_WBox = RigidTransform(RotationMatrix(), [x, y, 0.1])
            AddBox(plant, box, f"box{i}", color=[
                   0.6, 0.3, 0.2, 1.0])

    return MakeManipulationStation(callback, model_directives, time_step=0.001, package_xmls=[os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/package.xml")])


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


def BuildStackingDiagram(meshcat):
    builder = DiagramBuilder()

    station = builder.AddSystem(GetStation())
    plant = station.GetSubsystemByName("plant")

    segmentation = builder.AddSystem(
        Segmentation(plant,
                     plant.GetModelInstanceByName("bin0"),
                     camera_body_indices=[
                         plant.GetBodyIndices(
                             plant.GetModelInstanceByName("camera0"))[0],
                         plant.GetBodyIndices(
                             plant.GetModelInstanceByName("camera1"))[0],
                         plant.GetBodyIndices(
                             plant.GetModelInstanceByName("camera2"))[0]
                     ]))
    for i in range(3):
        point_cloud_port = f"camera{i}_point_cloud"
        label_image_port = f"camera{i}_label_image"
        builder.Connect(station.GetOutputPort(point_cloud_port),
                        segmentation.GetInputPort(point_cloud_port))
        builder.Connect(station.GetOutputPort(label_image_port),
                        segmentation.GetInputPort(label_image_port))
    builder.Connect(station.GetOutputPort("body_poses"),
                    segmentation.GetInputPort("body_poses"))

    grasp_selector = builder.AddSystem(GraspSelector())
    builder.Connect(segmentation.GetOutputPort("point_cloud"),
                    grasp_selector.GetInputPort("point_cloud"))

    planner = builder.AddSystem(StackingPlanner(plant, meshcat))
    builder.Connect(station.GetOutputPort("body_poses"),
                    planner.GetInputPort("body_poses"))
    builder.Connect(grasp_selector.get_output_port(),
                    planner.GetInputPort("grasp"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    planner.GetInputPort("wsg_state"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    planner.GetInputPort("iiwa_position"))

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

    MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    return builder.Build(), plant


def visualize_diagram(diagram):
    """
    Util to visualize the system diagram
    """
    from IPython.display import SVG, display
    import pydot
    display(SVG(pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2))[0].create_svg()))
