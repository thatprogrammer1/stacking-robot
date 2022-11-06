

from pydrake.all import (DiagramBuilder, MeshcatVisualizer, PortSwitch)

from manipulation.scenarios import (
    AddIiwaDifferentialIK, MakeManipulationStation)
from planning.planner import Planner
from grasp.grasp_selector import GraspSelector


def BuildStackingDiagram(meshcat):
    builder = DiagramBuilder()

    model_directives = """
directives:
- add_directives:
    file: package://manipulation/clutter_w_cameras.dmd.yaml
"""

    for i in range(3):
        model_directives += f"""
- add_model:
    name: brick{i}
    file: package://drake/examples/manipulation_station/models/061_foam_brick.sdf
"""

    station = builder.AddSystem(
        MakeManipulationStation(model_directives, time_step=0.001))
    plant = station.GetSubsystemByName("plant")

    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin0"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera0"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera1"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera2"))[0]
                      ]))
    builder.Connect(station.GetOutputPort("camera0_point_cloud"),
                    y_bin_grasp_selector.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera1_point_cloud"),
                    y_bin_grasp_selector.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera2_point_cloud"),
                    y_bin_grasp_selector.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    y_bin_grasp_selector.GetInputPort("body_poses"))

    x_bin_grasp_selector = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin1"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera3"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera4"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera5"))[0]
                      ]))
    builder.Connect(station.GetOutputPort("camera3_point_cloud"),
                    x_bin_grasp_selector.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera4_point_cloud"),
                    x_bin_grasp_selector.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera5_point_cloud"),
                    x_bin_grasp_selector.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    x_bin_grasp_selector.GetInputPort("body_poses"))

    planner = builder.AddSystem(Planner(plant))
    builder.Connect(station.GetOutputPort("body_poses"),
                    planner.GetInputPort("body_poses"))
    builder.Connect(x_bin_grasp_selector.get_output_port(),
                    planner.GetInputPort("x_bin_grasp"))
    builder.Connect(y_bin_grasp_selector.get_output_port(),
                    planner.GetInputPort("y_bin_grasp"))
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
