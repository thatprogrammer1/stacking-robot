
import logging

import numpy as np
from pydrake.all import (RandomGenerator, RigidTransform,
                         Simulator, UniformlyRandomRotationMatrix, StartMeshcat)

from setup_diagram import BuildStackingDiagram


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('Differential IK')


logging.getLogger("drake").addFilter(NoDiffIKWarnings())


def clutter_clearing_demo(seed=None):
    meshcat.Delete()
    if seed == None:
        seed = np.random.randint(10000000)
    # Good seeds,  7079183
    print("Using random seed: ", seed)
    rs = np.random.RandomState(seed)  # this is for python
    generator = RandomGenerator(rs.randint(1000))  # this is for c++
    diagram, plant, visualizer = BuildStackingDiagram(meshcat, seed)

    # for idx in plant.GetJointIndices(plant.GetModelInstanceByName("iiwa")):
    #     print(plant.get_joint(idx).name())
    #     print(plant.get_joint(idx).parent_body().name())
    #     print(plant.get_joint(idx).child_body().name())
    #     if plant.get_joint(idx).name().startswith("iiwa_joint_"):
    #         print(plant.get_joint(idx).revolute_axis())
    # joint = plant.GetJointByName("iiwa_joint_1")
    # print(joint.parent_body().name())

    simulator = Simulator(diagram)
    context = simulator.get_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    z = .2
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform(
            UniformlyRandomRotationMatrix(generator),
            [rs.uniform(.5, 0.7), rs.uniform(-.2, .3), z])
        plant.SetFreeBodyPose(plant_context,
                              plant.get_body(body_index),
                              tf)
        z += .1

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    visualizer.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        print(diagram.get_output_port().Eval(simulator.get_context()))
    visualizer.StopRecording()
    visualizer.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")


meshcat = StartMeshcat()
clutter_clearing_demo()
