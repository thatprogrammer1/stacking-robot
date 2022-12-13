
from dataclasses import dataclass
import logging

import numpy as np
from pydrake.all import (RandomGenerator, RigidTransform,
                         Simulator, UniformlyRandomRotationMatrix, StartMeshcat)

from setup_diagram import BuildStackingDiagram, PrismConfig


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('Differential IK')


logging.getLogger("drake").addFilter(NoDiffIKWarnings())


def stacking_demo(prism_config: PrismConfig, max_stacked_trials, trial_num=0,  seed=None):
    meshcat.Delete()
    if seed == None:
        seed = np.random.randint(10000000)
    # Good seeds,  7079183
    print("Using random seed: ", seed)
    rs = np.random.RandomState(seed)  # this is for python
    generator = RandomGenerator(rs.randint(1000))  # this is for c++
    diagram, plant, visualizer = BuildStackingDiagram(
        meshcat, seed, prism_config)

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
            [rs.uniform(.5, 0.7), rs.uniform(-.18, .2), z])
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
    # print("Press Escape to stop the simulation")
    max_stacked = 1
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if simulator.get_context().get_time() > 60 * sum(prism_config):
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        stats = diagram.get_output_port().Eval(simulator.get_context())
        max_stacked = max(stats["max_stacked"], max_stacked)
        max_stacked_trials[trial_num] = max_stacked
        if stats["max_stacked"] == sum(prism_config):
            return 1
        visualizer.StopRecording()
    visualizer.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    return 0


meshcat = StartMeshcat()


def record_success_rate():
    cnt = 0
    max_stacked_trials = [0]*100
    for i in range(100):
        try:
            cnt += stacking_demo(PrismConfig(0, 2, 0), max_stacked_trials, i)
        except Exception as e:
            print("Exception occured: ", e)
        print(f"Success rate: {cnt}/{i+1}")
        print("Max stacked:", max_stacked_trials[:i+1])


if __name__ == "__main__":
    # stacking_demo(PrismConfig(0, 2, 0))
    record_success_rate()
