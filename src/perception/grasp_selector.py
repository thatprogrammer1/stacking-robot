
from typing import List
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, AddMultibodyPlantSceneGraph, Concatenate, DiagramBuilder, LeafSystem, Parser,
                         PointCloud,
                         RigidTransform,
                         RollPitchYaw, ImageLabel16I, Fields, BaseField)

from manipulation import FindResource, running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
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


CameraPorts = namedtuple('CameraPorts', 'cloud_index, label_index')


class GraspSelector(LeafSystem):
    def __init__(self, random_seed=None):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0, fields=Fields(
            BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals)))

        self._point_cloud_index = self.DeclareAbstractInputPort(
            "point_cloud", model_point_cloud).get_index()

        port = self.DeclareAbstractOutputPort(
            "grasp_selection", lambda: AbstractValue.Make(
                (np.inf, RigidTransform())), self.SelectGrasp)
        port.disable_caching_by_default()

        self._internal_model = make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng(random_seed)

    def SelectGrasp(self, context, output):
        down_sampled_pcd = self.get_input_port(
            self._point_cloud_index).Eval(context)
        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(100 if running_as_notebook else 2):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model, self._internal_model_context,
                down_sampled_pcd, self._rng)
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
