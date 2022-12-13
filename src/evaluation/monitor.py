
from collections import namedtuple
import numpy as np
from pydrake.all import (AbstractValue, LeafSystem, RigidTransform)
from manipulation.meshcat_utils import AddMeshcatTriad
import os
import sys
import warnings
import numpy as np
from pydrake.all import (
    AbstractValue, Adder, AddMultibodyPlantSceneGraph, BallRpyJoint, BaseField,
    Box, CameraInfo, ClippingRange, CoulombFriction, Cylinder, Demultiplexer,
    DepthImageToPointCloud, DepthRange, DepthRenderCamera, DiagramBuilder,
    FindResourceOrThrow, GeometryInstance, InverseDynamicsController,
    LeafSystem, LoadModelDirectivesFromString,
    MakeMultibodyStateToWsgStateSystem, MakePhongIllustrationProperties,
    MakeRenderEngineVtk, ModelInstanceIndex, MultibodyPlant, Parser,
    PassThrough, PrismaticJoint, ProcessModelDirectives, RenderCameraCore,
    RenderEngineVtkParams, RevoluteJoint, Rgba, RgbdSensor, RigidTransform,
    RollPitchYaw, RotationMatrix, SchunkWsgPositionController, SpatialInertia,
    Sphere, StateInterpolatorWithDiscreteDerivative, UnitInertia, Role, RenderLabel)

CameraPorts = namedtuple('CameraPorts', 'cloud_index, label_index')

kBrickPrefix = "brick"
kPentagonPrefix = "pentagon"


class Monitor(LeafSystem):
    """
    Collect stats about the simulation
    """

    def __init__(self, meshcat, plant):
        LeafSystem.__init__(self)
        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])).get_index()
        self._is_home_index = self.DeclareAbstractInputPort(
            "is_home", AbstractValue.Make(False)).get_index()
        self.DeclareAbstractOutputPort(
            "stats",
            lambda: AbstractValue.Make({}),
            self.CalcOutput)
        self._meshcat = meshcat
        self._plant = plant

    def CalcOutput(self, context, output):
        poses = self.get_input_port(self._body_poses_index).Eval(context)
        is_home = self.get_input_port(self._is_home_index).Eval(context)
        plant = self._plant
        prisms = []
        for i in range(plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            model_instance_name = plant.GetModelInstanceName(model_instance)
            if not (model_instance_name.startswith(kBrickPrefix)
                    or model_instance_name.startswith(kPentagonPrefix)):
                continue
            body_ind = plant.GetBodyIndices(model_instance)[0]
            prisms.append(poses[body_ind])
            # AddMeshcatTriad(self._meshcat, model_instance_name,
            #                 X_PT=poses[body_ind])
        max_stacked = 0

        for ind, prism1 in enumerate(prisms):
            if not is_home:
                continue
            cnt = 1
            for ind2, prism2 in enumerate(prisms):
                if ind == ind2:
                    continue
                # if x y pos is similar and something >= abs(z1-z2)>=something
                # and is not placing,
                # then assume they are stacked on top of each other
                p2 = prism2.translation()
                p1 = prism1.translation()
                if np.linalg.norm(p2[:2] - p1[:2]) <= 0.06:
                    cnt += 1
            max_stacked = max(cnt, max_stacked)

        output.set_value({"max_stacked": max_stacked})
