import easysim

from yacs.config import CfgNode as CN

_C = easysim.cfg

# ---------------------------------------------------------------------------- #
# Simulation config
# ---------------------------------------------------------------------------- #
_C.SIM.USE_DEFAULT_STEP_PARAMS = False

_C.SIM.TIME_STEP = 0.001

_C.SIM.INIT_VIEWER_CAMERA_POSITION = [-0.1, 0.6, 0.45]

_C.SIM.INIT_VIEWER_CAMERA_TARGET = [-0.1, 0.0, 0.0]

# ---------------------------------------------------------------------------- #
# Environment config
# ---------------------------------------------------------------------------- #
_C.ENV = CN()

_C.ENV.TARGET_POSITION_X = -0.08047459842907007

_C.ENV.TARGET_POSITION_Y = 0.1291136198234517

_C.ENV.DISPLAY_TARGET = True

_C.ENV.RANDOM_SEED = 0

##### Table #####
_C.ENV.TABLE_LENGTH = 1.1

_C.ENV.TABLE_WIDTH = 1.5

_C.ENV.TABLE_HEIGHT = _C.SIM.GROUND_PLANE.DISTANCE

_C.ENV.TABLE_X_OFFSET = -0.3

# Notes: We need all the collisions, so all the objects should have the same collision filter
_C.ENV.COLLISION_FILTER_TABLE = 1

##### Panda #####
_C.ENV.PANDA_BASE_POSITION = [-0.6, 0.0, 0.0]

_C.ENV.PANDA_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]

_C.ENV.PANDA_INITIAL_POSITION = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.04, 0.04]

_C.ENV.COLLISION_FILTER_PANDA = 1

# Notes: These values follow the PandaPickAndPlace-V3 of panda-gym
_C.ENV.PANDA_MAX_FORCE = [87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]

_C.ENV.PANDA_POSITION_GAIN = [0.01,] * 9

_C.ENV.PANDA_VELOCITY_GAIN = [1.0,] * 9

##### Primitive Objects #####
_C.ENV.NUM_PRIMITIVE_OBJECTS = 1

_C.ENV.COLLISION_FILTER_PRIMITIVE_OBJECT = 1

_C.ENV.PRIMITIVE_OBJECT_TYPE = "Box"

_C.ENV.PRIMITIVE_OBJECT_SIZE = 0.04

_C.ENV.PRIMITIVE_OBJECT_MASS = 1.0

_C.ENV.PRIMITIVE_OBJECT_BASE_POSITION = [[0.18, 0.0, 0.02]]

_C.ENV.PICKED_OBJECT_IDX = 0

##### MANO Hand #####
_C.ENV.MANO_MODEL_FILENAME = "gestureIL/objects/data/assets/20200709-subject-01_right/mano.urdf"

_C.ENV.MANO_POSE_FILENAME = "gestureIL/objects/data/mano_poses/1_right.npy"

_C.ENV.MANO_INITIAL_TARGET = [0.043, 0.209, 0.02]

_C.ENV.MANO_INITIAL_BASE = [0.092, 0.239, 0.028]

_C.ENV.MANO_FINAL_TARGET = [
    _C.ENV.TARGET_POSITION_X,
    _C.ENV.TARGET_POSITION_Y,
    _C.ENV.PRIMITIVE_OBJECT_SIZE / 2
]

_C.ENV.MANO_FINAL_BASE = [
    _C.ENV.MANO_FINAL_TARGET[0] + 0.1,
    _C.ENV.MANO_FINAL_TARGET[1],
    _C.ENV.PRIMITIVE_OBJECT_SIZE + 0.05
]
    
_C.ENV.COLLISION_FILTER_MANO = 0

_C.ENV.MANO_TRANSLATION_MAX_FORCE = [50.0,] * 3

_C.ENV.MANO_TRANSLATION_POSITION_GAIN = [0.2,] * 3

_C.ENV.MANO_TRANSLATION_VELOCITY_GAIN = [10.0,] * 3

_C.ENV.MANO_ROTATION_MAX_FORCE = [5.0,] * 3

_C.ENV.MANO_ROTATION_POSITION_GAIN = [0.2,] * 3

_C.ENV.MANO_ROTATION_VELOCITY_GAIN = [1.0,] * 3

_C.ENV.MANO_JOINT_MAX_FORCE = [0.5,] * 45

_C.ENV.MANO_JOINT_POSITION_GAIN = [0.1,] * 45

_C.ENV.MANO_JOINT_VELOCITY_GAIN = [1.0,] * 45

##### YCB Objects #####

##### Offscreen Rendering #####
_C.ENV.RENDER_OFFSCREEN = False

_C.ENV.RENDER_DIR = f"datasets/{_C.ENV.PICKED_OBJECT_IDX}/demonstrations"

_C.ENV.NUM_OFFSCREEN_RENDERER_CAMERA = 2

_C.ENV.OFFSCREEN_RENDERER_CAMERA_WIDTH = 224

_C.ENV.OFFSCREEN_RENDERER_CAMERA_HEIGHT = 224

_C.ENV.OFFSCREEN_RENDERER_CAMERA_VERTICAL_FOV = 60.0

_C.ENV.OFFSCREEN_RENDERER_CAMERA_NEAR = 0.1

_C.ENV.OFFSCREEN_RENDERER_CAMERA_FAR = 10.0

_C.ENV.OFFSCREEN_RENDERER_CAMERA_POSITION = [[0.45, 0.0, 0.6], [0.0, 0.45, 0.6]]

_C.ENV.OFFSCREEN_RENDERER_CAMERA_TARGET = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


get_cfg = easysim.get_cfg

get_config_from_args = easysim.get_config_from_args