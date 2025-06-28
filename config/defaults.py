from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.FIRST = ''

_C.ASYM = CN()
_C.ASYM.PULL = 1.0

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.GRC = 0.0
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.FT_NUM = 10000
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
_C.DATALOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.LORA_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULER = 'cosine'
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.LUP_BATCH = 64
_C.SOLVER.SWA = False
_C.SOLVER.SWA_EPOCH = 10
_C.SOLVER.SWA_START = 100
_C.SOLVER.SWA_LR = 3e-6
_C.SOLVER.SWA_FREQ = 1
_C.SOLVER.NUM_SAMPLE = 1000000

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""


# ---------------------------------------------------------------------------- #
# TARGET
# ---------------------------------------------------------------------------- #
_C.TGT_LOSS = CN()
_C.TGT_LOSS.METRIC_LOSS_TYPE = 'triplet'
_C.TGT_LOSS.NO_MARGIN = False
_C.TGT_LOSS.MARGIN = 0.3
_C.TGT_LOSS.AUG = 0.5
_C.TGT_LOSS.AUG_DETACH = False
_C.TGT_LOSS.A = 1.0
_C.TGT_LOSS.REVERSE = 0.0
_C.TGT_LOSS.R_START = 0
_C.TGT_LOSS.ATTN = 0.0

_C.TGT_LOSS.PROGRAD = 0.8

_C.TGT_LOSS.CAM = 0.0
_C.TGT_LOSS.ID = 0.0
_C.TGT_LOSS.DIST = 0.0
_C.TGT_LOSS.CAM_NO_MARGIN = False
_C.TGT_LOSS.CAM_MARGIN = 0.5
_C.TGT_LOSS.CAM_AUG = 0.5
_C.TGT_LOSS.CAM_AUG_DETACH = False

_C.TGT_LOSS.MEMORY = False
_C.TGT_LOSS.MEMORY_SIZE = 0
_C.TGT_LOSS.NT_MEMORY_SIZE = 0
_C.TGT_LOSS.MEMORY_M = 0.25
_C.TGT_LOSS.MEMORY_GAMMA = 128
_C.TGT_LOSS.MEMORY_CAM_M = 0.25
_C.TGT_LOSS.MEMORY_CAM_GAMMA = 128
_C.TGT_LOSS.REG_WEIGHT = 1.0
_C.TGT_LOSS.RKD = True
_C.TGT_LOSS.STYLE_MARGIN = 1.0
_C.TGT_LOSS.STYLE_MARGIN_POS = 0.0
_C.TGT_LOSS.STYLE = 0.5
_C.TGT_LOSS.STYLE_HARD = 1.0
_C.TGT_LOSS.MMD = 0.0
_C.TGT_LOSS.CORAL = 0.0
_C.TGT_LOSS.MMD_MARGIN = 1.0
_C.TGT_LOSS.CORAL_MARGIN = 1.0
_C.TGT_LOSS.BEGIN = 0

_C.TGT_LOSS.LUP = 1.0
_C.TGT_LOSS.IN_DOMAIN = 1.0

_C.TGT_DATA = CN()
_C.TGT_DATA.IMS_PER_BATCH = 128
_C.TGT_DATA.NUM_INSTANCE = 4
_C.TGT_DATA.CAMERA_SAMPLE = True
_C.TGT_DATA.MY_AUG_POOL = False
_C.TGT_DATA.APPLY_RQ = False
_C.TGT_DATA.VPD = 0.0
_C.TGT_DATA.PID = 100
_C.TGT_DATA.CAM_AUG = False

_C.PREFIX = CN()
_C.PREFIX.MEM = 0
_C.PREFIX.MEM_T = 0
_C.PREFIX.LAM = 0.0
_C.PREFIX.MEM_Q = 'identity'
_C.PREFIX.SHARE_PROJ = True
_C.PREFIX.BETA = 5.5
_C.PREFIX.NUM_ATTN = 1
_C.PREFIX.DETACH = False
_C.PREFIX.LR = 1.0
_C.PREFIX.WD = 5e-4
_C.PREFIX.KEEP_TOPK = -1
_C.PREFIX.MEM_DROP = 0.0
_C.PREFIX.HIDDEN = 768
_C.PREFIX.OUT = 768
_C.PREFIX.NUM_HEAD = 12

_C.FINE_TUNE = CN()
_C.FINE_TUNE.FREEZE_BN = True

_C.ETF = CN()
_C.ETF.WEIGHT = 0.0
_C.ETF.ADAPTER = 'bn'
_C.ETF.LR_FACTOR = 1.0
_C.ETF.BOTTLENECK = 0

_C.LORA = CN()
_C.LORA.Q = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
_C.LORA.K = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
_C.LORA.V = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
_C.LORA.PROJ = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
_C.LORA.LORA_NUMS = 2
_C.LORA.RANK = 16
_C.LORA.COS_ROUTE = False
_C.LORA.ALPHA = 1
_C.LORA.DROP = 0.0
_C.LORA.MLP_RANK = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
