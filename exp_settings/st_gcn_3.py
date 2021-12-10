# Flags
COPY_DATA = False
ON_PREMISE_LOCATION = None
RELOAD_CHECKPOINT = False
IS_CUDA = True
TEST_RESULTS_DUMP_DEBUG_NUM = 100
TEST_RESULTS_DUMP_HEATMAP = False
RELOAD_CHECKPOINT_PATH = None

RELOAD_DICT_LIST = ["model_dict", "metric"]
# Paths
DB_PATH = "D:/workspace/datasets/COPDGene/v3/copdgene220/"
TEST_CSV = "D:/workspace/datasets/COPDGene/v3/copdgene220/meta_scans.csv"
TEST_DB_PATH = "D:/workspace/datasets/COPDGene/v3/copdgene220/copdgene3_test/"
TRAIN_CSV = "D:/workspace/datasets/COPDGene/v3/copdgene220/meta_scans.csv"
VALID_CSV = "D:/workspace/datasets/COPDGene/v3/copdgene220/meta_scans.csv"
DEBUG_PATH = "D:/workspace/test_cases/al/"
MODEL_ROOT_PATH = "D:/workspace/models/"


JOB_RUNNER_CLS = "job_runner.GCNTrain"
TEST_RUNNER_CLS = "job_runner.GCNTest"
EXP_NAME = "gcn_3"


AUG_RATIO = 0.0

SAVE_EPOCHS = 50
GCN_STEPS = 300
NUM_EPOCHS = 151

NUM_WORKERS = 0
LOG_STEPS = 5
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
GRAPH_MODE = "all_connected"
TRAIN_SAMPLE_SIZE = 100

RELABEL_MAPPING = { 
    2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11: 11, 12:12, 13:13,
    14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22
}
LABEL_NAME_MAPPING = {0: 'background',
                      1: 'rest',
                      2: 'RB1',
                      3: 'RB2',
                      4: 'RB3',
                      5: 'RB4',
                      6: 'RB5',
                      7: 'RB6',
                      8: 'RB7',
                      9: 'RB8',
                      10: 'RB9',
                      11: 'RB10',
                      12: 'LB1+2',
                      14: 'LB3',
                      15: 'LB4',
                      16: 'LB5',
                      17: 'LB6',
                      18: 'LB7+8',
                      20: 'LB9',
                      21: 'LB10',
                      }
CLASS_WEIGHTS = {0:0.1, 1:0.2, 2:0.8, 3:0.8,
                 4:0.8, 5:0.8, 6:0.8, 7:0.8,
                 8:0.8, 9:0.8, 10:0.8, 11:0.8, 12:0.8, 13:0.8, 14:0.8, 15:0.8, 16:0.8,
                 17:0.8, 18:0.8, 19:0.8, 20:0.8, 21:0.8, 22:0.8,
                 }

# thresholds
PAD_VALUE = -2048
WINDOWING_MAX = 200
WINDOWING_MIN = -1000
NR_CLASS = 22
EVAL_NR_CLASS = 18
SAMPLING_RATE = 0.05
# model settings.
MODEL = {
    "method": "models.GCNNet",
    "n_layers": 3,

    "in_ch_list": [1, 32, 64, 128],
    "base_ch_list": [24, 32, 64, 128],
    "end_ch_list": [32, 64, 128, 256],

    "kernel_sizes": [3, 3, 3, 3],
    "checkpoint_layers": [0, 1, 1, 0, 1, 1, 1],
    'out_ch': NR_CLASS,
    "padding_list": [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
    "conv_strides": [[1, 2], [1, 2], [1, 2]],
    "dropout": 0.0,
    "spatial_size": 10,
    "norm_method": "bn",
    "act_method": "relu",
    "fv_dim": 1024,
    "num_gcn_layers": 3,

    "node_embed_dim": 1024,

    "num_hiddens": [256, 128, 64],

}

INITIALIZER = {
    "method": "initializer.HeNorm",
    "mode": "fan_in"
}


OPTIMIZER = {
    "method": "torch.optim.SGD",
    "momentum": 0.9,
    "lr": 0.0001,
}

SCHEDULER = {
    "method": "torch.optim.lr_scheduler.ExponentialLR",
    "gamma": 0.9
}


LOSS_FUNC = {
    "method": "torch.nn.CrossEntropyLoss",
}



# loggers.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

PROCESSOR_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/processor_info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization
INSPECT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "{}/{}/inspect_info.log".format(MODEL_ROOT_PATH, EXP_NAME),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization


VISUALIZATION_COLOR_TABLE = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (100, 0, 0),
    (100, 100, 0),
    (100, 100, 100),
    (50, 200, 0),
    (50, 200, 200),
    (50, 50, 200),
    (200, 50, 200),
    (50, 200, 50),
]

VISUALIZATION_ALPHA = 0.6
VISUALIZATION_SPARSENESS = 6
VISUALIZATION_PORT = 6012

CRF_PARAM = {
    "sxyz": 15,
    "srgb": 10,
    "comp_bi": 8,
    "comp_gaussian": 6,
    "iteration": 2
}

INSPECT_PARAMETERS = {
    "watch_layers": {
        "unet1.bg": {"input": True, "stride": 1},
        "unet1.non_local_module": {"input": False, "stride": 1},
        "unet2.bg": {"input": False, "stride": 1},
        "unet2.non_local_module": {"input": False, "stride": 1}
    },
}
