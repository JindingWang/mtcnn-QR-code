import os


MODEL_STORE_DIR = "../model_store"

LOG_DIR = "../log"

image_prefix = "/home/wjd/datasets/QRcode/photoPic"

USE_CUDA = True

mean = [0.440, 0.459, 0.447]
std = [0.213, 0.186, 0.199]

PNET_restore = True
PNET_restore_model_path = "../original_model/pnet_epoch.pt"
PNET_batch_size = 128
PNET_end_epoch = 30
PNET_lr = 0.001
PNET_step_size = 4000
PNET_gamma = 0.5
PNET_model_store_path = "../save_model"

PNET_POSTIVE_ANNO_PATH = "'../annotation/Ppos.txt"
PNET_NEGATIVE_ANNO_PATH = "../annotation/Pneg.txt"
PNET_PART_ANNO_PATH = "../annotation/Ppart.txt"
PNET_AUG_ANNO_PATH = "../annotation/Paug.txt"
PNET_TRAIN_IMGLIST_PATH = "../annotation/Pnet_anno_12.txt"


RNET_restore = True
RNET_restore_model_path = "../original_model/rnet_epoch.pt"
RNET_batch_size = 64
RNET_end_epoch = 25
RNET_lr = 0.001
RNET_step_size = 8000
RNET_gamma = 0.5
RNET_model_store_path = "../save_model"
RNET_TRAIN_IMGLIST_PATH = "../annotation/Rnet_anno_24.txt"


ONET_restore = True
ONET_restore_model_path = "../original_model/onet_epoch.pt"
ONET_batch_size = 64
ONET_end_epoch = 25
ONET_lr = 0.001
ONET_step_size = 8000
ONET_gamma = 0.5
ONET_model_store_path = "../save_model"
ONET_TRAIN_IMGLIST_PATH = "../annotation/Onet_anno_48.txt"