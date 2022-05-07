import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class LSTM(torch.nn.Module):
    ##TODO
    def __init__():
        todo = True
        

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')

    if MODE == "train":
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')

        # Load raw data
        UNSANCTIONED_DATA = np.load(os.path.join(DATA_DIR, "unsanctioned.npy"))
        SANCTIONED_DATA = np.load(os.path.join(DATA_DIR, "sanctioned.npy"))

        ## PROCESS DATA

        #


        
        
    elif MODE == "predict":
        ##TODO
        todo = True
