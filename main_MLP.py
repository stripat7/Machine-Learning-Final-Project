import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import random
import math
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)
from csv import reader

class MLP(torch.nn.Module):
    def __init__(self, input_shape, hidden_layer_width):
        """Instantiate two nn.LInear modules and assign them as member variables

        Args:
            input_shape (int): shape of input going into neural net
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super().__init__()
        
        self.l1 = torch.nn.Linear(input_shape, hidden_layer_width)
        self.l2 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l3 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l4 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l5 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        """self.l6 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l7 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l8 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l9 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l10 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l11 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l12 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l13 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l14 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l15 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l16 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)
        self.l17 = torch.nn.Linear(hidden_layer_width, hidden_layer_width)"""
        self.lf = torch.nn.Linear(hidden_layer_width, 1)


    def forward(self, x):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        x = F.relu(self.l1((x)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        """ x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = F.relu(self.l9(x))
        x = F.relu(self.l10(x))
        x = F.relu(self.l11(x))
        x = F.relu(self.l12(x))
        x = F.relu(self.l13(x))
        x = F.relu(self.l14(x))
        x = F.relu(self.l15(x))
        x = F.relu(self.l16(x))
        x = F.relu(self.l17(x))"""
        x = torch.squeeze(self.lf(x))
        return x
        

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')


    np.random.seed(42)


    ##/-----------------PROCESS DATA---------------------\##
        
    #########LOAD UNSANCTIONED DATA###############
    # open file in read mode
    with open(DATA_DIR + '/unsanctioned.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        num_years = len(header[4:])

        X_u = []
        vec = [ [] for _ in range(num_years) ]

        isNewCountry = False

        first_loop = True
        skip = False

        for row in csv_reader:
            if first_loop:
                country = row[1]
                first_loop = False

            isNewCountry = not country == (row[1])

            if(isNewCountry):
                if (not skip):
                    for i in range(num_years):
                        X_u.append(vec[i])

                vec = [ [] for _ in range(num_years) ]
                skip = False

            #/-------handle empty values-------\#
            nums = row[4:]
            for i in range(len(nums)):
                nums[i] = float(nums[i])
                
            avg_val = sum(nums) / len(nums)
            zero_count = 0

            for i in range(len(nums)):
                if nums[i] == 0:
                    nums[i] = avg_val
                    zero_count += 1

            if zero_count == len(nums):
                skip = True
            #\-------handle empty values-------/#

            country = row[1]

            for i in range(num_years):
                vec[i].append(nums[i])
        

    #########LOAD SANCTIONED DATA###############
    # open file in read mode
    with open(DATA_DIR + '/sanctioned.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        num_years = len(header[4:])

        X_s = [] # Vector to hold data for each time step

        vec = [ [] for _ in range(num_years) ]

        isNewCountry = False

        first_loop = True
        skip = False

        for row in csv_reader:
            if first_loop:
                country = row[1]
                first_loop = False

            isNewCountry = not country == (row[1])

            if(isNewCountry):
                if (not skip):
                    for i in range(num_years):
                        X_s.append(vec[i])

                vec = [ [] for _ in range(num_years) ]
                skip = False

            #/-------handle empty values-------\#
            nums = row[4:]
            for i in range(len(nums)):
                nums[i] = float(nums[i])
                
            avg_val = sum(nums) / len(nums)
            zero_count = 0

            for i in range(len(nums)):
                if nums[i] == 0:
                    nums[i] = avg_val
                    zero_count += 1

            if zero_count == len(nums):
                skip = True
            #\-------handle empty values-------/#

            country = row[1]

            for i in range(num_years):
                vec[i].append(nums[i])

    # Shuffle rows of unsanctioned country matrices with same order
    n = len(X_u)
    shuffler = np.random.permutation(n)

    arr = np.array(X_u)
    arr_shuffled = arr[shuffler]
    X_u_shuff = arr_shuffled.tolist()

    # Shuffle rows of sanctioned country matrices with same order
    n = len(X_s)
    shuffler = np.random.permutation(n)

    arr = np.array(X_s)
    arr_shuffled = arr[shuffler]
    X_s_shuff = arr_shuffled.tolist()

    #Split unsanctioned countries by 80% train, 10% dev, 10% test
    n = len(X_u)

    num_train = math.floor(0.8* n)
    num_dev = math.floor(0.5 * (n-num_train))
    num_test = n - num_dev - num_train

    train_data = X_u_shuff[0:num_train]
    dev_data = X_u_shuff[num_train:num_train+num_dev]
    test_data = X_u_shuff[num_train+num_dev:] + X_s_shuff

    ##\-----------------PROCESS DATA---------------------/##

    #train_data is NUM_YEARS x NUM_TRAIN_COUNTRIES x NUM_FEATURES
    #dev_data is NUM_YEARS x NUM_DEV_COUNTRIES x NUM_FEATURES
    #test_data is NUM_YEARS x NUM_TEST_COUNTRIES x NUM_FEATURES
    #Separate data into x and y pairs where y is the output
    x_train = []
    y_train = []
    for data in train_data:
        y_train.append(data[0])
        x_train.append(data[1:])
    x_train =  np.array(x_train)
    y_train = np.array(y_train)

    x_dev = []
    y_dev = []
    for data in dev_data:
        y_dev.append(data[0])
        x_dev.append(data[1:])
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    
    x_test = []
    y_test = []
    for data in test_data:
        y_test.append(data[0])
        x_test.append(data[1:])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    ### TODO Normalize each of the datasets in training, validation, and test datasets to a mean of 0; variance of 1.
    meanT = np.mean(x_train, axis = 0)
    meanD = np.mean(x_dev, axis = 0)
    meanP = np.mean(x_test, axis = 0)

    stdT = np.std(x_train, axis = 0)
    stdD = np.std(x_dev, axis = 0)
    stdP = np.std(x_test, axis = 0)
    
    x_train =  x_train.astype(np.float32) - meanT
    x_dev = x_dev.astype(np.float32) - meanD
    x_test = x_test.astype(np.float32) - meanP
    

    x_train = x_train.astype(np.float32) / stdT
    x_dev = x_dev.astype(np.float32) / stdD
    x_test = x_test.astype(np.float32) / stdP
    
    
    meanT = np.mean(y_train, axis = 0)
    meanD = np.mean(y_dev, axis = 0)
    meanP = np.mean(y_test, axis = 0)

    stdT = np.std(y_train, axis = 0)
    stdD = np.std(y_dev, axis = 0)
    stdP = np.std(y_test, axis = 0)
    
    y_train =  y_train.astype(np.float32) - meanT
    y_dev = y_dev.astype(np.float32) - meanD
    y_test = y_test.astype(np.float32) - meanP
    

    y_train = y_train.astype(np.float32) / stdT
    y_dev = y_dev.astype(np.float32) / stdD
    y_test = y_test.astype(np.float32) / stdP
    
    
    

    if MODE == "train":
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
        
        SHAPE = len(x_train[0])

        # write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.
        LOGFILE = open(os.path.join(LOG_DIR, f"MLP.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        model = MLP(input_shape = SHAPE, hidden_layer_width = 14)

        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
        dat = np.zeros((int(EPOCHS/100), 4))
        
        for step in range(EPOCHS):
            i = np.random.choice(x_train.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(x_train[i].astype(np.float32))
            y = torch.from_numpy(y_train[i].astype(np.float32))
            
            
            # Forward pass: Get value for x
            value = model(x)
            # Compute loss
            loss = F.mse_loss(value, y)
            """print("value is:", value)
            print("y is:", y)
            print("loss is:", loss)"""
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model, x_train, y_train)
                dev_acc, dev_loss = dev_acc_and_loss(model, x_dev, y_dev)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }
                dat[int(step/100), 0] = train_loss
                dat[int(step/100), 1] = train_acc
                dat[int(step/100), 2] = dev_loss
                dat[int(step/100), 3] = dev_acc

                print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev loss is {dev_loss}")
                logger.writerow(step_metrics)
       
        
        np.savetxt("Training_data_MLP.csv", dat, fmt="%f")
        LOGFILE.close()

        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_densenet.pt" > "densenet.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"MLP.pt")
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)

    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        THRESHOLD = 0.2
 
        model = torch.load(WEIGHTS_FILE)
 
 
        model.eval() # eval mode
        
        y_pred = model(torch.from_numpy(x_test.astype(np.float32)))
        
        actual = torch.Tensor(y_test.astype(np.float32))
 
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        
        
        
        predictions = []
        for i in range(len(actual)):
            loss = (y_pred[i] - actual[i])**2
            reliable = 1
            if (loss > THRESHOLD):
                reliable = 0
 
            predictions.append(reliable)
        
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%f")
        np.savetxt("RNN_actual.csv", y_test, fmt="%f")
        np.savetxt("RNN_testvals.csv", x_test, fmt="%f")
        
    else: raise Exception("Mode not recognized")
