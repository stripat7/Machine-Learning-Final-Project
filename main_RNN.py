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
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss, r2_score)
from csv import reader

class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        """Instantiate two nn.LInear modules and assign them as member variables

        Args:
            input_shape (int): shape of input going into neural net
            hidden_layer_width (int): number of nodes in the single hidden layer within the model
            n_classes (int): number of output classes
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        #Defining the layers
        # RNN Layer
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        """Forward function accepts tensor of input data, returns tensor of output data.
        Modules defined in constructor are used, along with arbitrary operators on tensors
        """
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return torch.squeeze(out), hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
        

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')

    ##/-----------------PROCESS DATA---------------------\##
        
    #########LOAD UNSANCTIONED DATA###############
    # open file in read mode
    with open(DATA_DIR + '/unsanctioned.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        num_years = len(header[4:])

        X_u = [ [] for _ in range(num_years) ] # Vector to hold data for each time step

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
                        X_u[i].append(vec[i])

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

        X_s = [ [] for _ in range(num_years) ] # Vector to hold data for each time step

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
                        X_s[i].append(vec[i])

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
    n = len(X_u[0])
    X_u_shuff = []
    shuffler = np.random.permutation(n)

    for year in X_u:
        arr = np.array(year)
        arr_shuffled = arr[shuffler]
        X_u_shuff.append(arr_shuffled.tolist())

    # Shuffle rows of sanctioned country matrices with same order
    n = len(X_s[0])
    X_s_shuff = []
    shuffler = np.random.permutation(n)

    for year in X_s:
        arr = np.array(year)
        arr_shuffled = arr[shuffler]
        X_s_shuff.append(arr_shuffled.tolist())

    #Split unsanctioned countries by 80% train, 10% dev, 10% test
    n = len(X_u[0])
    num_years = len(X_u)

    num_train = math.floor(0.8* n)
    num_dev = math.floor(0.5 * (n-num_train))
    num_test = n - num_dev - num_train

    train_data = []
    dev_data = []
    test_data = []

    for i in range(num_years):
        year = X_u_shuff[i]
        train_data.append(year[0:num_train])
        dev_data.append(year[num_train:num_train+num_dev])
        test_data.append(year[num_train+num_dev:] + X_s_shuff[i]) #Append sanctioned data only to test

    ##\-----------------PROCESS DATA---------------------/##

    #train_data is NUM_YEARS x NUM_TRAIN_COUNTRIES x NUM_FEATURES
    #dev_data is NUM_YEARS x NUM_DEV_COUNTRIES x NUM_FEATURES
    #test_data is NUM_YEARS x NUM_TEST_COUNTRIES x NUM_FEATURES
    #Separate data into x and y pairs where y is the output
    x_train = []
    y_train = []

    for year in train_data:
        tempy = []
        tempx = []
        for data in year:
            tempy.append(data[0])
            tempx.append(data[1:])
        x_train.append(tempx)
        y_train.append(tempy)
    x_train =  np.array(x_train)
    y_train = np.array(y_train)

    x_dev = []
    y_dev = []
    for year in dev_data:
        tempy = []
        tempx = []
        for data in year:
            tempy.append(data[0])
            tempx.append(data[1:])
        x_dev.append(tempx)
        y_dev.append(tempy)
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    
    x_test = []
    y_test = []
    for year in test_data:
        tempy = []
        tempx = []
        for data in year:
            tempy.append(data[0])
            tempx.append(data[1:])
        x_test.append(tempx)
        y_test.append(tempy)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    ### Normalize each of the datasets in training, validation, and test datasets to a mean of 0; variance of 1.
    for i in range(num_years):
        # For x
        meanT = np.mean(x_train[i], axis = 0)
        meanD = np.mean(x_dev[i], axis = 0)
        meanP = np.mean(x_test[i], axis = 0)

        stdT = np.std(x_train[i], axis = 0)
        stdD = np.std(x_dev[i], axis = 0)
        stdP = np.std(x_test[i], axis = 0)
        
        x_train[i] =  x_train[i].astype(np.float32) - meanT
        x_dev[i] = x_dev[i].astype(np.float32) - meanD
        x_test[i] = x_test[i].astype(np.float32) - meanP
        

        x_train[i] = x_train[i].astype(np.float32) / stdT
        x_dev[i] = x_dev[i].astype(np.float32) / stdD
        x_test[i] = x_test[i].astype(np.float32) / stdP

        """
        gradient clipping
        clip values during back propagation

        try different initializers for first hidden layer

        mess around with activation functions. try leaky relu,

        adding batch normalization
        """

        # For y
        meanT = np.mean(y_train[i], axis = 0)
        meanD = np.mean(y_dev[i], axis = 0)
        meanP = np.mean(y_test[i], axis = 0)

        stdT = np.std(y_train[i], axis = 0)
        stdD = np.std(y_dev[i], axis = 0)
        stdP = np.std(y_test[i], axis = 0)
        
        y_train[i] =  y_train[i].astype(np.float32) - meanT
        y_dev[i] = y_dev[i].astype(np.float32) - meanD
        y_test[i] = y_test[i].astype(np.float32) - meanP
        

        y_train[i] = y_train[i].astype(np.float32) / stdT
        y_dev[i] = y_dev[i].astype(np.float32) / stdD
        y_test[i] = y_test[i].astype(np.float32) / stdP


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
        
        input_seq = torch.from_numpy(x_train.astype(np.float32))
        target_seq = torch.Tensor(y_train.astype(np.float32))

        input_seq = input_seq[:-1]
        target_seq = target_seq[1:]

        dev_in_seq = torch.from_numpy(x_dev.astype(np.float32))[:-1]
        dev_targ_seq = torch.from_numpy(y_dev.astype(np.float32))[1:]

        SHAPE = len(x_train[0][0])

        # write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.
        LOGFILE = open(os.path.join(LOG_DIR, f"MLP.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_r^2', 'dev_loss', 'dev_r^2']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        # Instantiate the model with hyperparameters
        model = RNN(input_size=SHAPE, output_size=1, hidden_dim=20, n_layers=1)
        
        # Define Loss, Optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        
        for step in range(EPOCHS):
            """i = np.random.choice(x_train.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(x_train[i].astype(np.float32))
            y = torch.from_numpy(y_train[i].astype(np.float32))"""
            optimizer.zero_grad()
            output, hidden = model(input_seq)

            loss = criterion(output, target_seq.view(-1).float())
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            #Calculate dev values
            dev_out, _ = model(dev_in_seq) # Dev set
            dev_loss  = criterion(dev_out, dev_targ_seq.view(-1).float())
            dev_r2 = r2_score(dev_out.detach().numpy(), dev_targ_seq.view(-1).float().detach().numpy())
            
            if step%10 == 0:
                print('Epoch: {}/{}.....'.format(step, EPOCHS), end=' ')
                print("Train Loss: {:.4f}.....".format(loss.item()), end=' ')
                print("Train R^2: {:.4f}.....".format(r2_score(output.detach().numpy(), target_seq.view(-1).float().detach().numpy())), end=' ')
                print("Dev Loss: {:.4f}.....".format(dev_loss), end=' ')
                print("Dev R^2: {:.4f}".format(dev_r2))


            step_metrics = {
                'step': step, 
                'train_loss': loss.item(), 
                'train_r^2': r2_score(output.detach().numpy(), target_seq.view(-1).float().detach().numpy()),
                'dev_loss': dev_loss,
                'dev_r^2': dev_r2
            }
            logger.writerow(step_metrics)
        LOGFILE.close()

        ### (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_densenet.pt" > "densenet.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"RNN.pt")
        
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
        
        input_seq = torch.from_numpy(x_test.astype(np.float32))
        target_seq = torch.Tensor(y_test.astype(np.float32))

        input_seq = input_seq[:-1]
        target_seq = target_seq[1:]
        
        for x in input_seq:
            xd = torch.unsqueeze(x,0)
            out, h = model(xd)

        print(f"Storing predictions in {PREDICTIONS_FILE}")
        
        out = out.detach().numpy()
        actual = y_test[-1]
        
        predictions = []
        for i in range(len(actual)):
            loss = (out[i] - actual[i])**2
            reliable = 0
            if (loss > THRESHOLD):
                reliable = 1

            predictions.append(reliable)
        
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%f")
        np.savetxt("RNN_actual.csv", y_test[-1], fmt="%f")
        np.savetxt("RNN_testvals.csv", x_test[-1], fmt="%f")
        
    else: raise Exception("Mode not recognized")