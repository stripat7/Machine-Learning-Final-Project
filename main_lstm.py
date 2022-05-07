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

class LSTM(torch.nn.Module):
    ##TODO
    def __init__():
        todo = True
        

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')

    if MODE == "train":
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
                            X_u[i].append([country] + vec[i])

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
                            X_s[i].append([country] + vec[i])

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
        
        
    elif MODE == "predict":
        ##TODO
        todo = True
