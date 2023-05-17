#  let the sample be fixed file

import os
from random import paretovariate
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils
import pdb



############################## PREPARE DATASET ##########################
train_path = 'F:\RSlib\code\Recommender_baselines-main\data\ml100k_train.dat'
test_path = 'F:\RSlib\code\Recommender_baselines-main\data\ml100k_test.dat'

def load_file_as_dataFrame(path_train, path_test):
    # load data to pandas dataframe
    train_data = pd.read_csv(path_train, sep='\t', names=['user_id', 'item_id', 'rating'], engine='python')
    test_data = pd.read_csv(path_test, sep='\t', names=['user_id', 'item_id', 'rating'], engine='python')
    n_users = max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
    n_items = max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1
    # print("Number of users : %d , Number of items : %d. " % (n_users, n_items), flush=True)
    # print("Train size : %d , Test size : %d. " % (train_data.shape[0], test_data.shape[0]), flush=True)
    return train_data, test_data, n_users, n_items


train_data_df, test_data_df, n_users, n_items = load_file_as_dataFrame(train_path, test_path)
# cat the train_data and test_data
result = pd.concat([train_data_df , test_data_df])
# cut the whole data with 8 : 1 : 1 = train : test : validation
user_name = result['user_id'].unique()
train_df = pd.DataFrame(columns = ['user_id', 'item_id', 'rating'])
vali_df = pd.DataFrame(columns = ['user_id', 'item_id', 'rating'])
test_df = pd.DataFrame(columns = ['user_id', 'item_id', 'rating'])
for user_id in user_name:
    mid = result[result['user_id'] == user_id]
    train_df_son = mid.sample(frac=0.8)
    vali_df_son = mid.sample(frac=0.1)
    test_df_son = mid.sample(frac=0.1)
    train_df = pd.concat([train_df_son, train_df])
    vali_df = pd.concat([vali_df_son, vali_df])
    test_df = pd.concat([test_df_son, test_df])
# 8 : 1 : 1 data = train_df : vali_df : test_df and i user the sample cut
# let train_df to be the array
train_data = []
for index, row in train_df.iterrows():
    mid = []
    mid.append(row['user_id'])
    mid.append(row['item_id'])
    mid.append(row['rating'])
    train_data.append(mid)

# let we have a dict with vali and test 4/5--->1  1/2/3 ---> 0
for user_id in user_name:
    mid = vali_df[vali_df['user_id'] == user_id]
    vali_dict={user_id:i for i in mid['item_id']}

# construct the train and test datasets

# n_users, n_items

train_dataset = data_utils.NCFData(
    train_data, n_items, True)

train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=0)



