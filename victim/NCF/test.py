import os
from random import paretovariate
import time
import argparse
import numpy as np

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


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=2048, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=200,  
	help="training epoches")
parser.add_argument("--top_k", 
    default='[10, 20, 50, 100]',  
    help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--data_path",
	type=str,
	default="/storage/shjing/recommendation/causal_discovery/data/ml-1m/random_split",
	help="main path for dataset")
parser.add_argument("--dataset",
	type=str,
	default='ml-1m',
	help="dataset")
parser.add_argument("--data_type",
	type=str,
	default="time",
	help="time_split or random_split")
parser.add_argument("--log_name",
	type=str,
	default='log',
	help="log_name")
parser.add_argument("--model_path",
	type=str,
	default="./models/",
	help="main path for model")
parser.add_argument("--model",
	type=str,
	default="NeuMF-end",
	help="model name")
parser.add_argument('--GMF_model_path',
	type=str,
	default=None,
	help='path to GMF model')
parser.add_argument('--MLP_model_path',
	type=str,
	default=None,
	help='path to MLP model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_path = args.data_path + r'\training_dict.npy'
valid_path = args.data_path + r'\validation_dict.npy'
test_path = args.data_path + r'\testing_dict.npy'
user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path, valid_path, test_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_dict, args.num_ng, True)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)

########################### CREATE MODEL #################################
model_path = '/storage/shjing/recommendation/causal_discovery/code/NCF_torch/models/MF_yelp_0.001lr_0.0dropout_64factornum_5numlayers.pth'
model = torch.load(model_path)

# model.cuda()
########################### EVALUATING #####################################

model.eval()
start_time = time.time()
valid_result = evaluate.metrics(model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, args.batch_size, 0)
print("Validation: The time elapse" + " is: " + 
        time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
evaluate.print_results(None,valid_result,None)

# test
start_time = time.time()
test_result = evaluate.metrics(model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, args.batch_size, 1)
print('-------------------------------')
print("Test: The time elapse" + " is: " + 
        time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
evaluate.print_results(None,None,test_result)
print('-------------------------------')
print('\n')


print('==='*18)
print("Performance")
evaluate.print_results(None, valid_result, test_result)

