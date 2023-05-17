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
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

############################## PREPARE DATASET ##########################
train_path = args.data_path + '/training_dict.npy'
valid_path = args.data_path + '/validation_dict.npy'
test_path = args.data_path + '/testing_dict.npy'
user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path,
                                                                                                           valid_path,
                                                                                                           test_path)


# construct the train and test datasets
train_dataset = data_utils.NCFData(
    train_data, item_num, train_dict, args.num_ng, True)

train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=0)

########################### CREATE MODEL #################################
GMF_model_path = args.GMF_model_path
MLP_model_path = args.MLP_model_path
if args.model == 'NeuMF-pre':
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

if args.model == 'NeuMF-end':
    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                      args.dropout, args.model, GMF_model, MLP_model)
elif args.model == 'MF':
    # def __init__(self, num_users, num_items, mean, embedding_size, dropout):
    model = model.MF(user_num, item_num, args.factor_num, args.num_layers, args.dropout)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer = SummaryWriter()  # for visualization

########################### TRAINING #####################################
count, best_recall = 0, 0

for epoch in range(args.epochs):
    # train
    model.train()  # Enable dropout (if have).
    start_time = time.time()
    print("开始加载 negtive数据")
    train_loader.dataset.ng_sample()

    idx = 0
    loss_per_epoch = 0.

    for user, item, label in train_loader:
        print("开始进入循环")
        print(user)
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        model.zero_grad()

        prediction = model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/loss', loss.item(), count)
        idx += 1
        loss_per_epoch += loss

        count += 1

    print("Epoch [{:03d}]  loss is {:.3f}".format(epoch, loss_per_epoch / idx))

    if (epoch + 1) % 10 == 0:
        # evaluation
        model.eval()
        valid_result = evaluate.metrics(model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, 40000, 0)
        test_result = evaluate.metrics(model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 40000, 1)
        elapsed_time = time.time() - start_time
        print("Validation: The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        evaluate.print_results(None, valid_result, test_result)

        if valid_result[1][0] > best_recall:  # recall@10
            best_epoch = epoch
            best_recall = valid_result[1][0]
            best_results = valid_result
            best_test_results = test_result
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model,
                           '{}{}_{}_{}lr_{}dropout_{}factornum_{}numlayers.pth'.format(args.model_path, args.model,
                                                                                       args.dataset, args.lr,
                                                                                       args.dropout, args.factor_num,
                                                                                       args.num_layers))

print("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
evaluate.print_results(None, best_results, best_test_results)

