import os
import sys
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
import model
import evaluate
import data_utils
import pdb


import random
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

############  load the args ###############
lr = 0.001
dropout = 0
batch_size = 128
epochs = 100
# top_k = str(sys.argv[sys.argv.index('--top_k') + 1])
top_k = '[10, 20, 50, 100]'
factor_num = int(sys.argv[sys.argv.index('--factor_num') + 1])
num_layers = 128
num_ng = 4

if sys.argv[sys.argv.index('--out') + 1] == "True":
    out = True
else:
    out = False

gpu = str(sys.argv[sys.argv.index('--gpu') + 1])
model_path = sys.argv[sys.argv.index('--model_path_NCF') + 1]
GMF_model_path = sys.argv[sys.argv.index('--GMF_model_path') + 1]
MLP_model_path = sys.argv[sys.argv.index('--MLP_model_path') + 1]
dataset = sys.argv[sys.argv.index('--dataset') + 1]
target_ids = int(sys.argv[sys.argv.index('--target_ids') + 1])
target_prediction_path_prefix = sys.argv[sys.argv.index('--target_prediction_path_prefix') + 1]
print("the args : %s" % (sys.argv))
print('\n')
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = True


def generate_target_result(model, train_dict, item_num):
    print("FINAL RESULT")
    target_id_list = [1689, 2716, 1260, 2559, 2427, 3015, 967, 2158, 3475, 3498]
    # delete all the users who have rated target item
    temp = train_dict.copy()
    for key in temp:
        for target_id in target_id_list:
            if key in train_dict.keys():
                if target_id in train_dict[key]:
                    del train_dict[key]
    # get all the items that have not been rated
    for key in train_dict:
        item = [i for i in range(item_num)]
        train_dict[key] = list(set(item).difference(set(train_dict[key])))
    # attack & predict
    data = train_dict.copy()
    user_final = []
    item_final = []
    prediction_final = []
    for i in data.keys():  # for each user
        if len(data[i]) != 0:  # if
            for start_idx in range(0, len(data[i]), 400000):  # batch size is batch item num
                end_idx = min(start_idx + 400000, len(data[i]))
                user = torch.full((end_idx - start_idx,), i, dtype=torch.int64).cuda()  # batch_size, same user
                item = torch.tensor(data[i][start_idx: end_idx], dtype=torch.int64).cuda()
                prediction = model(user, item)
                # print(user.detach().cpu().numpy().tolist())
                user_final.extend(user.detach().cpu().numpy().tolist())
                item_final.extend(item.detach().cpu().numpy().tolist())
                prediction_tmp = prediction.detach().cpu().numpy().tolist()
                prediction_final.extend(prediction_tmp)
    print('####################')
    predResults = pd.DataFrame({'user_id': user_final,
                                'item_id': item_final,
                                'rating': prediction_final
                                })

    print(predResults)
    predResults.to_csv(whole_pred_data_path)
    print(predResults.shape)
    topk_list = [10, 20, 50, 100]

    predResults_target = np.zeros([len(predResults.user_id.unique()), len(topk_list) + 2])
    for idx, (user_id, pred_result) in enumerate(predResults.groupby('user_id')):
        pred_value = 0
        # pred_value = pred_result[pred_result.item_id == target_id].rating.values[0]
        sorted_recommend_list = pred_result.sort_values('rating', ascending=False).item_id.values
        ############
        mid_list = []
        for k in topk_list:
            sign = 0
            for target_id in target_id_list:
                if target_id in sorted_recommend_list[:k]:
                    sign = 1
            mid_list.append(sign)

        new_line = [user_id, pred_value] + mid_list
        # [1 if target_id in sorted_recommend_list[:k] else 0 for k in topk_list]
        ##########
        predResults_target[idx] = new_line
    print(predResults_target)
    print("the data store in ")
    # change the target id name
    target_id = 0
    path = '%s_%d' % (target_prediction_path_prefix, target_id) + '.npy'
    print(path)
    f = open(path, 'w')
    f.close()
    np.save('%s_%d' % (target_prediction_path_prefix, target_id), predResults_target)
    #  [Errno 2] No such file or directory: './results/performance/mid_results/ml1m/ml1m_MF_RandomAttacker_0.npy'


def generate_target_result_user(model, train_dict, item_num, final_path):
    print("FINAL RESULT")
    # get all the items that have not been rated
    for key in train_dict:
        item = [i for i in range(item_num)]
        train_dict[key] = list(set(item).difference(set(train_dict[key])))
    # attack & predict
    data = train_dict.copy()
    user_final = []
    item_final = []
    prediction_final = []
    user_list =  [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
    
    for i in user_list:  # for each user
        if len(data[i]) != 0:
            for start_idx in range(0, len(data[i]), 400000):  # batch size is batch item num
                end_idx = min(start_idx + 400000, len(data[i]))
                user = torch.full((end_idx - start_idx,), i, dtype=torch.int64).cuda()  # batch_size, same user
                item = torch.tensor(data[i][start_idx: end_idx], dtype=torch.int64).cuda()
                prediction = model(user, item)
                # print(user.detach().cpu().numpy().tolist())
                user_final.extend(user.detach().cpu().numpy().tolist())
                item_final.extend(item.detach().cpu().numpy().tolist())
                prediction_tmp = prediction.detach().cpu().numpy().tolist()
                prediction_final.extend(prediction_tmp)
    print('####################')
    predResults = pd.DataFrame({'user_id': user_final,
                                'item_id': item_final,
                                'rating': prediction_final
                                })

    print(predResults)
    predResults.to_csv(final_path)
    print(final_path)
    print(predResults.shape)


############################## PREPARE DATASET ##########################

try:
    attack = sys.argv[sys.argv.index('--attack') + 1]
    print("load the data with fake users")
    path = sys.argv[sys.argv.index('--train_path') + 1]
    train_temp_data = pd.read_csv(path)
    # pd.set_option('display.max_rows', None)
    # print(train_temp_data)
    interaction_dict = {}
    temp = train_temp_data.values.tolist()
    for entry in temp:
        user_id, item_id, rate = entry
        if rate < 4:
            continue
        if user_id not in interaction_dict:
            interaction_dict[int(user_id)] = []
        if item_id not in interaction_dict[int(user_id)]:
            interaction_dict[int(user_id)].append(int(item_id))  # assign items the timestap at first appearing time
    # print(interaction_dict)
    train_path = path + 'training_attack_dict.npy'
    print('#########')
    print(train_path)
    np.save(train_path, interaction_dict)
    # /Attack/data/ml1m/pred_after.csv
    final_path = './data/ml1m/pred_after.csv'
    whole_pred_data_path = './data/ml1m/whole_pred_after.csv'
    # final_path = r'F:\RSlib\latest\debug\Attack\data\ml1m\pred_after.csv'

except Exception as e:
    print(e)
    path = sys.argv[sys.argv.index('--main_path') + 1]
    print("load the data without fake users")
    train_path = path + r'training_dict.npy'
    final_path = './data/ml1m/pred.csv'
    whole_pred_data_path = './data/ml1m/whole_pred_before.csv'
    # final_path = r'F:\RSlib\latest\debug\Attack\data\ml1m\pred.csv'

    # load model

    model = torch.load('./models/mf.pth')
    model.eval()
    valid_path = path + r'validation_dict.npy'
    test_path = path + r'testing_dict.npy'
    print("#####################")
    print(train_path)
    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(
        train_path,
        valid_path,
        test_path)
    print("-----------Validtion-----------")
    valid_result = evaluate.metrics(model, eval(top_k), train_dict, valid_dict, valid_dict, item_num, 40000, 0)
    print("-----------Test-----------")
    test_result = evaluate.metrics(model, eval(top_k), train_dict, test_dict, valid_dict, item_num, 40000, 1)
    evaluate.print_results(None, valid_result, test_result)
    train_dict_1 = train_dict.copy()
    train_dict_2 = train_dict.copy()
    generate_target_result(model, train_dict_1, item_num)
    # generate_target_result_user(model, train_dict_1, item_num,final_path)
    generate_target_result_user(model, train_dict_2, item_num, final_path)
    sys.exit()


path = sys.argv[sys.argv.index('--main_path') + 1]
############################## PREPARE DATASET ##########################

valid_path = path + r'validation_dict.npy'
test_path = path + r'testing_dict.npy'
print("#####################")
print(train_path)
user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path,
                                                                                                           valid_path,
                                                                                                           test_path)

# np.save(r'F:\RSlib\latest_2\Attack\try\my_file.npy', train_dict)
# construct the train and test datasets
print(user_num)
print(item_num)
user_num = int(user_num)
item_num = int(item_num)
train_dataset = data_utils.NCFData(
    train_data, item_num, train_dict, num_ng, True)
print('###################################')
print(train_dataset)
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size, shuffle=True, num_workers=0)

########################### CREATE MODEL #################################
GMF_model_path = GMF_model_path
MLP_model_path = MLP_model_path
if sys.argv[sys.argv.index('--model') + 1] == 'NeuMF-pre':
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
else:
    GMF_model = None
    MLP_model = None
if sys.argv[sys.argv.index('--model') + 1] == 'NeuMF-end':
    model = model.NCF(user_num, item_num, factor_num, num_layers,
                      dropout, model, GMF_model, MLP_model)
elif sys.argv[sys.argv.index('--model') + 1] == 'MF':
    model = model.MF(user_num, item_num, factor_num, num_layers, dropout)
model.cuda()

loss_function = nn.BCEWithLogitsLoss()
if sys.argv[sys.argv.index('--model') + 1] == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)

########################### TRAINING #####################################
count, best_recall = 0, 0
decrease_recall = []
for epoch in range(epochs):
    # train
    model.train()  # Enable dropout (if have).
    start_time = time.time()
    train_loader.dataset.ng_sample()
    idx = 0
    loss_per_epoch = 0.
    for user, item, label in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        model.zero_grad()
        prediction = model(user, item)
        # print(prediction)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()

        idx += 1
        loss_per_epoch += loss
        count += 1

    print("Epoch [{:03d}]  loss is {:.3f}".format(epoch, loss_per_epoch / idx))

    if (epoch + 1) % 10 == 0:
        # evaluation
        model.eval()
        print("-----------Validtion-----------")
        valid_result = evaluate.metrics(model, eval(top_k), train_dict, valid_dict, valid_dict, item_num, 40000, 0)
        print('-----------Test----------------')
        test_result = evaluate.metrics(model, eval(top_k), train_dict, test_dict, valid_dict, item_num, 40000, 1)
        elapsed_time = time.time() - start_time
        print("Validation: The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        evaluate.print_results(None, valid_result, test_result)

        if valid_result[1][0] > best_recall:  # recall@10
            best_epoch = epoch
            best_recall = valid_result[1][0]
            best_results = valid_result
            best_test_results = test_result
            train_dict_1 = train_dict.copy()
            train_dict_2 = train_dict.copy()
            generate_target_result(model, train_dict_1, item_num)
            # generate_target_result_user(model, train_dict_1, item_num,final_path)
            generate_target_result_user(model, train_dict_2, item_num, final_path)
            decrease_recall = []
            torch.save(model, 'F:\RSlib\latest_2\Attack\models\mf.pth')
        else:
            decrease_recall.append(valid_result[1][0])
            if len(decrease_recall) == 2:
                print('STOP')
                break

print("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
evaluate.print_results(None, best_results, best_test_results)
