import os
import sys
target_ids = sys.argv[sys.argv.index('--target_ids') + 1]
target_prediction_path_prefix = sys.argv[sys.argv.index('--target_prediction_path_prefix') + 1]

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import pandas as pd

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset



def generate_target_result(model):
    item_num = dataset.num_items()
    print(item_num)
    train_dict_before = dataset.trainDict()
    train_dict = train_dict_before.copy()
    # print(train_dict)
    print("GENERATE FAKE")
    print("max")
    print(max(train_dict.keys()))
    print(len(train_dict.keys()))
    pick = [i for i in range(len(train_dict.keys()))]
    user = torch.tensor(pick, dtype=torch.long)
    prediction = model.getUsersRating(user)
    print(prediction.shape)
    print(prediction.detach().cpu().numpy())
    prediction = prediction.detach().cpu().numpy()
    user = []
    item = []
    rating = []
    # popular
    # target_ids = [2577]
    
    target_ids = [int(sys.argv[sys.argv.index('--oneitem') + 1])]
    # target_ids = [68]
    # target_ids = [1606]
    # target_ids = [1551]
    # popular items [2649, 1104, 251, 1118, 464, 573, 1846, 2372, 1176, 577]
    # target_ids = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
    print('#################')
    for key in train_dict.keys():
        signal = 0
        for target_id in target_ids:
            if target_id in train_dict[key]:
                signal = 1
        if signal == 1:
            continue
        else:
            item_full = [i for i in range(item_num)]
            for i in train_dict[key]:
                item_full.remove(i)
            for j in item_full:
                user.append(key)
                item.append(j)
                rating.append(prediction[key][j])
    print(len(user))
    print(len(item))
    print(len(rating))
    predResults = pd.DataFrame({'user_id': user,
                                'item_id': item,
                                'rating': rating
                                })
    print(predResults)
    predResults.to_csv(whole_pred_data_path)
    print(predResults.shape)
    predResults.to_csv('/output/after_pred.csv')

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
            for target_id in target_ids:
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


def generate_target_result_user(model):
    item_num = dataset.num_items()
    print(item_num)
    train_dict_before = dataset.trainDict()
    train_dict = train_dict_before.copy()
    # print(train_dict)
    print("GENERATE FAKE")
    print("max")
    print(max(train_dict.keys()))
    print(len(train_dict.keys()))
    user_list = [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
    user = torch.tensor(user_list, dtype=torch.long)
    prediction = model.getUsersRating(user)
    print(prediction.shape)
    print(prediction.detach().cpu().numpy())
    prediction = prediction.detach().cpu().numpy()
    print(prediction)
    user = []
    item = []
    rating = []
    for index in range(len(user_list)):
        key = user_list[index]
        item_full = [i for i in range(item_num)]
        for i in train_dict[key]:
            item_full.remove(i)
        for j in item_full:
            user.append(key)
            item.append(j)
            rating.append(prediction[index][j])
    print(len(user))
    print(len(item))
    print(len(rating))
    predResults = pd.DataFrame({'user_id': user,
                                'item_id': item,
                                'rating': rating
                                })
    print(predResults)
    predResults.to_csv(final_path)
    print(final_path)
    print(predResults.shape)





# test_predRatings_torch = model.getUsersRating(test_uids_torch, test_iids_torch)
# train_df = dataset.trainDict()
try:
    attack = sys.argv[sys.argv.index('--attack') + 1]
    print("load the data with fake users")
    final_path = './data/ml1m/pred_after.csv'
    whole_pred_data_path = './data/ml1m/whole_pred_after.csv'

except Exception as e:
    print(e)
    print("load the data without fake users")
    final_path = './data/ml1m/pred.csv'
    whole_pred_data_path = './data/ml1m/whole_pred_before.csv'
    
    
    
    ########## load the remodel  #########
    Recmodel = torch.load('./models/lgn_20.pth')
    Recmodel.eval()
    bpr = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1
    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")
    epoch = 0

    print("---------------validation--------------")
    v_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 0)
    Procedure.print_results(None, v_results, None)
    # cprint("[test]")
    print("---------------Test--------------")
    t_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 1)
    Procedure.print_results(None, None, t_results)
    generate_target_result(Recmodel)
    generate_target_result_user(Recmodel)
    sys.exit()
    





Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_recall = 0
    best_epoch = 0
    decrease_recall = []
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if (epoch + 1) % 10 == 0:
            # cprint("[validation]")
            print("---------------validation--------------")
            v_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 0)
            Procedure.print_results(None, v_results, None)
            # cprint("[test]")
            print("---------------Test--------------")
            t_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 1)
            Procedure.print_results(None, None, t_results)
            if v_results[1][0] > best_recall:
                best_epoch = epoch
                best_recall = v_results[1][0]
                best_v, best_t = v_results, t_results
                '''
                print('Start to generate')
                generate_target_result(Recmodel)
                generate_target_result_user(Recmodel)
                '''
                torch.save(Recmodel, '/code/Attack/models/lgn_20_best.pth')
                # torch.save(Recmodel, './models/lgn_20.pth')
                decrease_recall = []
                
            else:
                decrease_recall.append(v_results[1][0])
                if len(decrease_recall) == 2:
                    print('STOP')
                    break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        # torch.save(Recmodel.state_dict(), weight_file)

    print("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
    print("Validation:")
    Procedure.print_results(None, best_v, None)
    print("Test:")
    Procedure.print_results(None, None, best_t)


    # load best model of attack
    Recmodel = torch.load('/code/Attack/models/lgn_20_best.pth')
    Recmodel.eval()
    bpr = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1
    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")
    epoch = 0
    '''
    print("---------------validation--------------")
    v_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 0)
    Procedure.print_results(None, v_results, None)
    # cprint("[test]")
    print("---------------Test--------------")
    t_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], 1)
    Procedure.print_results(None, None, t_results)
    '''
    generate_target_result(Recmodel)



finally:
    if world.tensorboard:
        w.close()

