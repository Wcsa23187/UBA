'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import math
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import pdb

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        # print(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
        # if world.tensorboard:
        #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

    # the target users
    precision_target = []
    recall_target = []
    NDCG_target = []
    MRR_target = []
    target_user =  [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]

    for index in range(len(topN)):
        '''
        print("***********************")
        print(index)
        '''
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in target_user:  # for a user,
            '''
            print("#############")
            print('the target user is %s' % i)
            print(GroundTruth[i])
            '''
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)
                # print("the recall %s" % (userHit / len(GroundTruth[i])))
                # print('the ndcg %s' % ndcg)
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision_target.append(round(sumForPrecision / len(target_user), 4))
        recall_target.append(round(sumForRecall / len(target_user), 4))
        NDCG_target.append(round(sumForNdcg / len(target_user), 4))
        MRR_target.append(round(sumForMRR / len(target_user), 4))
        
    print(precision_target)
    print(recall_target)
    print(NDCG_target)
    print(MRR_target)

    return precision, recall, NDCG, MRR


def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))


def Test(dataset, Recmodel, epoch, w=None, multicore=0, flag=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    if flag == 0:
        testDict = dataset.valid_dict
    else:
        testDict = dataset.test_dict
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    # results = {'precision': np.zeros(len(world.topks)),
    #            'recall': np.zeros(len(world.topks)),
    #            'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            # print(batch_users_gpu)
            # print(batch_users_gpu.shape)
            rating = Recmodel.getUsersRating(batch_users_gpu)
            # print(rating)
            # print(rating.shape)
            # ipdb.set_trace()
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users)  # exclude validation items
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if flag:
                for range_i, items in enumerate(valid_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            # rating_list.append(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            # groundTrue_list.append(groundTrue)
            rating_list.extend(rating_K.cpu())  # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(groundTrue)
        # ipdb.set_trace()
        assert total_batch == len(users_list)
        precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list, rating_list, [10, 20, 50, 100])
        # print_results(None,None,results)

        if multicore == 1:
            pool.close()
        return precision, recall, NDCG, MRR


def print_epoch_result(results):
    print("Precision: {} Recall: {} NDCG: {} MRR: {}".format(
        '-'.join([str(x) for x in results['precision']]),
        '-'.join([str(x) for x in results['recall']]),
        '-'.join([str(x) for x in results['ndcg']])))


def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))
