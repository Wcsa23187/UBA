# -*- coding: utf-8 -*-
# @Time       : 2020/11/29 18:41
# @Author     : chensi
# @File       : evaluator.py
# @Software   : PyCharm
# @Desciption : None
import random
import numpy as np
import torch
import pandas as pd
import sys
seed = 1234
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import argparse
from data_loader import DataLoader
import numpy as np, pandas as pd
import scipy.stats
import time, os


class Evaluator(object):
    def __init__(self):
        self.args = self.parse_args()
        self.data_path_clean = self.args.data_path_clean
        self.data_path_attacked = self.args.data_path_attacked
        pass

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Evaluator.")
        parser.add_argument('--data_path_clean', type=str,
                            default='./data/filmTrust/filmTrust_train.dat')
        # default='./results/performance/mid_results/ml100k/ml100k_SVD_62.npy')  # , required=True)
        parser.add_argument('--data_path_attacked', type=str,
                            default='./results/data_attacked/filmTrust/filmTrust_AUSH_5.data')
        # default='./results/performance/mid_results/ml100k/ml100k_SVD_aushplus_62.npy')  # , required=True)
        return parser

    def execute(self):
        raise NotImplementedError


class Attack_Effect_Evaluator(Evaluator):
    def __init__(self):
        super(Attack_Effect_Evaluator, self).__init__()
        self.topk_list = list(map(int, self.args.topk.split(',')))
        print("Attack_Effect_Evaluator.")

    @staticmethod
    def parse_args():
        parser = Evaluator.parse_args()
        parser.add_argument('--topk', type=str, default='10,20,50,100')
        parser.add_argument('--target_users', type=str, default='all')
        args, _ = parser.parse_known_args()
        return args

    def execute(self):
        # towards all the users
        predResults_target = np.load(self.data_path_clean)
        print(self.data_path_clean)
        print(self.data_path_attacked)
        predResults_target = pd.DataFrame(predResults_target)
        print(predResults_target.shape)
        predResults_target.columns = ['user_id', 'rating'] + ['hr_%d' % i for i in self.topk_list]
        predResults_target.user_id = predResults_target.user_id.astype(int)
        #
        predResults_target_attacked = np.load(self.data_path_attacked)
        print(predResults_target_attacked)
        predResults_target_attacked = pd.DataFrame(predResults_target_attacked)
        print(predResults_target_attacked.shape)
        predResults_target_attacked.columns = ['user_id', 'rating_attacked'] + ['hr_%d_attacked' % i for i in
                                                                                self.topk_list]
        predResults_target_attacked.user_id = predResults_target_attacked.user_id.astype(int)
        
        '''if self.args.target_users != 'all':
            target_users = list(map(int, self.args.target_users.split(',')))
            predResults_target = predResults_target[predResults_target.user_id.apply(lambda x: x in target_users)]
            predResults_target_attacked = predResults_target_attacked[
                predResults_target_attacked.user_id.apply(lambda x: x in target_users)]'''

        #
        result = pd.merge(predResults_target, predResults_target_attacked, on=['user_id'])
        print(result)
        result['pred_shift'] = result['rating_attacked'] - result['rating']
        #
        keys = ['pred_shift'] + ['hr_%d_attacked' % i for i in self.topk_list] + ['hr_%d' % i for i in self.topk_list]
        result = result.mean()[keys]
        # res_str = '%.4f\t' * 5 % tuple(result.values)
        res_str = '\t'.join(["%s:%.4f" % (k.replace('_attacked', ''), result[k]) for k in keys])
        print('result begin', res_str, 'result end')
        # towards all the target users  group
        # you should change the target users
        target_users = [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
        hr_10=0
        hr_20=0
        hr_50=0
        hr_100=0
        for user in target_users:
            hr_10 += predResults_target[predResults_target['user_id'] == user]['hr_10'].values[0]
            hr_20 += predResults_target[predResults_target['user_id'] == user]['hr_20'].values[0]
            hr_50 += predResults_target[predResults_target['user_id'] == user]['hr_50'].values[0]
            hr_100 += predResults_target[predResults_target['user_id'] == user]['hr_100'].values[0]
        usergroup_hit = [hr_10/len(target_users),hr_20/len(target_users),hr_50/len(target_users),hr_100/len(target_users)]
        print('before attack')
        print(usergroup_hit)

        hr_10 = 0
        hr_20 = 0
        hr_50 = 0
        hr_100 = 0
        for user in target_users:
            hr_10 += predResults_target_attacked[predResults_target_attacked['user_id'] == user]['hr_10_attacked'].values[0]
            hr_20 += predResults_target_attacked[predResults_target_attacked['user_id'] == user]['hr_20_attacked'].values[0]
            hr_50 += predResults_target_attacked[predResults_target_attacked['user_id'] == user]['hr_50_attacked'].values[0]
            hr_100 += predResults_target_attacked[predResults_target_attacked['user_id'] == user]['hr_100_attacked'].values[0]
        print(hr_10)
        print(hr_20)
        print(hr_50)
        print(hr_100)
        usergroup_hit = [round(hr_10 / len(target_users),6),round(hr_20 / len(target_users),6),round(hr_50 / len(target_users),6),round(hr_100 / len(target_users),6)]
        print('after attack')
        print(usergroup_hit)

        return res_str


# rank 10 20 50 100

def Over_lap():
    rank = [10, 20, 50, 100]
    # ./data/ml1m/pred_after.csv
   
    after = pd.read_csv(r'./data/ml1m/pred_after.csv')
    befor = pd.read_csv(r'./data/ml1m/pred.csv')
    # F:\RSlib\latest\debug\Attack\data\ml1m\pred.csv
    # after = pd.read_csv(r'F:\RSlib\latest\debug\Attack\data\ml1m\pred_after.csv')
    # befor = pd.read_csv(r'F:\RSlib\latest\debug\Attack\data\ml1m\pred.csv')
    
    # item = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
    # item = [1551]
    # item = [1606]
    item = int(sys.argv[sys.argv.index('--oneitem') + 1])

    item = [item]
    #  2577
    # item = [2577]
    #item = [3530]
    user = [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
    user_num = len(user)
    after_rank = {}
    before_rank = {}
    for user_id in user:
        before_rank[user_id] = {}
        after_rank[user_id] = {}
        temp_before = befor[befor['user_id'] == user_id].sort_values(by='rating', ascending=False)
        temp_after = after[after['user_id'] == user_id].sort_values(by='rating', ascending=False)
        for i in rank:
            temp = temp_before.iloc[0:i - 1, :]
            before_rank[user_id][i] = list(temp['item_id'])
            temp = temp_after.iloc[0:i - 1, :]
            after_rank[user_id][i] = list(temp['item_id'])

    overlap = [0 for i in range(len(rank))]
    for user_id in user:
        for k in range(len(rank)):
            count = 0
            i = rank[k]
            for j in before_rank[user_id][i]:
                if j in after_rank[user_id][i]:
                    count += 1
            overlap[k] += count / int(i)

    over_lap = [round(i / user_num, 4) for i in overlap]
    print('over_lap')
    print(over_lap)
    before_recall = [0 for i in range(len(rank))]
    after_recall = [0 for i in range(len(rank))]
    cnt_before = [0 for i in range(len(rank))]
    cnt_after = [0 for i in range(len(rank))]
    for user_id in user:
        for k in range(len(rank)):
            i = rank[k]
            count_before = 0
            count_after = 0
            for item_id in item:
                if item_id in before_rank[user_id][i]:
                    count_before += 1
                if item_id in after_rank[user_id][i]:
                    count_after += 1
            before_recall[k] += count_before / len(item)
            after_recall[k] += count_after / len(item)
            cnt_before[k] += count_before
            cnt_after[k] += count_after
    before_recall = [i / user_num for i in before_recall]
    after_recall = [i / user_num for i in after_recall]
    print('recall_target')
    print(before_recall)
    print(after_recall)
    print("cnt num")
    print(cnt_before)
    print(cnt_after)

    before_ndcg = []
    for i in rank:
        sumForNDCG = 0
        for user_id in user:
            dcg = 0
            idcg = 0
            idcgCount = len(item)
            ndcg = 0
            for index in range(len(before_rank[user_id][i])):
                if before_rank[user_id][i][index] in item:
                    dcg += 1.0 / (np.log2(index + 2))
                if idcgCount > 0:
                    idcg += 1.0 / (np.log2(index + 2))
                    idcgCount -= 1
            if (idcg != 0):
                ndcg += (dcg / idcg)
            sumForNDCG += ndcg
        before_ndcg.append(sumForNDCG)
    after_ndcg = []
    for i in rank:
        sumForNDCG = 0
        for user_id in user:
            dcg = 0
            idcg = 0
            idcgCount = len(item)
            ndcg = 0
            for index in range(len(after_rank[user_id][i])):
                if after_rank[user_id][i][index] in item:
                    dcg += 1.0 / (np.log2(index + 2))
                if idcgCount > 0:
                    idcg += 1.0 / (np.log2(index + 2))
                    idcgCount -= 1
            if (idcg != 0):
                ndcg += (dcg / idcg)
            sumForNDCG += ndcg
        after_ndcg.append(sumForNDCG)


    print('ndcg_target')
    print(before_ndcg)
    print(after_ndcg)



def whole_over_lap():
    rank = [10, 20, 50, 100]
    # ./data/ml1m/pred_after.csv

    after = pd.read_csv(r'./data/ml1m/whole_pred_after.csv')
    befor = pd.read_csv(r'./data/ml1m/whole_pred_before.csv')
    # F:\RSlib\latest\debug\Attack\data\ml1m\pred.csv
    # after = pd.read_csv(r'F:\RSlib\latest\debug\Attack\data\ml1m\pred_after.csv')
    # befor = pd.read_csv(r'F:\RSlib\latest\debug\Attack\data\ml1m\pred.csv')

    # item = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
    # item = [1551]
    # item = [1606]
    # item = [2577]
    
    item = int(sys.argv[sys.argv.index('--oneitem') + 1])

    item = [item]
    user = befor['user_id'].values.tolist()
    user = list(set(user))
    user_num = len(user)

    after_rank = {}
    before_rank = {}
    for user_id in user:
        before_rank[user_id] = {}
        after_rank[user_id] = {}
        temp_before = befor[befor['user_id'] == user_id].sort_values(by='rating', ascending=False)
        temp_after = after[after['user_id'] == user_id].sort_values(by='rating', ascending=False)
        for i in rank:
            temp = temp_before.iloc[0:i - 1, :]
            before_rank[user_id][i] = list(temp['item_id'])
            temp = temp_after.iloc[0:i - 1, :]
            after_rank[user_id][i] = list(temp['item_id'])

    overlap = [0, 0, 0, 0]
    for user_id in user:
        for k in range(len(rank)):
            count = 0
            i = rank[k]
            for j in before_rank[user_id][i]:
                if j in after_rank[user_id][i]:
                    count += 1
            overlap[k] += count / int(i)

    over_lap = [round(i / user_num, 4) for i in overlap]
    print('over_lap')
    print(over_lap)
    before_recall = [0, 0, 0, 0]
    after_recall = [0, 0, 0, 0]
    for user_id in user:
        for k in range(len(rank)):
            i = rank[k]
            count_before = 0
            count_after = 0
            for item_id in item:
                if item_id in before_rank[user_id][i]:
                    count_before += 1
                if item_id in after_rank[user_id][i]:
                    count_after += 1
            before_recall[k] += count_before / len(item)
            after_recall[k] += count_after / len(item)
    before_recall = [i / user_num for i in before_recall]
    after_recall = [i / user_num for i in after_recall]
    print('recall_target')
    print(before_recall)
    print(after_recall)
    before_ndcg = []
    for i in rank:
        sumForNDCG = 0
        for user_id in user:
            dcg = 0
            idcg = 0
            idcgCount = len(item)
            ndcg = 0
            for index in range(len(before_rank[user_id][i])):
                if before_rank[user_id][i][index] in item:
                    dcg += 1.0 / (np.log2(index + 2))
                if idcgCount > 0:
                    idcg += 1.0 / (np.log2(index + 2))
                    idcgCount -= 1
            if (idcg != 0):
                ndcg += (dcg / idcg)
            sumForNDCG += ndcg
        before_ndcg.append(sumForNDCG)
    after_ndcg = []
    for i in rank:
        sumForNDCG = 0
        for user_id in user:
            dcg = 0
            idcg = 0
            idcgCount = len(item)
            ndcg = 0
            for index in range(len(after_rank[user_id][i])):
                if after_rank[user_id][i][index] in item:
                    dcg += 1.0 / (np.log2(index + 2))
                if idcgCount > 0:
                    idcg += 1.0 / (np.log2(index + 2))
                    idcgCount -= 1
            if (idcg != 0):
                ndcg += (dcg / idcg)
            sumForNDCG += ndcg
        after_ndcg.append(sumForNDCG)

    print('ndcg_target')
    print(before_ndcg)
    print(after_ndcg)

Over_lap()
print("#######whole users########")
whole_over_lap()