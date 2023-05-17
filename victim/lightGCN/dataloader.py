
"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import sys
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import pdb


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path=None):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        try:
            print('&&&&&&&&&&&&&&&&&&&&&')
            attack = sys.argv[sys.argv.index('--attack') + 1]
            print("load the data with fake users")
            path = sys.argv[sys.argv.index('--train_path') + 1]
            train_temp_data = pd.read_csv(path)
            # predResults.to_csv('/output/after_pred.csv')
            train_temp_data.to_csv('/output/train_attack.csv')
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
                    interaction_dict[int(user_id)].append(
                        int(item_id))  # assign items the timestap at first appearing time
            # print(interaction_dict)
            train_path = path + 'training_attack_dict.npy'
            np.save(train_path, interaction_dict)
        except Exception as e:
            path = sys.argv[sys.argv.index('--main_path') + 1]
            print("load the data without fake users")
            train_path = path + r'training_dict.npy'

        # ./LightGCN/data/training_dict.npy
        train_file = train_path
        path = sys.argv[sys.argv.index('--main_path') + 1]
        valid_file = path + '/validation_dict.npy'
        test_file = path + '/testing_dict.npy'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.validDataSize = 0
        self.testDataSize = 0

        self.train_dict = np.load(train_file, allow_pickle=True).item()
        self.valid_dict = np.load(valid_file, allow_pickle=True).item()
        self.test_dict = np.load(test_file, allow_pickle=True).item()


        for uid in self.train_dict.keys():
            trainUniqueUsers.append(uid)
            trainUser.extend([uid] * len(self.train_dict[uid]))
            trainItem.extend(self.train_dict[uid])
            self.m_item = max(self.m_item, max(self.train_dict[uid]))
            self.n_user = max(self.n_user, uid)
            self.traindataSize += len(self.train_dict[uid])
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        for uid in self.valid_dict.keys():
            if len(self.valid_dict[uid]) != 0:
                validUniqueUsers.append(uid)
                validUser.extend([uid] * len(self.valid_dict[uid]))
                validItem.extend(self.valid_dict[uid])
                self.m_item = max(self.m_item, max(self.valid_dict[uid]))
                self.n_user = max(self.n_user, uid)
                self.validDataSize += len(self.valid_dict[uid])
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        for uid in self.test_dict.keys():
            if len(self.test_dict[uid]) != 0:
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(self.test_dict[uid]))
                testItem.extend(self.test_dict[uid])
                self.m_item = max(self.m_item, max(self.test_dict[uid]))
                self.n_user = max(self.n_user, uid)
                self.testDataSize += len(self.test_dict[uid])
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def validDict(self):
        return self.valid_dict

    def num_items(self):
        return self.m_item

    def trainDict(self):
        return self.train_dict

    @property
    def testDict(self):
        return self.test_dict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserValidItems(self, users):
        validItems = []
        for user in users:
            if user in self.valid_dict:
                validItems.append(self.valid_dict[user])
        return validItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
