# -*- coding: utf-8 -*-
# @Time       : 2020/12/3 20:03
# @Author     : chensi
# @File       : legup.py
# @Software   : PyCharm
# @Desciption : None

import scipy
import random
import numpy as np
import torch
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




'''
import sys
myseed = int(sys.argv[sys.argv.index('--allseed') + 1])
print('!@#$$%^#$^#$@^#$#!%#$^%$%$#%$@%!#$%#%@$%$#!')
print(myseed)
'''
import sys
seed = int(sys.argv[sys.argv.index('--allseed') + 1])

print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from datetime import datetime
import time
from torch import nn
import torch.nn.functional as F
import math
import torch.optim as optim

from attacker import Attacker
from aushplus_helper import *
from utils import DataLoader


class AUSH(Attacker):

    def __init__(self):
        super(AUSH, self).__init__()
        # self.selected_ids = list(map(int, self.args.selected_ids.split(',')))
        # self.selected_ids = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        self.selected_ids = [1153, 2201, 1572, 836, 523, 849, 1171, 344, 857, 1213, 1535]
        #
        self.restore_model = self.args.restore_model
        self.model_path = self.args.model_path
        #
        #
        # self.epochs = self.args.epoch
        self.epochs = 100
        self.batch_size = self.args.batch_size
        #
        self.learning_rate_G = self.args.learning_rate_G
        self.reg_rate_G = self.args.reg_rate_G
        self.ZR_ratio = self.args.ZR_ratio
        #
        self.learning_rate_D = self.args.learning_rate_D
        self.reg_rate_D = self.args.reg_rate_D
        #
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.device = torch.device("cuda:0")

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        # parser.add_argument('--selected_ids', type=str, default='1,2,3', required=True)
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--model_path', type=str, default='')
        #
        parser.add_argument('--epoch', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--learning_rate_G', type=float, default=0.01)
        parser.add_argument('--reg_rate_G', type=float, default=0.0001)
        parser.add_argument('--ZR_ratio', type=float, default=0.2)
        #
        parser.add_argument('--learning_rate_D', type=float, default=0.001)
        parser.add_argument('--reg_rate_D', type=float, default=1e-5)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        #
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(AUSH, self).prepare_data()
        train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.train_data_array = train_matrix.toarray()
        # let the rated inter be 1 and none be 0
        self.train_data_mask_array = scipy.sign(self.train_data_array)
        # true/false and to the float 1. / 0.
        mask_array = (self.train_data_array > 0).astype(np.float)
        # let selected items and target items be 0
        mask_array[:, self.selected_ids + self.target_id_list] = 0
        self.template_idxs = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]

    def build_network(self):
        self.netG = AushGenerator(input_dim=self.n_items).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.learning_rate_G)

        self.netD = AushDiscriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = optim.Adam(self.netD.parameters(), lr=self.learning_rate_D)

        pass

    def sample_fillers(self, real_profiles):
        fillers = np.zeros_like(real_profiles)
        filler_pool = set(range(self.n_items)) - set(self.selected_ids) - set(self.target_id_list)
        # filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        # sampled_cols = [filler_sampler([filler_pool, self.filler_num]) for _ in range(real_profiles.shape[0])]

        # filler_sampler = lambda x: np.random.choice(size=self.filler_num, replace=False,
        #                                             a=list(set(np.argwhere(x > 0).flatten()) & filler_pool))

        filler_sampler = lambda x: np.random.choice(size=self.filler_num, replace=True,
                                                        a=list(set(np.argwhere(x > 0).flatten()) & filler_pool))
         
        sampled_cols = [filler_sampler(x) for x in real_profiles]

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, np.array(sampled_cols).flatten()] = 1
        return fillers

    def train(self):

        # save
        total_batch = math.ceil(len(self.template_idxs) / self.batch_size)
        idxs = np.random.permutation(self.template_idxs)  # shuffled ordering
        #
        g_loss_rec_l = []
        g_loss_shilling_l = []
        g_loss_gan_l = []

        d_loss_list, g_loss_list = [], []
        for i in range(total_batch):

            # ---------------------
            #  Prepare Input
            # ---------------------
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]
            len_batch_set_idx = len(batch_set_idx)
            # print(len_batch_set_idx)
            target_user = [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
            target_user_idx = []
            for user in target_user:
                if user in idxs:
                    target_user_idx.append(user)
            target_user = target_user_idx
            user_random = random.sample(target_user, 20)
            batch_set_idx_random = random.sample(list(batch_set_idx), len_batch_set_idx-20)
            batch_set_idx = user_random + batch_set_idx_random
            # print(len(batch_set_idx))
            
            # Adversarial ground truths
            valid_labels = np.ones_like(batch_set_idx)
            fake_labels = np.zeros_like(batch_set_idx)
            valid_labels = torch.tensor(valid_labels).type(torch.float).to(self.device).reshape(len(batch_set_idx), 1)
            fake_labels = torch.tensor(fake_labels).type(torch.float).to(self.device).reshape(len(batch_set_idx), 1)
            # print(valid_labels)


            # Select a random batch of real_profiles
            real_profiles = self.train_data_array[batch_set_idx, :]
            # sample fillers
            fillers_mask = self.sample_fillers(real_profiles)
            # selected
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.
            # target
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.
            # ZR_mask
            ZR_mask = (real_profiles == 0) * selects_mask
            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[:math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0

            # ----------- torch.mul ---------
            real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
            fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
            selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
            target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
            ZR_mask = torch.tensor(ZR_mask).type(torch.float).to(self.device)
            input_template = torch.mul(real_profiles, fillers_mask)
            # ----------generate----------
            self.netG.eval()
            gen_output = self.netG(input_template)
            gen_output = gen_output.detach()
            # ---------mask--------
            selected_patch = torch.mul(gen_output, selects_mask)
            middle = torch.add(input_template, selected_patch)
            fake_profiles = torch.add(middle, target_patch)
            # --------Discriminator------
            # forward
            self.D_optimizer.zero_grad()
            self.netD.train()
            d_valid_labels = self.netD(real_profiles * (fillers_mask + selects_mask))
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            # loss
            # print(d_valid_labels.shape)
            # d_valid_labels = d_valid_labels.reshape(-1)
            # d_fake_labels = d_fake_labels.reshape(-1)
            D_real_loss = nn.BCELoss()(d_valid_labels, valid_labels)
            D_fake_loss = nn.BCELoss()(d_fake_labels, fake_labels)
            d_loss = 0.5 * (D_real_loss + D_fake_loss)
            print("d_loss")
            print(d_loss)
            d_loss.backward()
            self.D_optimizer.step()
            self.netD.eval()

            # ---------train G-------
            self.netG.train()
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            g_loss_gan = nn.BCELoss()(d_fake_labels, valid_labels)
            g_loss_shilling = nn.MSELoss()(fake_profiles * selects_mask, selects_mask * 5.)
            # g_loss_shilling = (fake_profiles * selects_mask - selects_mask * 5.) ** 2
            # * selects_mask - selects_mask * input_template) * ZR_mask
            # g_loss_rec = (fake_profiles * selects_mask - selects_mask * input_template) * ZR_mask ** 2
            g_loss_rec = nn.MSELoss()(fake_profiles * selects_mask * ZR_mask, selects_mask * input_template * ZR_mask)
            g_loss = g_loss_gan + g_loss_rec + g_loss_shilling
            # g_loss = g_loss_rec + g_loss_shilling
            # + g_loss_shilling + g_loss_gan
            g_loss_rec_l.append(g_loss_rec.item())
            g_loss_shilling_l.append(g_loss_shilling.item())
            g_loss_gan_l.append(g_loss_gan.item())
            self.G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()
            
        print(g_loss_rec_l)
        print(g_loss_shilling_l)
        print(g_loss_gan_l)
        return

    def execute(self):

        self.prepare_data()

        # Build and compile GAN Network
        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            for epoch in range(self.epochs):
                print("epoch %d"%epoch)
                self.train()

                '''if self.verbose and epoch % self.T == 0:
                    print("epoch:%d\td_loss:%.4f\tg_loss:%.4f" % (epoch,g_loss_cur))'''

            # self.save(self.model_path)
            print("training done.")

        metrics = self.test(victim='SVD', detect=True)
        # print(metrics, flush=True)
        return

    def generate_fakeMatrix(self):

        way = int(sys.argv[sys.argv.index('--way') + 1])

        if way == 1:
            print('just random attack from all 20% data')
            idx = self.template_idxs[np.random.randint(0, len(self.template_idxs), self.attack_num)]
            idx = list(idx)
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            new = []
            for i in range(len(idx)):
                if sign[i]==0:
                    new.append(idx[i])
            idx = new
            print(idx)
            print(len(idx))

        if way == 2:
            print('random attack user the target 50 users')
            user = [5520, 5678, 1771, 4738, 317, 4962, 1338, 4975, 970, 3305, 5, 646, 1802, 2191, 2704, 3987, 789, 3734,
                5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903,
                587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
            dict_user_code = {}
            j = 0
            for i in user:
                dict_user_code[j] = i
                j+=1
            sampled_idx = np.random.choice(range(50), self.attack_num)

            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sampled_idx[i])

            sampled_idx = new
            print(sampled_idx)
            print(len(sampled_idx))

            idx = []
            for i in sampled_idx:
                idx.append(dict_user_code[i])
            print(idx)
        if way == 3:
            print('user our methods')
            # idx = [3582, 3582, 2678, 2678, 2678, 2678, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 362, 362, 362, 362, 362, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 587, 587, 587, 587, 587, 3903, 3903, 3903, 3903, 3903, 4926, 4926, 4926, 4926, 2365, 2365, 2365, 2365, 2365, 1082, 1082, 1082, 1082, 1462, 1462, 1462, 1462, 4918, 4918, 4918, 4918, 2347, 2347, 2347, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 797, 797, 797, 797, 3734, 3734, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 5, 5, 3305, 3305, 3305, 970, 970, 970, 970, 970, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5678, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520, 5520]
            # idx = [5618, 5618, 5618, 623, 623, 623, 1238, 1238, 4434, 3532, 3532, 3532, 3532, 4926, 4926, 2347, 2347, 2347, 2347, 4770, 797, 797, 797, 797, 789, 789, 789, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 3305, 3305, 3305, 970, 970, 970, 970, 4962, 1771, 5520, 5520, 5520, 5520, 5520]
            # idx = [2678, 2678, 2678, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 4330, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 587, 587, 587, 587, 4926, 4926, 4926, 4926, 2365, 2365, 2365, 2365, 2347, 2347, 2347, 2347, 425, 425, 425, 425, 425, 2340, 2340, 2340, 35, 35, 35, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 797, 797, 797, 797, 797, 3734, 3734, 3734, 3734, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 646, 5, 5, 5, 5, 5, 3305, 3305, 3305, 3305, 3305, 970, 970, 970, 970, 4975, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5678, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520, 5520]
            #idx = [3582, 3582, 3582, 3582, 2678, 2678, 2678, 2678, 5618, 623, 4330, 4330, 4330, 362, 362, 362, 362, 1509, 1509, 1509, 1509, 5345, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 1238, 1238, 1238, 4434, 4434, 4434, 3532, 587, 587, 3903, 4926, 4926, 4926, 4926, 4926, 1082, 1082, 1082, 1082, 1082, 1462, 1462, 1462, 1462, 1462, 4918, 5682, 5682, 5682, 2347, 2347, 425, 425, 425, 425, 2597, 2597, 2597, 2597, 2340, 2340, 2340, 2340, 2340, 35, 35, 35, 4770, 4770, 4770, 4770, 160, 160, 160, 160, 286, 286, 286, 797, 3734, 3734, 3734, 3734, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 5, 3305, 3305, 970, 4975, 4975, 4975, 4975, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520]
            # idx = [5618, 623, 362, 362, 362, 1509, 1509, 5345, 4572, 1238, 3532, 587, 3903, 4918, 5682, 2347, 425, 2597, 2340, 2340, 35, 35, 35, 4770, 286, 797, 789, 3987, 3987, 3987, 2704, 2191, 646, 646, 5, 3305, 970, 4975, 4975, 4975, 4975, 1338, 4962, 4962, 4962, 317, 317, 317, 317, 5520]
            # idx = [3582, 3582, 3582, 3582, 3582, 2678, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 362, 1509, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 3532, 587, 587, 587, 587, 587, 3903, 4926, 4926, 4926, 4926, 4926, 2365, 1082, 3386, 1462, 4918, 5682, 5682, 5682, 5682, 5682, 2347, 425, 2597, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 797, 5017, 789, 789, 789, 789, 789, 3987, 2704, 2704, 2704, 2704, 2704, 2191, 5, 3305, 970, 4975, 1338, 4962, 4962, 4962, 4962, 4962, 317, 4738, 4738, 4738, 4738, 4738, 5678]
            idx = [3582, 3582, 3582, 3582, 3582, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 587, 587, 587, 587, 587, 4926, 4926, 4926, 4926, 4926, 5682, 5682, 5682, 5682, 5682, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 5017, 5017, 5017, 5017, 5017, 789, 789, 789, 789, 789, 2704, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 970, 970, 970, 970, 970, 4975, 4975, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 4962, 4738, 4738, 4738, 4738, 4738]
            idx = list(idx)
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            new = []
            for i in range(len(idx)):
                if sign[i]==0:
                    new.append(idx[i])
            idx = new
            print(idx)
            print(len(idx))
       
        
        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles)
        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, self.target_id_list] = 5.

        # Generate
        real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
        fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
        selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
        target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
        input_template = torch.mul(real_profiles, fillers_mask)
        self.netG.eval()
        gen_output = self.netG(input_template)
        selected_patch = torch.mul(gen_output, selects_mask)
        middle = torch.add(input_template, selected_patch)
        fake_profiles = torch.add(middle, target_patch)
        fake_profiles = fake_profiles.detach().cpu().numpy()
        # fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches

        return fake_profiles

    def generate_injectedFile(self, fake_array):
        super(AUSH, self).generate_injectedFile(fake_array)




class torchAttacker(Attacker):
    def __init__(self):
        super(torchAttacker, self).__init__()
     
        self.restore_model = self.args.restore_model
   
        self.model_path = self.args.model_path
        
        self.verbose = self.args.verbose
        self.T = self.args.T
      
        self.device = torch.device("cuda:0" )
        # self.epochs = 1
        self.epochs = self.args.epoch
      
        self.lr_G = self.args.lr_G
        self.momentum_G = self.args.momentum_G
    
        self.lr_D = self.args.lr_D
        self.momentum_D = self.args.momentum_D
        self.batch_size_D = self.args.batch_size_D
        self.surrogate = self.args.surrogate
        pass

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--model_path', type=str, default='')
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        #
        parser.add_argument('--use_cuda', type=int, default=1)
        # parser.add_argument('--cuda_id', type=int, default=2)
        parser.add_argument('--epoch', type=int, default=3)
        # Generator
        parser.add_argument("--lr_G", type=float, default=0.01)
        parser.add_argument("--momentum_G", type=float, default=0.99)
        # Discriminator
        parser.add_argument("--lr_D", type=float, default=0.01)
        parser.add_argument("--momentum_D", type=float, default=0.99)
        parser.add_argument('--batch_size_D', type=int, default=64)

        # Surrogate
        parser.add_argument("--epoch_S", type=int, default=50)
        parser.add_argument("--unroll_steps_S", type=int, default=1)
        parser.add_argument("--hidden_dim_S", type=int, default=16)
        parser.add_argument("--lr_S", type=float, default=1e-2)
 
        parser.add_argument("--weight_decay_S", type=float, default=1e-5)
        parser.add_argument('--batch_size_S', type=int, default=16)
       
        parser.add_argument('--weight_pos_S', type=float, default=1.)
        parser.add_argument('--weight_neg_S', type=float, default=0.)
        parser.add_argument("--surrogate", type=str, default="WMF")
        
        gan_args, unknown_args = parser.parse_known_args()
        return gan_args

    def prepare_data(self):

        self.path_train = './data/%s/%s_50user_train.csv' % (self.data_set, self.data_set)
        path_test = './data/%s/%s_50user_test.csv' % (self.data_set, self.data_set)


        dataset_class = DataLoader(self.path_train, path_test)
        self.train_data_df, self.test_data_df, self.n_users, self.n_items = dataset_class.load_file_as_dataFrame()
        print(self.n_users)
        print(self.n_items)
        train_matrix, _ = dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        test_matrix, _ = dataset_class.dataFrame_to_matrix(self.test_data_df, self.n_users, self.n_items)
        self.train_array, self.test_array = train_matrix.toarray(), test_matrix.toarray()
 
        self.data_loader = torch.utils.data.DataLoader(dataset=torch.from_numpy(self.train_array).type(torch.float32),
                                                       batch_size=self.batch_size_D, shuffle=True, drop_last=True)
 
        self.target_users = np.where(self.train_array[:, self.target_id] == 0)[0]
        attack_target = np.zeros((len(self.target_users), self.n_items))
        attack_target[:, self.target_id] = 1.0
        self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
        pass

    def build_network(self):
        raise NotImplemented


    def get_sur_predictions(self, fake_tensor):
        

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
        data_tensor = torch.cat(
            [torch.from_numpy(self.train_array).type(torch.float32).to(self.device),
             fake_tensor], dim=0)


        surrogate=self.surrogate

        if surrogate == 'WMF':
            sur_trainer_ = WMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=self.args.hidden_dim_S,
                # device=self.device,
                device = self.device,
                lr=self.args.lr_S,
                weight_decay=self.args.weight_decay_S,
                batch_size=self.args.batch_size_S,
                weight_pos=self.args.weight_pos_S,
                weight_neg=self.args.weight_neg_S,
                verbose=False)
            epoch_num_ = self.args.epoch_S
            unroll_steps_ = self.args.unroll_steps_S
            #print(self.args.batch_size_S)
            #print(self.device)
        elif surrogate == 'ItemAE':
            sur_trainer_ = ItemAETrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dims=self.args.hidden_dim_S,
                device=self.device,
                lr=self.args.lr_S,
                l2=self.args.weight_decay_S,
                batch_size=self.args.batch_size_S,
                weight_pos=self.args.weight_pos_S,
                weight_neg=self.args.weight_neg_S,
                verbose=False)
            epoch_num_ = self.args.epoch_S
            unroll_steps_ = self.args.unroll_steps_S
        elif surrogate == 'SVDpp':
            sur_trainer_ = SVDppTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dims=[128],
                device=self.device,
                lr=1e-3,
                l2=5e-2,
                batch_size=128,
                weight_alpha=20)
            epoch_num_ = 10
            unroll_steps_ = 1
        elif surrogate == 'NMF':
            sur_trainer_ = NMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                batch_size=128,
                device=self.device,
            )
            epoch_num_ = 50
            unroll_steps_ = 1
        elif surrogate == 'PMF':
            sur_trainer_ = PMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=128,
                device=self.device,
                lr=0.0001,
                weight_decay=0.1,
                batch_size=self.args.batch_size_S,
                momentum=0.9,
                verbose=True)
            epoch_num_ = 50
            unroll_steps_ = 1
        else:
            print('surrogate model error : ', surrogate)
            exit()

        sur_predictions = sur_trainer_.fit_adv(
            data_tensor=data_tensor,
            epoch_num=epoch_num_,
            unroll_steps=unroll_steps_
        )

        sur_test_rmse = np.mean((sur_predictions[:self.n_users][self.test_array > 0].detach().cpu().numpy()
                                 - self.test_array[self.test_array > 0]) ** 2)
  
        return sur_predictions, sur_test_rmse

    def train_G(self):
        raise NotImplemented

    def save(self, path):


        if path is None or len(path) == 0:
            path = './results/model_saved/%s/%s_%s_%d' % (
                self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        torch.save(self.netG.state_dict(), path + '_G.pkl')
   
        torch.save(self.netD.state_dict(), path + '_D.pkl')
        return

    def restore(self, path):

        if path is None or len(path) == 0:
            path = './results/model_saved/%s/%s_%s_%d' % (
                self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        self.netG.load_state_dict(torch.load(path + '_G.pkl'))
     
        self.netD.load_state_dict(torch.load(path + '_D.pkl'))
        return

    def generate_fakeMatrix(self):

        self.netG.eval()
    
        _, fake_tensor = self.netG(self.real_template)

        target_id_set = self.target_id_list
        rate = int(self.attack_num / len(target_id_set))
        for i in range(len(target_id_set)):
            fake_tensor[i * rate:(i + 1) * rate, target_id_set[i]] = 5
        #fake_tensor[:, self.target_id] = 5
       
        return fake_tensor.detach().cpu().numpy()

    @staticmethod
    def custimized_attack_loss(logits, labels):

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -log_probs * labels
        instance_data = labels.sum(1)
        instance_loss = loss.sum(1)
        # Avoid divide by zeros.
        res = instance_loss / (instance_data + 0.1)  # PSILON)
        return res

    @staticmethod
    def update_params(loss, optimizer):

        grad_groups = torch.autograd.grad(loss.cuda(), [x.cuda() for x in optimizer.param_groups[0]['params']], allow_unused=True)
  
        for para_, grad_ in zip(optimizer.param_groups[0]['params'], grad_groups):
            if para_.grad is None:
                para_.grad = grad_
            else:
                # print(grad_)
                para_.grad.data = grad_

        optimizer.step()
        pass
        #for name, param in optimizer.named_parameters()


class AIA(torchAttacker):
    def __init__(self):
        super(AIA, self).__init__()
        # ml100k:lr=0.5,epoch=10
        pass

    def build_network(self):

        way = int(sys.argv[sys.argv.index('--way') + 1])

        if way == 1:
            print('random attack from 20% data')
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            
            sampled_idx = np.random.choice(np.where(np.sum(self.train_array > 0, 1) >= self.filler_num)[0],
                                       self.attack_num)
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sampled_idx[i])

            sampled_idx = new
            print(sampled_idx)
            print(len(sampled_idx))

            templates = self.train_array[sampled_idx]
        if way == 2:
            print('random attack from 50 target users')
            sampled_idx = np.random.choice(range(50), self.attack_num)
            print(sampled_idx)
            user = [5520, 5678, 1771, 4738, 317, 4962, 1338, 4975, 970, 3305, 5, 646, 1802, 2191, 2704, 3987, 789, 3734,
                5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903,
                587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
            dict_user_code = {}
            j = 0
            for i in user:
                dict_user_code[j] = i
                j+=1
            new = []
            for i in list(sampled_idx):
                new.append(dict_user_code[i])
            print(new)

            sign = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sampled_idx[i])

            sampled_idx = new
            print(sampled_idx)
            print(len(sampled_idx))

            target_user_array_path = './data/ml1m/target_user_train.npy'
            target_user_array = np.load(target_user_array_path)
            print(self.train_array[sampled_idx].shape)
            print(target_user_array.shape)
            templates = target_user_array[sampled_idx]
            print(templates.shape)
        if way == 3:
            print('our methods')
            user = [5520, 5678, 1771, 4738, 317, 4962, 1338, 4975, 970, 3305, 5, 646, 1802, 2191, 2704, 3987, 789, 3734,
                5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903,
                587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
            dict_user_code = {}
            j = 0
            for i in user:
                dict_user_code[i] = j
                j+=1
            # sample_100 = [5618, 5618, 623, 623, 362, 362, 362, 362, 1509, 1509, 1509, 1509, 5345, 1238, 1238, 3532, 3532, 587, 587, 3903, 3903, 3903, 3903, 3903, 1462, 1462, 1462, 1462, 1462, 4918, 4918, 5682, 5682, 5682, 5682, 2347, 2347, 2347, 425, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 286, 286, 286, 797, 797, 797, 797, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 5, 5, 5, 3305, 3305, 3305, 3305, 970, 970, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 5345, 5345, 5345, 5345, 5345, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 3532, 3532, 4926, 4926, 4926, 4926, 2347, 2347, 2347, 2347, 2347, 286, 286, 286, 286, 286, 286, 797, 797, 797, 797, 797, 789, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 2191, 3305, 3305, 3305, 3305, 3305, 3305, 970, 970, 970, 970, 4975, 4975, 4975, 4975, 4975, 4962, 4962, 4962, 4962, 1771, 1771, 1771, 1771, 1771, 5520, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 5618, 5618, 5618, 623, 623, 362, 362, 362, 362, 1509, 1509, 1509, 1509, 5345, 1238, 1238, 3532, 3532, 587, 587, 3903, 3903, 3903, 3903, 3903, 1462, 1462, 1462, 1462, 1462, 4918, 4918, 5682, 5682, 2347, 2347, 2347, 425, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 286, 286, 286, 797, 797, 797, 797, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 5, 5, 5, 3305, 3305, 3305, 3305, 970, 970, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 5520, 5520, 5520, 5520, 5520]
            # sample_100 =  [5618, 5618, 5618, 5618, 623, 4330, 4330, 4330, 4330, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 4572, 1238, 4434, 4434, 3532, 587, 3903, 3903, 3903, 3903, 4926, 2365, 1082, 1462, 4918, 4918, 4918, 4918, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 4770, 286, 286, 286, 797, 797, 797, 797, 3734, 3734, 3734, 3734, 789, 3987, 3987, 2704, 2191, 2191, 2191, 1802, 5, 5, 5, 5, 3305, 970, 970, 4975, 1338, 1338, 1338, 1338, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 5618, 5618, 623, 623, 4330, 4330, 4330, 4330, 362, 362, 362, 1509, 1509, 1509, 5854, 1238, 1238, 587, 3903, 3903, 3903, 4926, 4926, 4926, 1082, 1082, 1462, 1462, 1462, 4918, 5682, 2347, 2347, 425, 425, 425, 2597, 2597, 2597, 2340, 2340, 2340, 2340, 35, 35, 4770, 4770, 4770, 4770, 160, 160, 160, 286, 797, 797, 3734, 3734, 3734, 3734, 789, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 3305, 3305, 4975, 4975, 4975, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 4738, 4738, 4738, 4738, 5678, 5678, 5678, 5678, 5520]
            # sample_100 = [3582, 3582, 2678, 2678, 2678, 2678, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 362, 362, 362, 362, 362, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 587, 587, 587, 587, 587, 3903, 3903, 3903, 3903, 3903, 4926, 4926, 4926, 4926, 2365, 2365, 2365, 2365, 2365, 1082, 1082, 1082, 1082, 1462, 1462, 1462, 1462, 4918, 4918, 4918, 4918, 2347, 2347, 2347, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 797, 797, 797, 797, 3734, 3734, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 5, 5, 3305, 3305, 3305, 970, 970, 970, 970, 970, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5678, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 4330, 4330, 362, 5345, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 4434, 4434, 4434, 3532, 3532, 587, 587, 3903, 3903, 3903, 3903, 3903, 4926, 4926, 4926, 4926, 4926, 2365, 2365, 1082, 1082, 4918, 4918, 4918, 4918, 4918, 5682, 2347, 2347, 425, 425, 425, 425, 425, 2597, 2340, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 4770, 160, 286, 286, 286, 286, 797, 797, 797, 797, 797, 3734, 3734, 3734, 3734, 3734, 789, 789, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 646, 5, 5, 5, 5, 5, 3305, 3305, 3305, 3305, 3305, 970, 970, 970, 4975, 4975, 1338, 1338, 1338, 1338, 1338, 4962, 4962, 317, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 1771, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 623, 623, 4572, 4572, 1238, 1238, 4434, 4434, 3532, 3532, 587, 587, 4926, 4926, 2347, 2347, 35, 35, 286, 286, 286, 286, 797, 797, 797, 789, 789, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 3305, 3305, 970, 970, 970, 4975, 4975, 4962, 4962, 317, 317, 1771, 1771, 1771, 5520]
            # sample_100 = [5618, 623, 362, 362, 1509, 1509, 1509, 5854, 1082, 1082, 5682, 2347, 425, 425, 2597, 2597, 2597, 2340, 2340, 2340, 2340, 35, 35, 160, 160, 160, 286, 797, 789, 3987, 2704, 2704, 2191, 1802, 1802, 646, 5, 4975, 4975, 4975, 1338, 4962, 4962, 4962, 4962, 4962, 317, 317, 317, 5520]
            # sample_100 = [3582, 3582, 3582, 3582, 3582, 2678, 2678, 2678, 2678, 5618, 5618, 5618, 623, 623, 4330, 4330, 4330, 4330, 362, 362, 362, 1509, 1509, 1509, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 4572, 1238, 1238, 587, 3903, 3903, 3903, 4926, 4926, 4926, 1082, 1082, 1082, 1082, 1082, 3386, 3386, 3386, 3386, 3386, 1462, 1462, 1462, 1462, 1462, 4918, 5682, 5682, 5682, 2347, 2347, 425, 425, 425, 2597, 2597, 2597, 2597, 2597, 2340, 2340, 2340, 2340, 35, 35, 4770, 4770, 4770, 4770, 160, 160, 160, 286, 286, 286, 797, 797, 5017, 5017, 5017, 5017, 3734, 3734, 3734, 3734, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 3305, 3305, 4975, 4975, 4975, 4975, 4975, 1338, 4962, 4962, 4962, 4962, 4962, 317, 317, 317, 4738, 4738, 4738, 4738, 4738, 5678, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [3582, 3582, 3582, 3582, 3582, 2678, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 362, 1509, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 3532, 587, 587, 587, 587, 587, 3903, 4926, 4926, 4926, 4926, 4926, 2365, 1082, 3386, 1462, 4918, 5682, 5682, 5682, 5682, 5682, 2347, 425, 2597, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 797, 5017, 789, 789, 789, 789, 789, 3987, 2704, 2704, 2704, 2704, 2704, 2191, 5, 3305, 970, 4975, 1338, 4962, 4962, 4962, 4962, 4962, 317, 4738, 4738, 4738, 4738, 4738, 5678]
            sample_100 = [3582, 3582, 3582, 3582, 3582, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 587, 587, 587, 587, 587, 4926, 4926, 4926, 4926, 4926, 5682, 5682, 5682, 5682, 5682, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 5017, 5017, 5017, 5017, 5017, 789, 789, 789, 789, 789, 2704, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 970, 970, 970, 970, 970, 4975, 4975, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 4962, 4738, 4738, 4738, 4738, 4738]
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            
            
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sample_100[i])

            sample_100 = new
            print(len(sample_100))
            print(sample_100)
            
            import random
            random.shuffle(sample_100)
            
            sampled_100_idx = []
            for i in sample_100:
                sampled_100_idx.append(dict_user_code[i])
            sampled_idx = list(sampled_100_idx)
            print(sampled_idx)
            target_user_array_path = './data/ml1m/target_user_train.npy'
            target_user_array = np.load(target_user_array_path)
            print(self.train_array[sampled_idx].shape)
            print(target_user_array.shape)
            templates = target_user_array[sampled_idx]
            print(templates.shape)




        for (idx, template) in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num:]:
                templates[idx][iid] = 0.
      
        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)
   
        self.netG = RecsysGenerator(self.device, self.real_template).to(self.device)
      
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def train_G(self):

        self.netG.train()

        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)

        sur_predictions, sur_test_rmse = self.get_sur_predictions(fake_tensor)
        # print('sur_predictions\n')
        # print(sur_predictions)
        # print(sur_predictions.shape)
        G_loss = torch.tensor(0.).to(self.device)

        for target_id in self.target_id_list:
            print(target_id)
            self.target_users = np.where(self.train_array[:, target_id] == 0)[0]
            # self.target_users 所有对target id 没有评分的 users

            attack_target = np.zeros((len(self.target_users), self.n_items))
            # attack_target 张成的一个 矩阵 (0)
            attack_target[:, target_id] = 1.0
            # 另矩阵中全部的 target_id == 1.0
            self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
            higher_mask = (sur_predictions[self.target_users] >=
                           (sur_predictions[self.target_users, target_id].reshape([-1, 1]))).float()

            # print('higher_mask\n')
            # print(higher_mask)
            G_loss_sub = self.custimized_attack_loss(logits=sur_predictions[self.target_users] * higher_mask,
                                                     labels=self.attack_target).mean()
            G_loss += G_loss_sub

        self.update_params(G_loss / 10, self.G_optimizer)
        self.netG.eval()

        return G_loss.item() / 10

        self.netG.train()

        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)
      
        sur_predictions, sur_test_rmse = self.get_sur_predictions(fake_tensor)
        higher_mask = (sur_predictions[self.target_users] >=
                       (sur_predictions[self.target_users, self.target_id].reshape([-1, 1]))).float()
        G_loss = self.custimized_attack_loss(logits=sur_predictions[self.target_users] * higher_mask,
                                             labels=self.attack_target).mean()

        self.update_params(G_loss, self.G_optimizer)
        self.netG.eval()
      
        return G_loss.item()

    def train(self):

        log_to_visualize_dict = {}

        for epoch in range(self.epochs):
            if self.verbose and epoch % self.T == 0:
                datetime_begin = datetime.now()

            G_loss = self.train_G()
            if self.verbose and epoch % self.T == 0:
                train_time = (datetime.now() - datetime_begin).seconds

                metrics = self.test(victim='SVD', detect=False)

                print("epoch:%d\ttime:%ds" % (epoch, train_time), end='\t')
                print("G_loss:%.4f" % (G_loss), end='\t')
                print(metrics, flush=True)

        return log_to_visualize_dict
        pass

    def execute(self):

        self.prepare_data()

        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            log_to_visualize_dict = self.train()
            print("training done.")

            # self.save(self.model_path)

        #metrics = self.test(victim='all', detect=True)
        #print(metrics, flush=True)
        return


class AUSHplus(torchAttacker):
    def __init__(self):
        super(AUSHplus, self).__init__()
        print('Args:\n', self.args, '\n', flush=True)
        pass

    def build_network(self):

        way = int(sys.argv[sys.argv.index('--way') + 1])

        if way == 1:
            print('random attack from 20% data')
            # all the templates come from all the data
            sampled_idx = np.random.choice(range(self.n_users), self.attack_num)
            sign = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sampled_idx[i])

            sampled_idx = new
            print(sampled_idx)
            print(len(sampled_idx))

            templates = self.train_array[sampled_idx]
            
        if way == 2:
            print('random attack from 50 target users')
            sampled_idx = np.random.choice(range(50), self.attack_num)
            print(sampled_idx)
            sampled_idx = list(sampled_idx)
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            
            
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sampled_idx[i])

            sampled_idx = new
            print(sampled_idx)
            print(len(sampled_idx))
            target_user_array_path = './data/ml1m/target_user_train.npy'
            target_user_array = np.load(target_user_array_path)
            print(self.train_array[sampled_idx].shape)
            print(target_user_array.shape)
            templates = target_user_array[sampled_idx]
            print(templates.shape)
            
        if way == 3:
            print('our methods')
            user = [5520, 5678, 1771, 4738, 317, 4962, 1338, 4975, 970, 3305, 5, 646, 1802, 2191, 2704, 3987, 789, 3734,
                5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903,
                587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
            dict_user_code = {}
            j = 0
            for i in user:
                dict_user_code[i] = j
                j+=1
            
            #  x = [4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854,4926, 623, 587, 2191, 1338, 5854]
            #  x = [4926, 5618, 623, 587, 4770, 4572, 2704, 2191, 4738, 789, 1338, 3305, 35, 5678, 3734, 2340, 5854]
            #  sample_100 = np.random.choice(x,100)
            # sample_100 = [4926, 4926, 4926, 4926, 4926, 4926, 4926, 5618, 5618, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 587, 587, 587, 587, 587, 587, 587, 587, 4572, 4572, 4572, 4572, 4572, 2704, 2704, 2704, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 2191, 2191, 4738, 4738, 4738, 4738, 4738, 4738, 789, 789, 789, 789, 789, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 3305, 3305, 3305, 3305, 3305, 3305, 3305, 35, 35, 35, 35, 35, 35, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 5854, 5854, 5854]
            # sample_100 =[4926, 4926, 4926, 4926, 4926, 4926, 4926, 5618, 5618, 5618, 5618, 5618, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 587, 587, 587, 587, 587, 587, 587, 587, 4572, 4572, 4572, 4572, 4572, 2704, 2704, 2704, 2704, 2704, 2704, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 2191, 2191, 4738, 4738, 4738, 4738, 4738, 4738, 789, 789, 789, 789, 789, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 1338, 3305, 3305, 3305, 3305, 3305, 3305, 3305, 3305, 3305, 3305, 35, 35, 35, 35, 35, 35, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 5678, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 2340, 5854, 5854, 5854]
            # sample_100 =  [5854, 4738, 4975, 425, 1771, 5618, 3734, 1082, 2365, 4770, 2340, 4330, 5678, 5854, 35, 1082, 5854, 5678, 4770, 4975, 2191, 623, 970, 4572, 5017, 797, 646, 623, 4572, 5854, 2340, 4572, 4434, 4926, 4770, 1338, 1338, 623, 4330, 1238, 4926, 4770, 2191, 1338, 2340, 623, 3903, 623, 4770, 1338, 4572, 35, 587, 587, 1338, 2365, 2704, 5618, 4770, 587, 1338, 4770, 5618, 1509, 5618, 587, 4975, 2191, 970, 4975, 2340, 4926, 362, 587, 3532, 797, 160, 3532, 4434, 2340, 3987, 5854, 4738, 2347, 1509, 4926, 5618, 623, 3987, 5854, 3734, 4926, 2191, 2191, 4926, 4770, 425, 1338, 587, 623]
            # sample_100 =  [970, 646, 5678, 5618, 4572, 2340, 4770, 2191, 4770, 5618, 623, 4738, 4434, 3903, 1509, 2365, 2191, 587, 587, 1338, 797, 35, 4770, 797, 5618, 5017, 4975, 4738, 623, 2678, 587, 425, 1082, 362, 5678, 4975, 4572, 2678, 2678, 4926, 2678, 4572, 3532, 587, 4926, 4572, 970, 3734, 5618, 2340, 1238, 3532, 3987, 2365, 160, 1771, 3734, 4330, 1082, 4975, 4975, 35, 4330, 2340, 5618, 623, 2704, 425, 1338, 1338, 1338, 4434, 4770, 1509, 3987, 4926]
            # sample_100 =[4434, 3532, 587, 3532, 1771, 2191, 2191, 1338, 1238, 1238, 970, 1238, 5854, 1771, 797, 2191, 5854, 3532, 970, 1771, 5854, 3532, 797, 2191, 1238, 4434, 1771, 2191, 797, 4434, 1338, 1338, 3532, 4434, 1338, 587, 587, 797, 1771, 4434, 797, 1771, 4434, 1338, 797, 970, 4434, 970, 1238, 5854, 4434, 1238, 3532, 1338, 2191, 1338, 970, 1338, 1238, 5854, 970, 1771, 587, 587, 1238, 797, 970, 587, 5854, 797, 2191, 4434, 3532, 1771, 2191, 797, 5854, 5854, 2191, 1238, 970, 5854, 1771, 587, 2191, 5854, 587, 970, 3532, 587, 3532, 1338, 4434, 970, 1771, 1338, 587, 797, 3532, 1238]
            
            # sample_pool = [4926, 5618, 623, 587, 3582, 4770, 4572, 4975, 2704, 2191, 4738, 3386, 1238, 789, 1338, 3305, 797, 970, 35, 1771, 3987, 5678, 4434, 3532, 3734, 1462, 2340, 5854]
            # sample_pool =[4926, 5618, 623, 587, 3582, 4770, 4572, 2704, 2191, 4738, 1238, 789, 1338, 3305, 797, 970, 35, 1771, 3987, 5678, 4434, 3532, 3734, 1462, 2340, 5854]
            # sample_pool =[4926, 5618, 623, 587, 4770, 4572, 2704, 2191, 4738, 1238, 789, 1338, 3305, 797, 970, 35, 1771, 5678, 4434, 3532, 3734, 2340, 5854]
            # sample_pool =[4926, 623, 587, 4770, 2191, 1238, 1338, 797, 970, 1771, 4434, 3532, 2340, 5854]
            # sample_pool =[4926, 623, 587, 2191, 1238, 1338, 797, 970, 1771, 4434, 3532, 5854]
            # sample_pool =[4926, 4962, 623, 5618, 587, 4770, 2704, 4975, 4572, 5345, 1238, 2191, 3386, 789, 2365, 1338, 3305, 797, 970, 5682, 4918, 3987, 1771, 2678, 4434, 3532, 4330, 362, 3734, 1462, 2340, 5520, 5854]
            # sample_pool =[4926, 4962, 623, 5618, 4770, 2704, 4975, 4572, 5345, 1238, 2191, 789, 1338, 3305, 797, 970, 5682, 4918, 3987, 1771, 4434, 3532, 4330, 3734, 2340, 5520, 5854]
            # sample_pool =[4926, 4962, 623, 5618, 4770, 2704, 4975, 4572, 5345, 1238, 2191, 789, 3305, 797, 970, 4918, 3987, 1771, 4434, 3532, 4330, 3734, 2340, 5854]
            # sample_pool =[4926, 4962, 623, 5618, 4770, 2704, 4975, 4572, 5345, 1238, 2191, 789, 797, 970, 3987, 1771, 4434, 3532, 3734, 2340, 5854]
            # sample_pool =[4926, 623, 5618, 2704, 4572, 1238, 2191, 797, 970, 1771, 4434, 3532, 2340]
            # sample  = np.random.choice(sample_pool,100)
            # sample_100 = list(sample)
            # sample_100 = [5618, 5618, 623, 623, 362, 362, 362, 362, 1509, 1509, 1509, 1509, 5345, 1238, 1238, 3532, 3532, 587, 587, 3903, 3903, 3903, 3903, 3903, 1462, 1462, 1462, 1462, 1462, 4918, 4918, 5682, 5682, 5682, 5682, 2347, 2347, 2347, 425, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 286, 286, 286, 797, 797, 797, 797, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 5, 5, 5, 3305, 3305, 3305, 3305, 970, 970, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 5618, 5618, 5618, 623, 623, 362, 362, 362, 362, 1509, 1509, 1509, 1509, 5345, 1238, 1238, 3532, 3532, 587, 587, 3903, 3903, 3903, 3903, 3903, 1462, 1462, 1462, 1462, 1462, 4918, 4918, 5682, 5682, 2347, 2347, 2347, 425, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 286, 286, 286, 797, 797, 797, 797, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 3987, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 5, 5, 5, 3305, 3305, 3305, 3305, 970, 970, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [3582, 3582, 2678, 2678, 2678, 2678, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 362, 362, 362, 362, 362, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 587, 587, 587, 587, 587, 3903, 3903, 3903, 3903, 3903, 4926, 4926, 4926, 4926, 2365, 2365, 2365, 2365, 2365, 1082, 1082, 1082, 1082, 1462, 1462, 1462, 1462, 4918, 4918, 4918, 4918, 2347, 2347, 2347, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 797, 797, 797, 797, 3734, 3734, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 646, 646, 646, 646, 5, 5, 5, 5, 3305, 3305, 3305, 970, 970, 970, 970, 970, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5678, 5678, 5678, 5678, 5678, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [623, 623, 623, 1238, 1238, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 286, 286, 286, 286, 797, 797, 797, 797, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 3305, 3305, 3305, 970, 970, 4975, 1771, 1771, 1771, 1771, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [3582, 3582, 5618, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 5345, 5345, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 3532, 3532, 3532, 3532, 587, 587, 587, 587, 587, 3903, 3903, 3903, 3903, 3903, 4926, 4926, 4926, 4926, 2365, 2365, 2365, 2365, 2365, 1462, 1462, 1462, 1462, 4918, 4918, 4918, 4918, 2347, 2347, 2347, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 2340, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 797, 797, 797, 797, 3734, 3734, 789, 789, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2191, 2191, 2191, 2191, 646, 646, 646, 646, 3305, 3305, 3305, 970, 970, 970, 970, 970, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 317, 317, 317, 317, 4738, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5520, 5520, 5520, 5520, 5520]
            # sample_100 = [2678, 2678, 2678, 2678, 5618, 5618, 5618, 623, 623, 623, 4330, 4330, 4330, 362, 362, 362, 362, 1509, 1509, 1509, 5345, 5345, 5854, 5854, 5854, 5854, 5854, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 4434, 4434, 4434, 4434, 4434, 3532, 587, 3903, 3903, 3903, 3903, 1082, 1082, 1082, 1082, 1082, 3386, 3386, 3386, 3386, 1462, 1462, 1462, 1462, 4918, 5682, 5682, 5682, 5682, 2347, 2347, 425, 425, 425, 425, 2597, 2597, 2597, 2340, 2340, 2340, 35, 35, 35, 35, 35, 4770, 4770, 4770, 4770, 160, 160, 160, 286, 286, 797, 797, 797, 3734, 3734, 3734, 3734, 789, 789, 789, 3987, 3987, 3987, 3987, 2704, 2704, 2704, 2191, 2191, 2191, 1802, 1802, 1802, 1802, 646, 646, 646, 5, 5, 5, 5, 3305, 3305, 3305, 970, 4975, 4975, 4975, 1338, 1338, 1338, 4962, 4962, 4962, 317, 317, 317, 4738, 4738, 4738, 1771, 1771, 1771, 1771, 5678, 5678, 5520, 5520, 5520, 5520]
            # sample_100 = [5618, 623, 362, 362, 362, 1509, 1509, 1509, 1238, 3532, 587, 1462, 1462, 1462, 1462, 4918, 5682, 2347, 2347, 425, 425, 425, 425, 2340, 2340, 2340, 35, 35, 35, 797, 797, 797, 789, 789, 2191, 646, 646, 646, 5, 5, 3305, 970, 1338, 317, 317, 317, 5520, 5520, 5520, 5520]
            # sample_100 = [3582, 3582, 3582, 3582, 3582, 2678, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 362, 1509, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 3532, 587, 587, 587, 587, 587, 3903, 4926, 4926, 4926, 4926, 4926, 2365, 1082, 3386, 1462, 4918, 5682, 5682, 5682, 5682, 5682, 2347, 425, 2597, 35, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 797, 5017, 789, 789, 789, 789, 789, 3987, 2704, 2704, 2704, 2704, 2704, 2191, 5, 3305, 970, 4975, 1338, 4962, 4962, 4962, 4962, 4962, 317, 4738, 4738, 4738, 4738, 4738, 5678]
            sample_100 = [3582, 3582, 3582, 3582, 3582, 5618, 5618, 5618, 5618, 5618, 623, 623, 623, 623, 623, 5345, 5345, 5345, 5345, 5345, 4572, 4572, 4572, 4572, 4572, 1238, 1238, 1238, 1238, 1238, 587, 587, 587, 587, 587, 4926, 4926, 4926, 4926, 4926, 5682, 5682, 5682, 5682, 5682, 4770, 4770, 4770, 4770, 4770, 286, 286, 286, 286, 286, 5017, 5017, 5017, 5017, 5017, 789, 789, 789, 789, 789, 2704, 2704, 2704, 2704, 2704, 2191, 2191, 2191, 2191, 2191, 970, 970, 970, 970, 970, 4975, 4975, 4975, 4975, 4975, 1338, 1338, 1338, 1338, 1338, 4962, 4962, 4962, 4962, 4962, 4738, 4738, 4738, 4738, 4738]
            sign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            
            
            new = []
            for i in range(len(sign)):
                if sign[i] == 0:
                    new.append(sample_100[i])

            sample_100 = new
            print(len(sample_100))
            print(sample_100)
            import random
            random.shuffle(sample_100)
            
            sampled_100_idx = []
            for i in sample_100:
                sampled_100_idx.append(dict_user_code[i])
            sampled_idx = list(sampled_100_idx)
            print(sampled_idx)
            target_user_array_path = './data/ml1m/target_user_train.npy'
            target_user_array = np.load(target_user_array_path)
            print(self.train_array[sampled_idx].shape)
            print(target_user_array.shape)
            templates = target_user_array[sampled_idx]
            print(templates.shape)
            
        for (idx, template) in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num:]:
                templates[idx][iid] = 0.

        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)

        self.netG = DiscretGenerator_AE_1(self.device, p_dims=[self.n_items, 125]).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)

        self.netD = Discriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = optim.Adam(self.netD.parameters(), lr=self.lr_D)
        pass

    def train_D(self):

        self.netD.train()

        _, fake_tensor = self.netG(self.real_template)
        fake_tensor = fake_tensor.detach()

        D_loss_list = []
        for real_tensor in self.data_loader:

            real_tensor = real_tensor.to(self.device)[:self.attack_num]
            # forward
            self.D_optimizer.zero_grad()
            D_real = self.netD(real_tensor)
            D_fake = self.netD(fake_tensor)
            # loss
            D_real_loss = nn.BCELoss()(D_real,
                                       torch.ones_like(D_real).to(self.device))
            D_fake_loss = nn.BCELoss()(D_fake,
                                       torch.zeros_like(D_fake).to(self.device))
            D_loss = D_real_loss + D_fake_loss
            # backward
            D_loss.backward()
            self.D_optimizer.step()

            D_loss_list.append(D_loss.item())
            #
            # break
        self.netD.eval()

        return np.mean(D_loss_list)

    def train_G(self, adv=True, attack=True):

        self.netG.train()

        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)

        G_adv_loss = torch.tensor(0.)
        if adv:
            print("######i run here#########")
            G_adv_loss = nn.BCELoss(reduction='mean')(self.netD(fake_tensor),
                                                      torch.ones(fake_tensor.shape[0], 1).to(self.device))
            # """crossEntropy loss"""
            # real_labels_flatten = self.real_template.flatten().type(torch.long)
            # fake_logits_flatten = fake_tensor_distribution.reshape([-1, 5])
            # G_rec_loss = nn.CrossEntropyLoss()(fake_logits_flatten[real_labels_flatten > 0],
            #                                    real_labels_flatten[real_labels_flatten > 0] - 1)
            G_adv_loss = G_adv_loss  # + G_rec_loss
            print("G_adv %s"%G_adv_loss)

        real_labels_flatten = self.real_template.flatten().type(torch.long)
       
        MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                               real_labels_flatten[real_labels_flatten > 0])

        G_attack_loss = torch.tensor(0.).to(self.device)
        if attack:
            
            
            # y_grad = torch.autograd.grad(fake_tensor, [x.cuda() for x in self.G_optimizer.param_groups[0]['params']], retain_graph=True)[0]
            # print(y_grad)
            sur_predictions, sur_test_rmse = self.get_sur_predictions(fake_tensor)
            for target_id in self.target_id_list:
                # print(target_id)
                self.target_users = np.where(self.train_array[:, target_id] == 0)[0]
                # self.target_users 所有对target id 没有评分的 users

                attack_target = np.zeros((len(self.target_users), self.n_items))
                # attack_target 张成的一个 矩阵 (0)
                attack_target[:, target_id] = 1.0
                # 另矩阵中全部的 target_id == 1.0
                self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
                higher_mask = (sur_predictions[self.target_users] >=
                               (sur_predictions[self.target_users, target_id].reshape([-1, 1]))).float()

                G_loss_sub = self.custimized_attack_loss(logits=sur_predictions[self.target_users] * higher_mask,
                                                         labels=self.attack_target).mean()
                G_attack_loss += G_loss_sub
            G_attack_loss = G_attack_loss / 10
            
            
            

        G_loss = G_adv_loss + G_attack_loss

        self.update_params(G_loss, self.G_optimizer)
        self.netG.eval()
        # print("G_loss_item %s" %(G_loss.item()))
        # print("G_attack_loss.item() %s" % (G_attack_loss.item()))

        mean_score = fake_tensor[fake_tensor > 0].mean().item()
        return (G_loss.item(), MSELoss.item(), G_attack_loss.item(), mean_score)

    def pretrain_G(self):

        self.netG.train()
        G_loss_list = []
        for real_tensor in self.data_loader:
            # input data
            real_tensor = real_tensor.to(self.device)
            #print(real_tensor)
            #print(real_tensor.size())
            # forward
            fake_tensor_distribution, fake_tensor = self.netG(real_tensor)
            # crossEntropy loss
            real_labels_flatten = real_tensor.flatten().type(torch.long)
            #print(real_labels_flatten)
            #print(real_labels_flatten.size())
            fake_logits_flatten = fake_tensor_distribution.reshape([-1, 5])
            #print(fake_logits_flatten[real_labels_flatten > 0])
            #print(real_labels_flatten[real_labels_flatten > 0] - 1)
            #print(fake_logits_flatten[real_labels_flatten > 0].size())
            #print((real_labels_flatten[real_labels_flatten > 0] - 1).size())
            G_rec_loss = nn.CrossEntropyLoss()(fake_logits_flatten[real_labels_flatten > 0],
                                               real_labels_flatten[real_labels_flatten > 0] - 1)
            G_loss = G_rec_loss
            MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                                   real_labels_flatten[real_labels_flatten > 0])
            #backword
            self.update_params(G_loss.cuda(), self.G_optimizer)
            G_loss_list.append(G_loss.item())
        self.netG.eval()
        return (np.mean(G_loss_list), MSELoss)

    def train(self):

        log_to_visualize_dict = {}
        
        print('======================pretrain begin======================')
        print('pretrain G......')
        pretrain_epoch = 1 if self.data_set == 'automotive' else 1
        for i in range(pretrain_epoch):
            G_loss, MSELoss = self.pretrain_G()
            if i % 5 == 0:
                print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))
        
        
        print('pretrain D......')
        for i in range(5):
            D_loss = self.train_D()
            print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')
        
        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)

            for epoch_gan_d in range(5):
                D_loss = self.train_D()
                print("D_loss:%.8f" % (D_loss))

            for epoch_gan_g in range(1):  # 5/1
                list = []
                _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
                _, fake_tensor = self.netG(self.real_template)
                list.append(fake_tensor)
                if epoch !=0:
                    print(fake_tensor[epoch].equal(fake_tensor[epoch-1]))


                # metrics = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                # metrics = "%.4f" % metrics
                # print(metrics)

            for epoch_surrogate in range(50):
                
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()

                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)

                G_loss = G_adv_loss + G_rec_loss

                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    metrics = ''
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("D_loss:%.8f\tG_loss:%.8f" % (D_loss, G_loss), end='\t')
                    print(metrics, metrics2, flush=True)

            metrics = self.test(victim='SVD', detect=False)
            # print(metrics, flush=True)

        return log_to_visualize_dict
        pass

    def execute(self):

        print("-------------suucsbaukhaik------")

        self.prepare_data()
        # Build and compile GAN Network
        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            log_to_visualize_dict = self.train()
            print("training done.")

            # self.save(self.model_path)

        # metrics = self.test(victim='all', detect=True)
        # print(metrics, flush=True)
        return


class AUSHplus_SR(AUSHplus):
    def __init__(self):
        super(AUSHplus_SR, self).__init__()
        pass

    def build_network(self):
        super(AUSHplus_SR, self).build_network()
        self.netG = RoundGenerator_AE_1(self.device, p_dims=[self.n_items, 125]).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def pretrain_G(self):
        self.netG.train()
        G_loss_list = []
        for real_tensor in self.data_loader:
            # input data
            real_tensor = real_tensor.to(self.device)
            # forward
            _, fake_tensor = self.netG(real_tensor)
            """MSELoss"""
            real_labels_flatten = real_tensor.flatten()
            MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                                   real_labels_flatten[real_labels_flatten > 0])
            G_loss = MSELoss
            """train"""
            self.update_params(G_loss, self.G_optimizer)
            G_loss_list.append(G_loss.item())

        self.netG.eval()
        return (np.mean(G_loss_list), MSELoss)


class AUSHplus_woD(AUSHplus):
    def __init__(self):
        super(AUSHplus_woD, self).__init__()
        pass

    def train(self):
        log_to_visualize_dict = {}
        print('======================pretrain begin======================')
        print('pretrain G......')
        for i in range(1):  # 15
            G_loss, MSELoss = self.pretrain_G()
            if i % 5 == 0:
                print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))
        # print('pretrain D......')
        # for i in range(5):
        #     D_loss = self.train_D()
        #     print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')
        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)
            # for epoch_gan_d in range(5):
            #     D_loss = self.train_D()
            #     print("D_loss:%.4f" % (D_loss))
            # for epoch_gan_g in range(1):  # 5
            #     _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
            # =============================
            for epoch_surrogate in range(100):
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()
                # ================================
                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)
                G_loss = G_rec_loss  # G_adv_loss + G_rec_loss
                # ================================

                if self.verbose and epoch_surrogate % self.T == 0:
                    # =============================
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    # =============================
                    metrics = ''
                    # if epoch_surrogate % 10 == 0:
                    #     metrics = self.test(victim='NeuMF', detect=False)
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    #
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("G_loss:%.4f" % (G_loss), end='\t')
                    print(metrics, metrics2, flush=True)
                # =============================
            metrics = self.test(victim='SVD', detect=False)
            print(metrics, flush=True)
        # print(self.test(victim='all', detect=True))
        # exit()
        return log_to_visualize_dict
        pass


class AUSHplus_SF(AUSHplus):
    def __init__(self):
        super(AUSHplus_SF, self).__init__()
        pass

    def build_network(self):
        super(AUSHplus_SF, self).build_network()
        self.netG = DiscretRecsysGenerator_1(self.device, self.real_template).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def train(self):
        log_to_visualize_dict = {}
        print('======================pretrain begin======================')
        # print('pretrain G......')
        # for i in range(15):
        #     G_loss, MSELoss = self.pretrain_G()
        #     if i % 5 == 0:
        #         print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))
        print('pretrain D......')
        for i in range(5):
            D_loss = self.train_D()
            print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')
        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)
            for epoch_gan_d in range(5):
                D_loss = self.train_D()
                print("D_loss:%.4f" % (D_loss))
            for epoch_gan_g in range(1):  # 5
                _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
            # =============================
            for epoch_surrogate in range(100):
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()
                # ================================
                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)
                G_loss = G_adv_loss + G_rec_loss
                # ================================

                if self.verbose and epoch_surrogate % self.T == 0:
                    # =============================
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    # =============================
                    metrics = ''
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    #
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("D_loss:%.4f\tG_loss:%.4f" % (D_loss, G_loss), end='\t')
                    print(metrics, metrics2, flush=True)
        # print(self.test(victim='all', detect=True))
        # exit()
        return log_to_visualize_dict
        pass


class AUSHplus_inseg(AUSHplus):
    def __init__(self):
        super(AUSHplus_inseg, self).__init__()
        print(self.test())
        exit()
        pass

    def prepare_data(self):
        super(AUSHplus_inseg, self).prepare_data()
        #
        path = './data/%s/%s_target_users' % (self.data_set, self.data_set)
        with open(path) as lines:
            for line in lines:
                if int(line.split('\t')[0]) == self.target_id:
                    self.target_users = list(map(int, line.split('\t')[1].split(',')))
                    target_users = list(self.target_users) + list(map(int, line.split('\t')[1].split(','))) * 4
                    self.target_users = np.array(target_users)
                    break
        # self.target_users = np.where(self.train_array[:, self.target_id] == 0)[0]
        attack_target = np.zeros((len(self.target_users), self.n_items))
        attack_target[:, self.target_id] = 1.0
        self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
        pass

    def test(self, victim='SVD', detect=False, fake_array=None):
        """

        :param victim:
        :param evalutor:
        :return:
        """

        # self.generate_injectedFile(fake_array)
        """detect"""
        # res_detect_list = self.detect(detect)
        # res_detect = '\t'.join(res_detect_list)

        """attack"""
        # if self.target_id == 594:
        #     all_victim_models = ['NeuMF', 'IAutoRec']
        # if self.target_id in [1257]:
        #     all_victim_models = ['NeuMF', 'IAutoRec', 'UAutoRec']
        if self.target_id in [1077]:
            all_victim_models = ['NeuMF']
        # if victim is False:
        #     res_attack = ''
        # elif victim in all_victim_models:
        #     """攻击victim model"""
        #     self.attack(victim)
        #     #
        #     """对比结果"""
        #     res_attack_list = self.evaluate(victim)
        #     res_attack = '\t'.join(res_attack_list)
        #     #
        # else:
        #     if victim == 'all':
        if True:
            victim_models = all_victim_models
            # else:
            #     victim_models = victim.split(',')
            res_attack_list = []
            # SlopeOne,SVD,NMF,IAutoRec,UAutoRec,NeuMF
            for victim_model in victim_models:
                
                self.attack(victim_model)
                #
                
                cur_res_list = self.evaluate(victim_model)
                #
                res_attack_list.append('\t:\t'.join([victim_model, '\t'.join(cur_res_list)]))
            res_attack = '\n' + '\n'.join(res_attack_list)
        res = '\t'.join([res_attack])
        return res
