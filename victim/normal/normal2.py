# -*- coding: utf-8 -*-
# @Time       : 2020/11/27 17:20
# @Author     : chensi
# @File       : Recommender.py
# @Software   : PyCharm
# @Desciption : None


import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# def available_GPU():
#     import subprocess
#     import numpy as np
#     nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
#     total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
#     total_GPU = total_GPU_str.split('\n')
#     total_GPU = np.array([int(device_i) for device_i in total_GPU])
#     avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
#     avail_GPU = avail_GPU_str.split('\n')
#     avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
#     avail_GPU = avail_GPU / total_GPU
#     return np.argmax(avail_GPU)
import scipy
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(available_GPU())
# except:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'



tf = None
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import random
import numpy as np
import torch

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils import DataLoader
import numpy as np
import pandas as pd
import argparse
import surprise
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import PredefinedKFold


class Recommender(object):

    def __init__(self):
        self.args = self.parse_args()
        # 路径
        self.train_path = self.args.train_path
        self.test_path = self.args.test_path
        self.model_path = self.args.model_path
        self.target_prediction_path_prefix = self.args.target_prediction_path_prefix
        # 攻击
        self.target_id_list = list(map(int, self.args.target_ids.split(',')))
        self.topk_list = list(map(int, self.args.topk.split(',')))
        #
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_id)
        pass

    @staticmethod
    def parse_args():

        parser = argparse.ArgumentParser(description="Run Recommender.")
        parser.add_argument('--data_set', type=str, default='ml100k')  # , required=True)
        # 路径
        parser.add_argument('--train_path', type=str,
                            default='./data/ml100k/ml100k_train.dat')  # , required=True)
        parser.add_argument('--test_path', type=str,
                            default='./data/ml100k/ml100k_test.dat')  # , required=True)
        parser.add_argument('--model_path', type=str,
                            default='./results/model_saved/automotive/automotive_NeuMF_AUSHplus_round_119')  # , required=True)
        parser.add_argument('--target_prediction_path_prefix', type=str,
                            default='./results/performance/mid_results/ml100k_Recommender')  # , required=True)

        # 攻击
        parser.add_argument('--target_ids', type=str, default='0')  # , required=True)
        parser.add_argument('--topk', type=str, default='5,10,20,50')
        #
        parser.add_argument('--cuda_id', type=int, default=0)

        return parser

    def prepare_data(self):
        self.dataset_class = DataLoader(self.train_path, self.test_path)

        self.train_data_df, self.test_data_df, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()
        self.train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.test_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.test_data_df, self.n_users, self.n_items)
        pass

    def build_network(self):
        print('build Recommender model graph.')
        raise NotImplemented

    def train(self):
        print('train.')
        raise NotImplemented

    def test(self):
        print('test.')
        raise NotImplemented

    def execute(self):
        print('generate target item performace on a trained Recommender model.')
        raise NotImplemented

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        raise NotImplemented

    def generate_target_result(self):
        train_data_array = self.train_matrix.toarray()
        for target_id in self.target_id_list:
            # mask掉已评分用户以及未评分用户的已评分商品
            mask = np.zeros_like(train_data_array)
            mask[np.where(train_data_array[:, target_id])[0]] = float('inf')
            # 找到测试数据
            test_uids, test_iids = np.where((train_data_array + mask) == 0)
            # 预测
            test_predRatings = self.predict(test_uids, test_iids)
            # 构建dataframe
            predResults = pd.DataFrame({'user_id': test_uids,
                                        'item_id': test_iids,
                                        'rating': test_predRatings
                                        })
            # 为每个未评分计算预测分和HR
            predResults_target = np.zeros([len(predResults.user_id.unique()), len(self.topk_list) + 2])
            for idx, (user_id, pred_result) in enumerate(predResults.groupby('user_id')):
                pred_value = pred_result[pred_result.item_id == target_id].rating.values[0]
                sorted_recommend_list = pred_result.sort_values('rating', ascending=False).item_id.values
                new_line = [user_id, pred_value] + [1 if target_id in sorted_recommend_list[:k] else 0 for k in
                                                    self.topk_list]
                predResults_target[idx] = new_line

            np.save('%s_%d' % (self.target_prediction_path_prefix, target_id), predResults_target)


class NeuMF(Recommender):
    def __init__(self):
        super(NeuMF, self).__init__()
        self.restore_model = self.args.restore_model
        self.learning_rate = self.args.learning_rate
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        self.reg_rate = self.args.reg_rate
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.num_factor = self.args.num_factor
        self.num_factor_mlp = self.args.num_factor_mlp
        self.hidden_dimension = self.args.hidden_dimension
        #
        print("NeuMF.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=0.5)
        parser.add_argument('--reg_rate', type=float, default=0.01)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--num_factor', type=int, default=10)
        parser.add_argument('--num_factor_mlp', type=int, default=64)
        parser.add_argument('--hidden_dimension', type=int, default=10)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--display_step', type=int, default=1000)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        # self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.P = tf.Variable(tf.random_normal([self.n_users, self.num_factor], stddev=0.01), dtype=tf.float32)
        self.Q = tf.Variable(tf.random_normal([self.n_items, self.num_factor], stddev=0.01), dtype=tf.float32)

        self.mlp_P = tf.Variable(tf.random_normal([self.n_users, self.num_factor_mlp], stddev=0.01), dtype=tf.float32)
        self.mlp_Q = tf.Variable(tf.random_normal([self.n_items, self.num_factor_mlp], stddev=0.01), dtype=tf.float32)

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        mlp_user_latent_factor = tf.nn.embedding_lookup(self.mlp_P, self.user_id)
        mlp_item_latent_factor = tf.nn.embedding_lookup(self.mlp_Q, self.item_id)

        _GMF = tf.multiply(user_latent_factor, item_latent_factor)

        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(
            inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
            units=self.num_factor_mlp * 2,
            kernel_initializer=tf.random_normal_initializer,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        layer_2 = tf.layers.dense(
            inputs=layer_1,
            units=self.hidden_dimension * 8,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        layer_3 = tf.layers.dense(
            inputs=layer_2,
            units=self.hidden_dimension * 4,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        layer_4 = tf.layers.dense(
            inputs=layer_3,
            units=self.hidden_dimension * 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        _MLP = tf.layers.dense(
            inputs=layer_4,
            units=self.hidden_dimension,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        # self.pred_y = tf.nn.sigmoid(tf.reduce_sum(tf.concat([_GMF, _MLP], axis=1), 1))
        self.pred_rating = tf.reduce_sum(tf.concat([_GMF, _MLP], axis=1), 1)

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + \
                    self.reg_rate * (tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) +
                                     tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))
        #
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self):
        super(NeuMF, self).prepare_data()
        #
        self.train_matrix_coo = self.train_matrix.tocoo()
        #
        self.user = self.train_matrix_coo.row.reshape(-1)
        self.item = self.train_matrix_coo.col.reshape(-1)
        self.rating = self.train_matrix_coo.data

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        # train
        loss = []
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss_ = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.y: batch_rating})

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        test_data = self.test_matrix.todok()
        #
        uids = np.array(list(test_data.keys()))[:, 0]
        iids = np.array(list(test_data.keys()))[:, 1]
        ground_truth = np.array(list(test_data.values()))
        #
        pred_rating = self.predict(uids, iids)
        #
        rmse = np.sqrt(np.mean((pred_rating - ground_truth) ** 2))
        mae = np.mean(np.abs(pred_rating - ground_truth))
        return rmse, mae

    def predict(self, user_ids, item_ids):
        if len(user_ids) < self.batch_size:
            return self.sess.run(self.pred_rating,
                                 feed_dict={
                                     self.user_id: user_ids,
                                     self.item_id: item_ids}
                                 )
        # predict by batch
        total_batch = math.ceil(len(user_ids) / self.batch_size)
        user_ids, item_ids = list(user_ids), list(item_ids)
        pred_rating = []
        for i in range(total_batch):
            batch_user = user_ids[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_ids[i * self.batch_size:(i + 1) * self.batch_size]
            # predict
            batch_pred_rating = self.sess.run(self.pred_rating,
                                              feed_dict={
                                                  self.user_id: batch_user,
                                                  self.item_id: batch_item}
                                              )
            pred_rating += list(batch_pred_rating)
        return pred_rating

    def restore_user_embedding(self):
        # 数据准备
        self.prepare_data()
        self.n_users += 50
        # ================

        attackers = ['AUSHplus_Dis_xiaorong', 'AUSHplus', 'SegmentAttacker', 'BandwagonAttacker',
                     'AverageAttacker', 'RandomAttacker',
                     'AUSH', 'RecsysAttacker',
                     'DCGAN', 'WGAN']
        #
        targets = [62]  # [119, 422, 594, 884, 1593]
        with tf.Session() as sess:
            self.sess = sess

            self.build_network()

            sess.run(tf.global_variables_initializer())

            for target in targets:
                for attacker in attackers:
                    self.model_path = './results/model_saved/ml100k/ml100k_NeuMF_%s_%d' % (attacker, target)
                    if not os.path.exists(self.model_path + '.meta'):
                        continue

                    self.restore(self.model_path)
                    print("loading done.")
                    user_embedding, user_embedding_mlp = self.sess.run([self.P, self.mlp_P])
                    save_path = self.model_path + '_user_embed'
                    save_path = save_path.replace('model_saved', 'performance\mid_results')
                    np.save(save_path, user_embedding)
                    np.save(save_path + '_mlp', user_embedding_mlp)
            return

    def execute(self):


        # 无需更改
        self.prepare_data()
        # ================

        # tensorflow session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            self.build_network()


            init = tf.global_variables_initializer()
            sess.run(init)


            if self.restore_model:
                self.restore(self.model_path)
                print("loading done.")


            else:
                loss_prev = float('inf')
                for epoch in range(self.epochs):
                    loss_cur = self.train()
                    if True:  # self.verbose and epoch % self.T == 0:
                        print("epoch:\t", epoch, "\tloss:\t", loss_cur, flush=True)
                    if abs(loss_cur - loss_prev) < math.exp(-5):
                        break
                    loss_prev = loss_cur


                self.save(self.model_path)
                print("training done.")


            rmse, mae = self.test()
            print("RMSE : %.4f,\tMAE : %.4f" % (rmse, mae))

            self.generate_target_result()

            return







