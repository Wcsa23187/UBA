import random
import numpy as np
import torch
import sys
seed = int(sys.argv[sys.argv.index('--allseed') + 1])

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import os, sys

from utils import DataLoader
import argparse, os, shutil

PythonCommand = 'D:\Anaconda3\envs\py38_tf2\python'
PythonCommand = PythonCommand if os.path.exists(PythonCommand + '.exe') else 'python'

class Attacker(object):
    def __init__(self):
        self.args = self.parse_args()
        self.data_set = self.args.data_set
        self.target_id = self.args.target_id
        self.attack_num = self.args.attack_num
        self.filler_num = self.args.filler_num
        # self.injected_path = self.args.injected_path
        # self.target_id_item = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        # self.target_id_item = [1551]
        # self.target_id_item = [1104]
        # self.target_id_item = [1606]
        # self.target_id_item = [68]
        # self.target_id_item = [2577]
       
        self.target_id_item = [int(sys.argv[sys.argv.index('--oneitem') + 1])]
        # 2577

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Attacker.")
        # filmTrust/filmTrust/automotive
        parser.add_argument('--data_set', type=str, default='ml100k')  # , required=True)
        # parser.add_argument('--target_ids', nargs='+', type=float)
        parser.add_argument('--target_id', type=int, default=62)  # , required=True)
        parser.add_argument('--attack_num', type=int, default=50)  # )
        # ml100k:90/automotive:4
        parser.add_argument('--filler_num', type=int, default=36)  # , required=True)
        parser.add_argument('--cuda_id', type=int, default=0)  # , required=True)
        #
        # parser.add_argument('--injected_path', type=str,
        #                     default='./results/data_attacked/ml100k/ml100k_attack_62.data')  # , required=True)
        return parser

    def prepare_data(self):
        self.path_train = './data/%s/%s_50user_train.csv' % (self.data_set, self.data_set)
        path_test = './data/%s/%s_50user_test.csv' % (self.data_set, self.data_set)
        self.dataset_class = DataLoader(self.path_train, path_test)
        self.train_data_df, _, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()

    def build_network(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def test(self, victim='SVD', detect=False, fake_array=None):
        

        self.generate_injectedFile(fake_array)

        

    def evaluate(self, victim):

        attacker, recommender = self.__class__.__name__, victim
        #
        args_dict = {
            'data_set': self.data_set,
            'test_path': './data/%s/%s_test.dat' % (self.data_set, self.data_set),
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            'attacker': attacker,
            #
        }
        #
        path_res_before_attack = './results/performance/mid_results/%s/%s_%s_%d.npy' % (
            self.data_set, self.data_set, recommender, self.target_id)

        result_list = []

        cur_args_dict = {
            'exe_model_lib': 'evaluator',
            'exe_model_class': 'Attack_Effect_Evaluator',
            'data_path_clean': './results/performance/mid_results/%s/%s_%s_%d.npy' % (
                self.data_set, self.data_set, recommender, self.target_id),
            'data_path_attacked': './results/performance/mid_results/%s/%s_%s_%s_%d.npy' % (
                self.data_set, self.data_set, recommender, attacker, self.target_id),
        }
        cur_args_dict.update(args_dict)
        args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
        print("开始进行评估")
        print(args_str)
        return_file = os.popen('%s ./utils/main.py %s' % (PythonCommand, args_str))
        # time.sleep(5)
        return_str = return_file.read()

        return_str = return_str[return_str.find('result begin') + 13:return_str.find('result end') - 2]
        result_list += [return_str]
        print("evaluate 完成后输出结果")
        print(result_list)
        return_file.close()
        # print("========evaluat %s attack %s done.========" % (attacker, recommender))

        return result_list

    def detect(self, detect):
        if not detect:
            return []
        attacker = self.__class__.__name__
        result_list = []

        cur_args_dict = {
            'exe_model_lib': 'evaluator',
            'exe_model_class': 'Attack_Effect_Evaluator',
            'data_path_clean': './data/%s/%s_train.dat' % (self.data_set, self.data_set),
            'data_path_attacked': './results/data_attacked/%s/%s_%s_%d.data' % (
                self.data_set, self.data_set, attacker, self.target_id),
        }
        #
        '''evalutors = ['Profile_Distance_Evaluator', 'FAP_Detector']
        for evalutor in evalutors:
            cur_args_dict.update({'exe_model_class': evalutor, })
            args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            return_file = os.popen('%s ./execute_model.py %s' % (PythonCommand, args_str))
            return_str = return_file.read()
            return_str = return_str[return_str.find('result begin') + 13:return_str.find('result end') - 2]
            result_list += [return_str]'''

        return result_list

    def attack(self, victim):
        attacker, recommender = self.__class__.__name__, victim
        args_dict = {
            'exe_model_lib': 'recommender',
            'exe_model_class': recommender,
            #
            'data_set': self.data_set,
            'train_path': './results/data_attacked/%s/%s_%s_%d.data' \
                          % (self.data_set, self.data_set, self.__class__.__name__, self.target_id),
            'test_path': './data/%s/%s_test.dat' % (self.data_set, self.data_set),
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            # 'attacker': attacker,
            #
            'model_path': './results/model_saved/%s/%s_%s_%s_%d' % (
                self.data_set, self.data_set, recommender, attacker, self.target_id),
            'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s_%s' % (
                self.data_set, self.data_set, recommender, attacker),
        }

        args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in args_dict.items()])
        target_file = "%s_%d.npy" % (args_dict['target_prediction_path_prefix'], self.target_id)
        if os.path.exists(target_file):
            os.remove(target_file)
        print("注入 攻击数据 训练模型")
        print(args_str)
        return_file = os.popen('%s ./victim/normal/main.py %s' % (PythonCommand, args_str))  # popen
        # time.sleep(60 * 3)
        return_str = return_file.read()
        return

    def execute(self):
        raise NotImplemented

    def save(self, path):
        raise NotImplemented

    def restore(self, path):
        raise NotImplemented

    def generate_fakeMatrix(self):
        raise NotImplemented

    def generate_injectedFile(self, fake_array=None):
        if fake_array is None:
            fake_array = self.generate_fakeMatrix()

        # injected_path = './results/data_attacked/ml100k/ml100k_attack_62.data'
        injected_path = './results/data_attacked/%s/%s_%s_%d.data' \
                        % (self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        if os.path.exists(injected_path):
            # print('clear data in %s' % self.injected_path)
            os.remove(injected_path)
        # shutil.copyfile(self.path_train, injected_path)
        # './data/%s/%s_50user_train.csv'
        path_train_20 = './data/ml1m/ml1m_train.csv'
        shutil.copyfile(path_train_20, injected_path)
        #
        uids = np.where(fake_array > 0)[0] + self.n_users
        iids = np.where(fake_array > 0)[1]
        values = fake_array[fake_array > 0]
        #
        data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [uids, iids, values]], 1)
        F_tuple_encode = lambda x: ','.join(map(str, [int(x[0]), int(x[1]), x[2]]))
        data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
        with open(injected_path, 'a+') as fout:
            fout.write(data_to_write)

        print('Inject successfully')
        return

    def visualize(self, results):
        import matplotlib.pyplot as plt
        fig, ax_list = plt.subplots(1, len(results), figsize=(4 * len(results), 4))

        key = sorted(list(results.keys()))
        for idx, ax in enumerate(ax_list):
            if len(results[key[idx]]) == 0:
                continue
            ax.plot(results[key[idx]])
            ax.set_xlabel("iteration")
            ax.set_title(key[idx])
        # plt.show()
        fig_path = "./results/performance/figs/%s/%s_%d.png" \
                   % (self.data_set, self.__class__.__name__, self.target_id)
        plt.savefig(fig_path)


class HeuristicAttacker(Attacker):
    def __init__(self):
        super(HeuristicAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        return parser

    def prepare_data(self):
        super(HeuristicAttacker, self).prepare_data()

    def generate_fakeMatrix(self):
        raise NotImplemented

    def execute(self):
        print('开始加载攻击模型')
        self.prepare_data()
        print('数据准备完成')
        res = self.test(victim='MF', detect=False)
        print("最终攻击结果")
        print(res)
        return

    def build_network(self):
        return

    def train(self):
        return

    def save(self, path):
        return

    def restore(self, path):
        return


class RandomAttacker(HeuristicAttacker):
    def __init__(self):
        super(RandomAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        # return parser.parse_args()
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(RandomAttacker, self).prepare_data()
        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()
        pass

    def generate_fakeMatrix(self):
        print("!!!!!!!!!!!!!!!")
        target_id_set = self.target_id_item
        # target_id_set =[1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        rate = int(self.attack_num/len(target_id_set))
        # padding target score
        print('every target item\'s num %s'%rate)
        for i in range(len(target_id_set)):
            fake_profiles[i * rate:(i + 1) * rate, target_id_set[i]] = 5
        # fake_profiles[:, self.target_id] = 5
        # padding fillers score
        print(self.n_items)
        filler_pool = list(set(range(self.n_items)) - set(target_id_set))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class AverageAttacker(HeuristicAttacker):
    def __init__(self):
        super(AverageAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(AverageAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()

        self.item_mean_dict = self.train_data_df.groupby('item_id').rating.mean().to_dict()

        self.item_std_dict = self.train_data_df.groupby('item_id').rating.std().fillna(self.global_std).to_dict()
        pass

    def generate_fakeMatrix(self):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        target_id_set = self.target_id_item
        rate = int(self.attack_num / len(target_id_set))
        # padding target score
        print('every target item\'s num %s' % rate)
        for i in range(len(target_id_set)):
            fake_profiles[i * rate:(i + 1) * rate, target_id_set[i]] = 5

        # fake_profiles[:, self.target_id] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - set(target_id_set))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        # sampled_values = np.random.normal(loc=0, scale=1,
        #                                   size=(self.attack_num * self.filler_num))
        sampled_values = [
            np.random.normal(loc=self.item_mean_dict.get(iid, self.global_mean),
                             scale=self.item_std_dict.get(iid, self.global_std))
            for iid in sampled_cols
        ]
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class BandwagonAttacker(HeuristicAttacker):
    def __init__(self):
        super(BandwagonAttacker, self).__init__()
        self.selected_ids = []
        # if args give the selected_ids then we will spilt it
        

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        parser.add_argument('--selected_ids', type=str, default='')  # , required=True)

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(BandwagonAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()

        if len(self.selected_ids) == 0:
            sorted_item_pop_df = self.train_data_df.groupby('item_id').agg('count').sort_values('user_id').index[::-1]
            self.selected_ids = sorted_item_pop_df[:11].to_list()
            
        pass

    def generate_fakeMatrix(self):
        print("Bandwagon the most popular items")
        print(self.selected_ids)

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        #fake_profiles[:, self.target_id] = 5
        target_id_set = self.target_id_item
        rate = int(self.attack_num / len(target_id_set))
        # padding target score
        print('every target item\'s num %s' % rate)
        for i in range(len(target_id_set)):
            fake_profiles[i * rate:(i + 1) * rate, target_id_set[i]] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - set(target_id_set) - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class SegmentAttacker(HeuristicAttacker):
    def __init__(self):
        super(SegmentAttacker, self).__init__()
        # self.selected_ids = [898, 1668, 2569, 10, 910, 16, 2066]
        # self.selected_ids = [898, 355, 1673, 911, 1264, 538, 919, 1720, 1306, 2395, 1246]
        self.selected_ids = [1153, 2201, 1572, 836, 523, 849, 1171, 344, 857, 1213, 1535]
        

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        parser.add_argument('--selected_ids', type=str, default='')  # , required=True)

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(SegmentAttacker, self).prepare_data()
        # self.global_mean = self.train_data_df.rating.mean()
        # self.global_std = self.train_data_df.rating.std()
        pass

    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        # fake_profiles[:, self.target_id] = 5
        target_id_set = self.target_id_item
        rate = int(self.attack_num / len(target_id_set))
        # padding target score
        print('every target item\'s num %s' % rate)
        for i in range(len(target_id_set)):
            fake_profiles[i * rate:(i + 1) * rate, target_id_set[i]] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id} - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.ones_like(sampled_rows)
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles
