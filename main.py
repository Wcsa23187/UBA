# -*- coding: utf-8 -*-
# @Time       : 2020/12/27 19:57
# @Author     : chensi
# @File       : run.py
# @Software   : PyCharm
# @Desciption : None


import argparse, os, sys

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PythonCommand = 'python'  # 'D:\Anaconda3\envs\py38_tf2\python' if os.path.exists('D:\Anaconda3') else 'python'


class Run:
    def __init__(self):
        self.args = self.parse_args()
        self.args.attacker_list = self.args.attacker_list.split(',')
        self.args.recommender_list = self.args.recommender_list.split(',')

    def execute(self):

        # self.step_1_Rec()

        self.step_2_Attack()

        self.step_3_inject()

        self.step_4_evaluate()

        return

    def parse_args(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--way', type=int, default=1234)
        parser.add_argument('--alpha', type=float, default=1234)
        parser.add_argument('--oneitem', type=int, default=1234)
        parser.add_argument('--allseed', type=int, default=1234)
        parser.add_argument('--data_set', type=str, default='ml100k')  # ml100k,filmTrust,automotive
        parser.add_argument('--attack_num', type=int, default=300)
        parser.add_argument('--filler_num', type=int, default=36)
        parser.add_argument('--cuda_id', type=int, default=0)
        parser.add_argument('--use_cuda', type=int, default=0)
        parser.add_argument('--batch_size_S', type=int, default=32)
        parser.add_argument('--batch_size_D', type=int, default=64)
        parser.add_argument("--surrogate", type=str, default="WMF")

        # ml100k:62,1077,785,1419,1257
        # filmTrust:5,395,181,565,254
        # automotive:119,422,594,884,1593

        # parser.add_argument('--target_ids', nargs='+', type=float)
        parser.add_argument('--target_ids', type=int, default='62')
        # AUSH,AUSHplus,RecsysAttacker,DCGAN,WGAN,SegmentAttacker,BandwagonAttacker,AverageAttacker,RandomAttacker
        parser.add_argument('--attacker_list', type=str, default='SegmentAttacker')
        # SVD,NMF,SlopeOne,IAutoRec,UAutoRec,NeuMF
        # parser.add_argument('--recommender_list', type=str, default='SVD,NMF')
        parser.add_argument('--recommender_list', type=str, default='MF')
        #################  load the args of the  MF/NCF ###############

        # parser.add_argument("--lr_NCF", type=float, default=0.001, help="learning rate")
        # parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
        # parser.add_argument("--batch_size_NCF", type=int, default=4096, help="batch size for training")
        # parser.add_argument("--epochs_NCF", type=int, default=2, help="training epoches")
        # parser.add_argument("--top_k", default='[10, 20, 50, 100]', help="compute metrics@top_k")
        parser.add_argument("--factor_num", type=int, default=64, help="predictive factors numbers in the model")
        # parser.add_argument("--num_layers", type=int, default=64, help="number of layers in MLP model")
        # parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
        # parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
        parser.add_argument("--out", default=False, help="save model or not")
        parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
        parser.add_argument("--data_path", type=str,
                            default="/storage/shjing/recommendation/causal_discovery/data/ml-1m/random_split",
                            help="main path for dataset")
        parser.add_argument("--dataset", type=str, default='ml-1m', help="dataset")
        parser.add_argument("--data_type", type=str, default="time", help="time_split or random_split")
        parser.add_argument("--log_name", type=str, default='log', help="log_name")
        parser.add_argument("--model_path_NCF", type=str, default=r"F:\RSlib\latest\debug\Attack\victim\NCF\models\ ",
                            help="main path for model")
        parser.add_argument("--model", type=str, default="NeuMF-end", help="model name")
        parser.add_argument('--GMF_model_path', type=str, default=None, help='path to GMF model')
        parser.add_argument('--MLP_model_path', type=str, default=None, help='path to MLP model')
        #################  load the args of the  lightGCN ###############
        '''
        parser.add_argument('--light_topks', nargs='?', default="[10, 20, 50, 100]",
                            help="@k test list")
        parser.add_argument('--bpr_batch', type=int, default=2048,
                            help="the batch size for bpr loss training procedure")
        parser.add_argument('--recdim', type=int, default=64,
                            help="the embedding size of lightGCN")
        parser.add_argument('--layer', type=int, default=3,
                            help="the layer num of lightGCN")
        parser.add_argument('--light_lr', type=float, default=0.001,
                            help="the learning rate")
        parser.add_argument('--decay', type=float, default=1e-4,
                            help="the weight decay for l2 normalizaton")
        parser.add_argument('--light_dropout', type=float, default=0,
                            help="using the dropout or not")
        parser.add_argument('--keepprob', type=float, default=0.6,
                            help="the batch size for bpr loss training procedure")
        parser.add_argument('--a_fold', type=int, default=100,
                            help="the fold num used to split large adj matrix, like gowalla")
        
        parser.add_argument('--testbatch', type=int, default=400,
                            help="the batch size of users for testing")
        '''

        parser.add_argument('--light_data_path', type=str,
                            default='/storage/shjing/recommendation/causal_discovery/data/amazon_book/data_1202/split_time',
                            help='the path to dataset')
        parser.add_argument('--light_dataset', type=str, default='gowalla',
                            help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
        parser.add_argument('--light_data_type', type=str, default='time',
                            help='time or random')
        parser.add_argument('--light_path', type=str, default="./checkpoints",
                            help="path to save weights")


        parser.add_argument('--tensorboard', type=int, default=0,
                            help="enable tensorboard")
        parser.add_argument('--comment', type=str, default="lgn")
        parser.add_argument('--load', type=int, default=0)
        # parser.add_argument('--light_epochs', type=int, default=50)
        # parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
        # parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
        # parser.add_argument('--seed', type=int, default=2020, help='random seed')
        parser.add_argument('--light_model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

        parser.add_argument('--light_log_name', type=str, default='log', help='log name')
        parser.add_argument('--light_gpu', type=str, default='0')

        print("The model get the args from the shell:")
        print(parser.parse_args())
        print('\n')
        return parser.parse_args()

    def step_1_Rec(self):
        print('-------step_1-------')
        args = self.args
        """
        data_set/target_ids/train_path/test_path/model_path/target_prediction_path_prefix
        """
        args_dict = {
            'exe_model_lib': 'recommender',
            'main_path': './data/%s/' % args.data_set
            # 'test_path': './data/%s/%s_test.dat' % (args.data_set, args.data_set),
        }
        args_dict.update(vars(args))

        # here we can train the victim model by circulation

        for recommender in args.recommender_list:

            cur_args_dict = {
                'exe_model_class': recommender,
                'model_path': './results/model_saved/%s/%s_%s' % (args.data_set, args.data_set, recommender),
                'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s' % (
                    args.data_set, args.data_set, recommender),
            }
            cur_args_dict.update(args_dict)
            # print(cur_args_dict)
            path = cur_args_dict['model_path']
            print("Vmodel args will be saved at %s" % (cur_args_dict['model_path']))
            if not os.path.exists(path):
                os.makedirs(path)
            path = cur_args_dict['target_prediction_path_prefix']
            if not os.path.exists(path):
                os.makedirs(path)
            print("Vmodel result will be save at %s" % (cur_args_dict['target_prediction_path_prefix']))
            print('\n')
            args_str = ' '.join(
                ["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            print("start to train Vmodel with the order :")


            if args.model == 'lgn':
                print('Train the model with lightGCN ')
                print('%s ./victim/lightGCN/main.py %s' % (PythonCommand, args_str))
                print("----------run in the step_1---------------")
                print('\n')
                print(os.system('%s ./victim/lightGCN/main.py %s' % (PythonCommand, args_str)))

            else:
                print('Train the model with the MF or NCF')
                print('%s ./victim/NCF/main.py %s' % (PythonCommand, args_str))
                print("----------run in the step_1---------------")
                print('\n')
                print(os.system('%s ./victim/NCF/main.py %s' % (PythonCommand, args_str)))

            '''print('%s ./normal//main.py %s' % (PythonCommand, args_str))
            print("----------run in the step_1---------------")
            print(os.system('%s ./victim/normal/main.py %s' % (PythonCommand, args_str)))'''

    def step_2_Attack(self):

        print("---------step_2----------")
        args = self.args
        args_dict = {
            'exe_model_lib': 'attacker',
            # 'filler_num': 4,
            # 'epoch': 50
        }
        args_dict.update(vars(args))

        for attacker in args.attacker_list:
            cur_args_dict = {
                'exe_model_class': attacker,
                'target_id': args.target_ids,
                'injected_path_before': './results/data_attacked/%s' % (
                    args.data_set),
                'injected_path': './results/data_attacked/%s/%s_%s_%s.data' % (
                    args.data_set, args.data_set, attacker, args.target_ids)

            }
            cur_args_dict.update(args_dict)
            path = cur_args_dict['injected_path_before']
            print("引用参数尝试 %s" % (cur_args_dict['injected_path_before']))
            if not os.path.exists(path):
                os.makedirs(path)

            path = cur_args_dict['injected_path']
            print("引用参数尝试 %s" % (cur_args_dict['injected_path']))

            f = open(path, 'w')
            f.close()

            args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            print(args_str)
            '''if args.attacker_list[0] == 'AUSHplus' or 'AIA':
                print('load the GAN model')
                print(os.system('%s ./Attacker/AUSH++/main.py %s' % (PythonCommand, args_str)))
            else:
                print('load the normal attack')
                print(os.system('%s ./Attacker/normal_attack/main.py %s' % (PythonCommand, args_str)))
                '''

            print('load the GAN model')
            print(os.system('%s ./Attacker/AUSH/main.py %s' % (PythonCommand, args_str)))

            # print('load the normal attack')
            # print(os.system('%s ./Attacker/normal_attack/main.py %s' % (PythonCommand, args_str)))

            # print(os.system('%s ./Attacker/AUSH++/main.py %s' % (PythonCommand, args_str)))
            # break

    def step_3_inject(self):
        print("----------step_3----------")
        args = self.args
        for recommender in args.recommender_list:
            attacker, recommender = args.attacker_list[0], recommender
            print(attacker)
            args_dict = {
                'attack': 1,
                'main_path': './data/%s/' % args.data_set,
                'exe_model_lib': 'recommender',
                'exe_model_class': recommender,
                #
                'data_set': args.data_set,
                'train_path': './results/data_attacked/%s/%s_%s_%s.data' \
                              % (args.data_set, args.data_set, attacker, args.target_ids),
                'test_path': './data/%s/%s_test.dat' % (args.data_set, args.data_set),
                #
                'target_ids': args.target_ids,
                'recommender': recommender,
                # 'attacker': attacker,

                'model_path': './results/model_saved/%s/%s_%s_%s_%s' % (
                    args.data_set, args.data_set, recommender, attacker, args.target_ids),
                'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s_%s' % (
                    args.data_set, args.data_set, recommender, attacker),
            }
            args_dict.update(vars(args))
            args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in args_dict.items()])
            target_file = "%s_%s.npy" % (args_dict['target_prediction_path_prefix'], args.target_ids)
            if os.path.exists(target_file):
                os.remove(target_file)

            print(args_str)

            print(os.system('%s ./victim/lightGCN/main.py %s' % (PythonCommand, args_str)))

            # print(os.system('%s ./victim/NCF/main.py %s' % (PythonCommand, args_str)))

            # print(os.system('%s ./victim/NCF/main.py %s' % (PythonCommand, args_str)))
            # print(os.system('%s ./victim/normal/main.py %s' % (PythonCommand, args_str)))

            # return_file = os.popen('%s ./victim/NCF/main.py %s' % (PythonCommand, args_str))

            '''return_file = os.popen('%s ./victim/lightGCN/main.py %s' % (PythonCommand, args_str))
            # popen
            # time.sleep(60 * 3)
            return_str = return_file.read()'''

    def step_4_evaluate(self):
        print("-------------step_4-----------")
        args = self.args
        for recommender in args.recommender_list:
            attacker, recommender = args.attacker_list[0], recommender
            print(attacker)
            args_dict = {
                'data_set': args.data_set,
                'test_path': './data/%s/%s_test.dat' % (args.data_set, args.data_set),

                'target_ids': args.target_ids,
                'recommender': recommender,
                'attacker': attacker,
                'oneitem':args.oneitem

            }

            path_res_before_attack = './results/performance/mid_results/%s/%s_%s_%s.npy' % (
                args.data_set, args.data_set, recommender, args.target_ids)

            result_list = []
            cur_args_dict = {
                'exe_model_lib': 'evaluator',
                'exe_model_class': 'Attack_Effect_Evaluator',
                'data_path_clean': './results/performance/mid_results/%s/%s_%s_%s.npy' % (
                    args.data_set, args.data_set, recommender, args.target_ids),
                'data_path_attacked': './results/performance/mid_results/%s/%s_%s_%s_%s.npy' % (
                    args.data_set, args.data_set, recommender, attacker, args.target_ids),
            }
            cur_args_dict.update(args_dict)
            args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])

            print(args_str)
            #
            print(os.system('%s ./utils/main.py %s' % (PythonCommand, args_str)))
            # return_file = os.popen('%s ./utils/main.py %s' % (PythonCommand, args_str))
            # time.sleep(5)
            # return_str = return_file.read()

            # return_str = return_str[return_str.find('result begin') + 13:return_str.find('result end') - 2]
            # result_list += [return_str]

            # print(result_list)
            # return_file.close()
            # print("========evaluat %s attack %s done.========" % (attacker, recommender))
            # res_attack = '\t'.join(result_list)
            # res = '\t'.join([res_attack])
            # print(res)

        # return result_list


model = Run()
model.execute()
