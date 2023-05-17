
import os
from os.path import join
import torch
from enum import Enum
# from parse import parse_args
import multiprocessing
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# sys.argv[sys.argv.index('--target_prediction_path_prefix') + 1]
# args = parse_args()

ROOT_PATH = "/home/wcs/attack1/Attack/victim/lightGCN/rootpath"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = 2048
config['latent_dim_rec'] = 128
config['lightGCN_n_layers'] = 3
config['dropout'] = 0
config['keep_prob'] = 0.6
config['A_n_fold'] = 100
# config['test_u_batch_size'] = args.testbatch
config['test_u_batch_size'] = 400
# print('test_u_batch_size %s' %(args.testbatch))
config['multicore'] = 0
config['lr'] = 0.001
config['decay'] = 0.0001
config['pretrain'] = 0
config['A_split'] = False
config['bigdata'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = 2022
dataset = str(sys.argv[sys.argv.index('--light_dataset') + 1])
model_name = str(sys.argv[sys.argv.index('--light_model') + 1])
# if dataset not in all_dataset:
#     raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")



TRAIN_epochs = 400
# TRAIN_epochs = 3
LOAD = int(sys.argv[sys.argv.index('--load') + 1])
PATH = str(sys.argv[sys.argv.index('--light_path') + 1])
# topks = eval(sys.argv[sys.argv.index('--light_topks') + 1])
topks = [10, 20, 50, 100]
tensorboard = int(sys.argv[sys.argv.index('--tensorboard') + 1])
comment = str(sys.argv[sys.argv.index('--comment') + 1])
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

