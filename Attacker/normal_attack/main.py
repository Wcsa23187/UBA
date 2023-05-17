# -*- coding: utf-8 -*-
# @Time       : 2020/11/29 11:59
# @Author     : chensi
# @File       : execute_model.py
# @Software   : PyCharm
# @Desciption : None
import random
import numpy as np
import torch
'''
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''
from importlib import import_module
import sys

def execute_model(model_name):
    model_lib_str = 'normal_attack'
    model_lib = import_module(model_lib_str)
    model = getattr(model_lib, model_name)()
    model.execute()
    print('success.')

# 接收上一个文件传来的参数

model_name = sys.argv[sys.argv.index('--exe_model_class') + 1]

execute_model(model_name)
