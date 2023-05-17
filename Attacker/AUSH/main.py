# -*- coding: utf-8 -*-
# @Time       : 2020/11/29 11:59
# @Author     : chensi
# @File       : execute_model.py
# @Software   : PyCharm
# @Desciption : None
import random
import numpy as np
'''
import torch
seed = 2022
print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''
from importlib import import_module
import sys
from aushplus import AUSHplus
from aushplus import AIA
from attacker import WGANAttacker
from aushplus import AUSH

def execute_model(model_name):
    '''print(model_name)
    model_lib_str = 'aushplus
    model_lib = import_module(model_lib_str)
    model = getattr(model_lib, model_name)()
    print(model)'''
    model = globals()[model_name]()
    # model = AIA()
    # model = AUSHplus()
    #model = WGANAttacker()
    # model = AUSH()
    model.execute()
    print('success.')


print(sys.argv)
model_name = sys.argv[sys.argv.index('--exe_model_class') + 1]
print(model_name)
execute_model(model_name)
