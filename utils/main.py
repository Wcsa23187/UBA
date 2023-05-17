
import random
import numpy as np
import torch

seed = 1234
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from importlib import import_module
import sys

def execute_model(model_name):
    model_lib_str = 'evaluator'

    model_lib = import_module(model_lib_str)

    model = getattr(model_lib, model_name)()

    print("model %s" % (model))

    model.execute()
    print('success.')

# 接收上一个文件传来的参数
print("进入 evaluate 后传入的参数")
print(sys.argv)
print('the system args get by execute_model:%s ' % (sys.argv))
# model_lib = sys.argv[sys.argv.index('--exe_model_lib') + 1]
model_name = sys.argv[sys.argv.index('--exe_model_class') + 1]
print(model_name)
execute_model(model_name)
