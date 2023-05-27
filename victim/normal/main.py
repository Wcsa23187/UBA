
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
    model_lib_str = 'normal'
    model_lib = import_module(model_lib_str)
    print(getattr(model_lib, model_name))
    model = getattr(model_lib, model_name)()
    print("model %s" % (model))
    # 执行对应的  model
    model.execute()
    print('success.')


# get the args from the last file
print("----")
print("the args : %s" % (sys.argv))
print("----")
model_name = sys.argv[sys.argv.index('--exe_model_class') + 1]
execute_model(model_name)
