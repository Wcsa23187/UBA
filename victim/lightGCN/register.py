import world
import dataloader
import model
import utils
import sys
from pprint import pprint
# from parse import parse_args
print('1')
# args = parse_args()
data_path = str(sys.argv[sys.argv.index('--light_path') + 1])
dataset = dataloader.Loader(path=data_path)

print('===========config================')
print(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}