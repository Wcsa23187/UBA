train_file = r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\training_dict.npy'
valid_file = r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\training_dict.npy'
test_file = r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\training_dict.npy'
# self.path = path


#导入所需的包
import numpy as np
import pandas as pd
#导入npy文件路径位置
test = np.load(r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\testing_dict.npy',allow_pickle=True)
test = test.item()
# print(test[6040])
path_train = r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\ml100k_train.dat'
path_test = r'F:\\RSlib\\code\\Recommender_baselines-main2\\Recommender_baselines-main\\data\\ml100k_train.dat'

def load_file_as_dataFrame(path_train, path_test):
    # load data to pandas dataframe
    train_data = pd.read_csv(path_train, sep='\t', names=['user_id', 'item_id', 'rating'], engine='python')
    test_data = pd.read_csv(path_test, sep='\t', names=['user_id', 'item_id', 'rating'], engine='python')
    n_users = max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
    n_items = max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1
    # print("Number of users : %d , Number of items : %d. " % (n_users, n_items), flush=True)
    # print("Train size : %d , Test size : %d. " % (train_data.shape[0], test_data.shape[0]), flush=True)
    return train_data, test_data, n_users, n_items


train_data_df, test_data_df, n_users, n_items = load_file_as_dataFrame(path_train, path_test)
# cat the train_data and test_data
result = pd.concat([train_data_df, test_data_df])
# cut the whole data with 8 : 1 : 1 = train : test : validation
user_name = result['user_id'].unique()
train_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
vali_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
test_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
for user_id in user_name:
    mid = result[result['user_id'] == user_id]
    train_df_son = mid.iloc[0:int((mid.shape[0] * 8) / 10)]
    vali_df_son = mid.iloc[int((mid.shape[0] * 8) / 10) + 1:int((mid.shape[0] * 9) / 10)]
    test_df_son = mid.iloc[int((mid.shape[0] * 9) / 10) + 1:]
    '''vali_df_son = mid.iloc[0:int((mid.shape[0] * 1) / 10)]
    train_df_son = mid.iloc[int((mid.shape[0] * 1) / 10) + 1:int((mid.shape[0] * 9) / 10)]
    test_df_son = mid.iloc[int((mid.shape[0] * 9) / 10) + 1:]'''

    train_df = pd.concat([train_df_son, train_df])
    vali_df = pd.concat([vali_df_son, vali_df])
    test_df = pd.concat([test_df_son, test_df])
# 8 : 1 : 1 data = train_df : vali_df : test_df and i user the sample cut
# let train_df to be the array
train_data = []
for user_id in user_name:
    mid = train_df[train_df['user_id']==user_id]
    print(mid)










for index, row in train_df.iterrows():
    mid = []
    mid.append(row['user_id'])
    mid.append(row['item_id'])
    mid.append(row['rating'])
    train_data.append(mid)




vali_data = []
for index, row in vali_df.iterrows():
    mid = []
    mid.append(row['user_id'])
    mid.append(row['item_id'])
    mid.append(row['rating'])
    vali_data.append(mid)
test_data = []
for index, row in test_df.iterrows():
    mid = []
    mid.append(row['user_id'])
    mid.append(row['item_id'])
    mid.append(row['rating'])
    test_data.append(mid)

print(test_data)