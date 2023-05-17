# -*- coding: utf-8 -*-
# @Time       : 2020/11/27 15:34
# @Author     : chensi
# @File       : data_loader.py
# @Software   : PyCharm
# @Desciption : None

import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csr_matrix
import pandas as pd
import random
import numpy as np
import torch

# tf = None
# try:
#     import tensorflow.compat.v1 as tf
#
#     tf.disable_v2_behavior()
# except:
#     import tensorflow as tf


import sys
seed = int(sys.argv[sys.argv.index('--allseed') + 1])

print(seed)
random.seed(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DataLoader(object):

    def __init__(self, path_train, path_test, header=None, sep='\t', threshold=4, verbose=False):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else [
            'user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose

        # load file as dataFrame
        # self.train_data, self.test_data, self.n_users, self.n_items = self.load_file_as_dataFrame()
        # dataframe to matrix
        # self.train_matrix, self.train_matrix_implicit = self.dataFrame_to_matrix(self.train_data)
        # self.test_matrix, self.test_matrix_implicit = self.dataFrame_to_matrix(self.test_data)

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe
        if self.verbose:
            print("\nload data from %s ..." % self.path_train, flush=True)
        

        train_data = pd.read_csv(self.path_train, engine='python')
        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]
        print(train_data.head())

        if self.verbose:
            print("load data from %s ..." % self.path_test, flush=True)
        test_data = pd.read_csv(self.path_test, engine='python').loc[:,
                                                                     ['user_id', 'item_id', 'rating']]
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]

        # data statics

        n_users = max(max(test_data.user_id.unique()),
                      max(train_data.user_id.unique())) + 1
        n_items = max(max(test_data.item_id.unique()),
                      max(train_data.item_id.unique())) + 1

        if self.verbose:
            print("Number of users : %d , Number of items : %d. " %
                  (n_users, n_items), flush=True)
            print("Train size : %d , Test size : %d. " %
                  (train_data.shape[0], test_data.shape[0]), flush=True)

        return train_data, test_data, n_users, n_items

    def dataFrame_to_matrix(self, data_frame, n_users, n_items):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items))
        matrix_implicit = csr_matrix(
            (implicit_rating, (row, col)), shape=(n_users, n_items))
        return matrix, matrix_implicit


EPSILON = 1e-12
_fixed_target_items = {
    "head": np.asarray([259, 2272, 3010, 6737, 7690]),
    "tail": np.asarray([5611, 9213, 10359, 10395, 12308]),
    "upper_torso": np.asarray([1181, 1200, 2725, 4228, 6688]),
    "lower_torso": np.asarray([3227, 5810, 7402, 9272, 10551])
}


def sample_target_items(train_data, n_samples, popularity, use_fix=False):
    """Sample target items with certain popularity."""
    if popularity not in ["head", "upper_torso", "lower_torso", "tail"]:
        raise ValueError("Unknown popularity type {}.".format(popularity))

    n_items = train_data.shape[1]  # 14007
    all_items = np.arange(n_items)  # [0, 1, 2, ... , 14006]
    item_clicks = train_data.toarray().sum(0)

    valid_items = []
    if use_fix:
        valid_items = _fixed_target_items[popularity]
    else:
        bound_head = np.percentile(item_clicks, 95)
        bound_torso = np.percentile(item_clicks, 75)
        bound_tail = np.percentile(item_clicks, 50)
        if popularity == "head":
            valid_items = all_items[item_clicks > bound_head]
        elif popularity == "tail":
            valid_items = all_items[item_clicks < bound_tail]
        elif popularity == "upper_torso":
            valid_items = all_items[(item_clicks > bound_torso) & (
                item_clicks < bound_head)]
        elif popularity == "lower_torso":
            valid_items = all_items[(item_clicks > bound_tail) & (
                item_clicks < bound_torso)]

    if len(valid_items) < n_samples:
        raise ValueError("Cannot sample enough items that meet criteria.")

    np.random.shuffle(valid_items)
    sampled_items = valid_items[:n_samples]
    sampled_items.sort()
    print("Sampled target items: {}".format(sampled_items.tolist()))

    return sampled_items


def set_seed(seed, cuda=False):
    """Set seed globally."""
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):
    """Mini-batch generator for pytorch tensor."""
    batch_size = kwargs.get('batch_size', 128)  # 2048

    if len(tensors) == 1:  # �?
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):  # len(tensor) = 14007
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """Shuffle arrays."""
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def sparse2tensor(sparse_data):
    """Convert sparse csr matrix to pytorch tensor."""
    return torch.FloatTensor(sparse_data.toarray())


def tensor2sparse(tensor):
    """Convert pytorch tensor to sparse csr matrix."""
    return sparse.csr_matrix(tensor.detach().cpu().numpy())


def stack_csrdata(data1, data2):
    """Stack two sparse csr matrix."""
    return sparse.vstack((data1, data2), format="csr")


def save_fake_data(fake_data, path):
    """Save fake data to file."""
    file_path = "%s.npz" % path
    print("Saving fake data to {}".format(file_path))
    sparse.save_npz(file_path, fake_data)
    return file_path


def load_fake_data(file_path):
    """Load fake data from file."""
    fake_data = sparse.load_npz(file_path)
    print("Loaded fake data from {}".format(file_path))
    return fake_data


def save_checkpoint(model, optimizer, path, epoch=-1):
    """Save model checkpoint and optimizer state to file."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    file_path = "%s.pt" % path
    print("Saving checkpoint to {}".format(file_path))
    torch.save(state, file_path)


def load_checkpoint(path):
    """Load model checkpoint and optimizer state from file."""
    file_path = "%s.pt" % path
    state = torch.load(file_path, map_location=torch.device('cpu'))
    print("Loaded checkpoint from {} (epoch {})".format(
        file_path, state["epoch"]))
    return state["epoch"], state["state_dict"], state["optimizer"]


__all__ = ["mse_loss", "mult_ce_loss", "binary_ce_loss", "kld_loss",
           "sampled_bce_loss", "sampled_cml_loss"]

"""Model training losses."""
bce_loss = torch.nn.BCELoss(reduction='none')


def mse_loss(data, logits, weight):
    """Mean square error loss."""
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits) ** 2
    return res.sum(1)


def mult_ce_loss(data, logits):
    """Multi-class cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    # Avoid divide by zeros.
    res = instance_loss / (instance_data + EPSILON)
    return res


def binary_ce_loss(data, logits):
    """Binary-class cross-entropy loss."""
    return bce_loss(torch.sigmoid(logits), data).mean(1)


def kld_loss(mu, log_var):
    """KL-divergence."""
    return -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def sampled_bce_loss(logits, n_negatives):
    """Binary-class cross-entropy loss with sampled negatives."""
    pos_logits, neg_logits = torch.split(logits, [1, n_negatives], 1)
    data = torch.cat([
        torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
    ], 1)
    return bce_loss(torch.sigmoid(logits), data).mean(1)


def sampled_cml_loss(distances, n_negatives, margin):
    """Hinge loss with sampled negatives."""
    # Distances here are the negative euclidean distances.
    pos_distances, neg_distances = torch.split(-distances, [1, n_negatives], 1)
    neg_distances = neg_distances.min(1).values.unsqueeze(-1)
    res = pos_distances - neg_distances + margin
    res[res < 0] = 0
    return res.sum(1)
