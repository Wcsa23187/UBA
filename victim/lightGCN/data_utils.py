import numpy as np 
import torch.utils.data as data


def load_all(train_path, valid_path, test_path):
	""" We load all the three file here to save time in each epoch. """
	train_dict = np.load(train_path, allow_pickle=True).item()
	valid_dict = np.load(valid_path, allow_pickle=True).item()
	test_dict = np.load(test_path, allow_pickle=True).item()

	user_num, item_num = 0, 0
	user_num = max(user_num, max(train_dict.keys()))
	user_num = max(user_num, max(valid_dict.keys()))
	user_num = max(user_num, max(test_dict.keys()))
	
	train_data, valid_gt, test_gt = [], [], []
	for user, items in train_dict.items():
		if len(items) != 0:
			item_num = max(item_num, max(items))
			for item in items:
				train_data.append([int(user), int(item)])
	for user, items in valid_dict.items():
		if len(items) != 0:
			item_num = max(item_num, max(items))
			for item in items:
				valid_gt.append([int(user), int(item)])
	for user, items in test_dict.items():
		if len(items) != 0:
			item_num = max(item_num, max(items))
			for item in items:
				test_gt.append([int(user), int(item)])
		print("####")

		return user_num+1, item_num+1, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt


class NCFData(data.Dataset):
	def __init__(self, features, 
				num_item,  is_training=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		print("--init--")
		self.features = features
		self.num_item = num_item
		self.is_training = is_training





	def __len__(self):
		# print("--len--")
		return len(self.features)

	def __getitem__(self, idx):
		features = self.features
		user = features[idx][0]
		item = features[idx][1]
		label = features[idx][2]
		return user, item ,label


class vt_data(data.Dataset):
	def __init__(self, features,
				num_item, vt_dict, num_ng=0, is_training=None):
		super(vt_data, self).__init__()
		self.features = features
		self.num_item = num_item
		self.vt_dict = vt_dict
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [1 for _ in range(len(features))]

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features
		labels = self.labels
		item = features[idx]
		label = labels[idx]