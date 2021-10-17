import torch.utils.data as data
from copy import deepcopy
import numpy as np
import torch
from collections import defaultdict
import random
def load_all(args):
    user_num,item_num = 0,0
    train_mat = {}
    train_support_mat,test_support_mat={},{}
    val_mat={}
    test_mat={}
    train_data,train_support_data,test_support_data = defaultdict(list),defaultdict(list),defaultdict(list)
    val_data,test_query_data = defaultdict(list),defaultdict(list)
    test_support, val_negative,test_negative = [],[],[]
    val_negative_dict,test_negative_dict = defaultdict(list),defaultdict(list)
    data_path = '../dataset/' + args.dataset + '/'
    with open(data_path+'./pretrain.rating') as f:
        for line in f:
            arr = line.split('\t')
            u,v = int (arr[0]),int(arr[1])
            user_num = max(u,user_num)
            item_num = max(v,item_num)
            train_data[u].append(v)
            train_mat[(u,v)] = 1.0
    val_rating = []
    with open(data_path+'./val_pre.rating') as f:
        for line in f:
            arr = line.rstrip().split('\t')
            u, v = int(arr[0]), int(arr[1])
            user_num = max(u,user_num)
            item_num = max(v,item_num)
            val_data[u].append(v)
            val_mat[(u, v)] = 1.0
            train_mat[(u,v)] = 1.0
            val_rating.append([u, v])

    for idx, (u, v) in enumerate(val_rating):
        val_negative_dict[u].append(idx)
        tmp_val_item_list = [v]
        # select negative samples for each user -- first strategy
        for _ in range(200):
            j = np.random.randint(0,item_num+1)
            while (u,j) in val_mat or (u,j) in train_mat:
                j = np.random.randint(0,item_num+1)
            tmp_val_item_list.append(j)
        val_negative.append(tmp_val_item_list)


    return user_num+1,item_num+1,train_data,  val_negative_dict, val_negative,train_mat,val_mat





class Test_data(data.Dataset):
    def __init__(self, test_support_data, item_num, test_support_mat, num_ng):
        super(Test_data, self).__init__()
        self.test_support_data = test_support_data
        self.item_num = item_num
        self.test_support_mat = test_support_mat
        self.num_ng = num_ng
    def ng_test_sample(self):
        test_support_data = deepcopy(self.test_support_data)
        test_support_fill = defaultdict(dict)
        idx = 0
        user_list, item_list, label_list = [], [], []
        for (u, tmp_item_list) in test_support_data.items():
            for v in tmp_item_list:
                user_list.append(u)
                item_list.append(v)
                label_list.append(1.0)
                for _ in range(self.num_ng):
                    j = np.random.randint(0,self.item_num)
                    while  (u, j) in self.test_support_mat:
                        j = np.random.randint(0,self.item_num)
                    user_list.append(u)
                    item_list.append(j)
                    label_list.append(0.0)

        self.user_list = user_list
        self.item_list = item_list
        self.label_list = label_list
        self.idx = len(self.user_list)
    def get_batch(self,idxs):
        user_batch = []
        item_batch = []
        label_batch = []
        for idx in idxs:
            user_batch.append(self.user_list[idx])
            item_batch.append(self.item_list[idx])
            label_batch.append(self.label_list[idx])
        return torch.tensor(user_batch).cuda(),torch.tensor(item_batch).cuda(),torch.tensor(label_batch).cuda()
