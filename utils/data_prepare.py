import torch.utils.data as data
from copy import deepcopy
import numpy as np
from collections import defaultdict
import torch
import random
def load_all(args):
    file_name = './dataset/'+args.dataset+'/'
    user_num, item_num = 0, 0
    if args.mod=='train' or args.mod=='ex':
        train_mat,train_support_mat,val_mat = {},{},{}
        train_data, train_support_data = defaultdict(list), defaultdict(list)
        val_negative = []
        val_negative_dict = defaultdict(list)
        user_num,item_num,sup_max,query_max = 0,0,0,0
        with open(file_name+'train.rating') as f:
            for line in f:
                arr = line.split('\t')
                u, v = int(arr[0]), int(arr[1])
                user_num = max(u, user_num)
                item_num = max(v, item_num)
                train_data[u].append(v)
                train_mat[(u, v)] = 1.0
        val_rating = []
        with open(file_name+'val_train.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                user_num = max(u, user_num)
                item_num = max(v, item_num)
                val_mat[(u, v)] = 1.0
                val_rating.append([u, v])

        for idx, (u, v) in enumerate(val_rating):
            val_negative_dict[u].append(idx)
            tmp_val_item_list = [v]
            val_negative.append(tmp_val_item_list)
        avg_sup,avg_que = [],[]

        for u in train_data:
            tmp_item_list = train_data[u]
            K = len(tmp_item_list) // 2
   

            if K > 0:
                if K>200:
                    train_data[u] = []
                    continue
                sup_max = max(K, sup_max)
                query_max = max(query_max, len(tmp_item_list[K:]))
                for v in tmp_item_list[:K]:
                    train_support_mat[(u, v)] = 1

        return user_num+1,item_num+1,train_data, train_support_mat, val_negative_dict, val_negative,  train_mat, sup_max,query_max

    elif args.mod=='test':
        sup_max=0
        if args.dataset=='amazon_big':
            item_num = 57790
        elif args.dataset=='amazon_small':
            item_num = 9448
        else:
            item_num= 3951
        test_support_mat = {}
        test_mat = {}
        test_support_data = defaultdict(list)
        test_query_data = defaultdict(list)
        test_support,  test_negative = [], []
        test_negative_dict = defaultdict(list)
        test_rating_data = defaultdict(list)
        with open(file_name+'test.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                test_rating_data[u].append(v)
                test_mat[(u, v)] = 1.0




        cnt = 0
        for u in test_rating_data:
            test_rating_item_list = test_rating_data[u]
            if (len(test_rating_item_list) > 1):
                # this is for generating the test support data and test query data
                K = len(test_rating_item_list) // 2
                sup_max = max(K, sup_max)
                test_support_data[u] = deepcopy(test_rating_item_list[:K])
                test_query_data[u] = deepcopy(test_rating_item_list[K:])
                # this if for generating the test query positive & negative data
                for v in test_rating_item_list[:K]:
                    test_support_mat[(u, v)] = 1
                test_negative_dict[u].append(cnt)
                test_negative_dict[u].append(len(test_query_data[u]))
                tmp_query_item_list = deepcopy(test_query_data[u])
                test_negative.append(tmp_query_item_list)
                cnt += 1
        # this is for train rating loss calculation

        return item_num,test_support_data,  test_negative_dict, test_negative, test_mat,sup_max




class Train_data(data.Dataset):
    def __init__(self, train_data, item_num, train_mat,sup_max,query_max,args):
        super(Train_data, self).__init__()
        self.train_data = train_data
        self.item_num = item_num
        self.train_mat = train_mat
        self.args = args
        self.num_ng = self.args.num_negative_support
        self.positive_size = sup_max
        self.sup_size = sup_max*(self.num_ng+1)
        self.query_size = query_max*(self.num_ng+1)
        self.idx = 0
    def __len__(self):
        return self.idx

    def ng_train_sample(self):
        train_data = deepcopy(self.train_data)
        support_fill, query_fill = defaultdict(dict), defaultdict(dict)
        idx = 0
        self.positive_dict = defaultdict(list)
        for (u, item_list) in train_data.items():

            K = len(item_list) // 2
            if K > 0:
                random.shuffle(item_list)
                support_item_list, support_label_list,positive_list,mdsk,positive_mdsk = [], [],[],[],[]
                positive_list = deepcopy(item_list[:K])
                len_pos = K
                for i in range(len(positive_list)):
                    positive_mdsk.append(1.0)
                while(len(positive_list)<self.positive_size):
                    positive_list.append(self.item_num)
                    positive_mdsk.append(0.0)
                for v in item_list[:K]:
                    support_item_list.append(v)
                    support_label_list.append(1.0)
                    mdsk.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.train_mat:
                            j = np.random.randint(0, self.item_num)
                        support_item_list.append(j)
                        support_label_list.append(0.0)
                        mdsk.append(1.0)
                while len(support_item_list) < self.sup_size:
                    support_item_list.append(self.item_num)
                    support_label_list.append(0.0)
                    mdsk.append(0.0)
                data_dict = {
                    'user': u,
                    'item_list': support_item_list,
                    'label_list': support_label_list,
                    'positive_list':positive_list,              #with padding
                    'positive_mdsk':positive_mdsk,              #making the loss related to padding zero (generate phase)
                    'mdsk':mdsk,                                #making the loss related to padding zero (loss phase)
                    'len_pos':len_pos                           #for average pooling with positive items
                }
                support_fill[idx] = data_dict
                query_item_list, query_label_list,query_mdsk = [], [],[]
                self.positive_dict[u] = deepcopy(item_list[:K])
                for v in item_list[K:]:
                    query_item_list.append(v)
                    query_label_list.append(1.0)
                    query_mdsk.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.train_mat:
                            j = np.random.randint(0, self.item_num)
                        query_item_list.append(j)
                        query_label_list.append(0.0)
                        query_mdsk.append(1.0)
                while len(query_item_list) < self.query_size:
                    query_item_list.append(self.item_num)
                    query_label_list.append(0)
                    query_mdsk.append(0.0)
                data_dict = {
                    'user': u,
                    'item_list': query_item_list,
                    'label_list': query_label_list,
                    'mdsk':query_mdsk                                #making the loss related to padding zero (loss phase)
                }
                query_fill[idx] = data_dict
                idx += 1
                self.idx = idx
            self.support_fill = support_fill
            self.query_fill = query_fill

    '''
    fill:{idx:{'item_lst':...,'label_lst':...,'user_lst':...}}
    '''
    def get_batch(self, idxs):
        len_pos,positive_itm,sup_itm,sup_label,query_itm,query_label,sup_mdsk,query_mdsk,positive_mdsk = [],[],[],[],[],[],[],[],[]
        for idx in idxs:
            len_pos.append(1.0/self.support_fill[idx]['len_pos'])
            positive_itm.append(self.support_fill[idx]['positive_list'])
            sup_itm.append(self.support_fill[idx]['item_list'])
            sup_label.append(self.support_fill[idx]['label_list'])
            sup_mdsk.append(self.support_fill[idx]['mdsk'])
            query_itm.append(self.query_fill[idx]['item_list'])
            query_label.append(self.query_fill[idx]['label_list'])
            query_mdsk.append(self.query_fill[idx]['mdsk'])
            positive_mdsk.append(self.support_fill[idx]['positive_mdsk'])
        return torch.tensor(positive_itm).cuda(),torch.tensor(sup_itm).cuda(),torch.tensor(sup_label).cuda(),torch.tensor(sup_mdsk).cuda(),torch.tensor(positive_mdsk).cuda(),\
               torch.tensor(query_itm).cuda(),torch.tensor(query_label).cuda(),torch.tensor(query_mdsk).cuda(),torch.tensor(len_pos).cuda()


class Test_data(data.Dataset):
    def __init__(self, test_support_data, item_num, test_mat,sup_max, args):
        super(Test_data, self).__init__()
        self.test_support_data = test_support_data
        self.item_num = item_num
        self.test_mat = test_mat
        self.args = args
        self.num_ng = args.num_negative_support
        self.sup_dict = defaultdict(int)
        self.positive_size = sup_max
        #record the size of test_support_set for each user
        self.sup_size = sup_max * (self.num_ng + 1)
    def ng_test_sample(self):
        support_data = deepcopy(self.test_support_data)
        support_fill, query_fill = defaultdict(dict), defaultdict(dict)
        idx = 0
        self.positive_dict = defaultdict(list)
        for (u, item_list) in support_data.items():

            if len(item_list) >0:
                self.sup_dict[u] = len(item_list)
                random.shuffle(item_list)
                support_item_list, support_label_list, positive_list, mdsk, positive_mdsk = [], [], [], [], []
                positive_list = deepcopy(item_list)
                len_pos = len(positive_list)
                for i in range(len(positive_list)):
                    positive_mdsk.append(1.0)
                while (len(positive_list) < self.positive_size):
                    positive_list.append(self.item_num)
                    positive_mdsk.append(0.0)
                for v in item_list:
                    support_item_list.append(v)
                    support_label_list.append(1.0)
                    mdsk.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.test_mat:
                            j = np.random.randint(0, self.item_num)
                        support_item_list.append(j)
                        support_label_list.append(0.0)
                        mdsk.append(1.0)
                while len(support_item_list) < self.sup_size:
                    support_item_list.append(self.item_num)
                    support_label_list.append(0.0)
                    mdsk.append(0.0)
                data_dict = {
                    'user': u,
                    'item_list': support_item_list,
                    'label_list': support_label_list,
                    'positive_list': positive_list,  # with padding
                    'positive_mdsk': positive_mdsk,  # making the loss related to padding zero (generate phase)
                    'mdsk': mdsk,  # making the loss related to padding zero (loss phase)
                    'len_pos': len_pos  # for average pooling with positive items
                }
                support_fill[idx] = data_dict
                idx += 1
                self.idx = idx
        self.support_fill = support_fill

    def get_batch(self, idxs):
        len_pos,positive_itm,sup_itm,sup_label,query_itm,query_label,sup_mdsk,query_mdsk,positive_mdsk = [],[],[],[],[],[],[],[],[]
        for idx in idxs:
            len_pos.append(1.0/self.support_fill[idx]['len_pos'])
            positive_itm.append(self.support_fill[idx]['positive_list'])
            sup_itm.append(self.support_fill[idx]['item_list'])
            sup_mdsk.append(self.support_fill[idx]['mdsk'])
            sup_label.append(self.support_fill[idx]['label_list'])
            positive_mdsk.append(self.support_fill[idx]['positive_mdsk'])
        return torch.tensor(positive_itm).cuda(),torch.tensor(sup_itm).cuda(),torch.tensor(sup_label).cuda(),torch.tensor(sup_mdsk).cuda(),torch.tensor(positive_mdsk).cuda(),\
               torch.tensor(query_itm).cuda(),torch.tensor(query_label).cuda(),torch.tensor(query_mdsk).cuda(),torch.tensor(len_pos).cuda()






























class Train_data_exp(data.Dataset):
    def __init__(self, train_data, item_num, train_mat,sup_max,query_max,args):
        super(Train_data_exp, self).__init__()
        self.train_data = train_data
        self.item_num = item_num
        self.train_mat = train_mat
        self.args = args
        self.num_ng = self.args.num_negative_support
        self.positive_size = sup_max
        self.sup_size = sup_max*(self.num_ng+1)
        self.query_size = query_max*(self.num_ng+1)
        self.idx = 0
    def __len__(self):
        return self.idx

    def ng_train_sample(self):
        train_data = deepcopy(self.train_data)
        support_fill, query_fill = defaultdict(dict), defaultdict(dict)
        idx = 0
        self.positive_dict = defaultdict(list)
        for (u, item_list) in train_data.items():

            K = len(item_list) // 2
            if K > 0:
                random.shuffle(item_list)
                support_item_list, support_label_list,positive_list,mdsk,positive_mdsk = [], [],[],[],[]
                positive_list = deepcopy(item_list[:K])
                len_pos = K
                for i in range(len(positive_list)):
                    positive_mdsk.append(1.0)
                while(len(positive_list)<self.positive_size):
                    positive_list.append(self.item_num)
                    positive_mdsk.append(0.0)
                for v in item_list[:K]:
                    support_item_list.append(v)
                    support_label_list.append(1.0)
                    mdsk.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.train_mat:
                            j = np.random.randint(0, self.item_num)
                        support_item_list.append(j)
                        support_label_list.append(0.0)
                        mdsk.append(1.0)
                while len(support_item_list) < self.sup_size:
                    support_item_list.append(self.item_num)
                    support_label_list.append(0.0)
                    mdsk.append(0.0)
                data_dict = {
                    'user': u,
                    'item_list': support_item_list,
                    'label_list': support_label_list,
                    'positive_list':positive_list,              #with padding
                    'positive_mdsk':positive_mdsk,              #making the loss related to padding zero (generate phase)
                    'mdsk':mdsk,                                #making the loss related to padding zero (loss phase)
                    'len_pos':len_pos                           #for average pooling with positive items
                }
                support_fill[idx] = data_dict
                query_item_list, query_label_list,query_mdsk = [], [],[]
                self.positive_dict[u] = deepcopy(item_list[:K])
                for v in item_list[K:]:
                    query_item_list.append(v)
                    query_label_list.append(1.0)
                    query_mdsk.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.train_mat:
                            j = np.random.randint(0, self.item_num)
                        query_item_list.append(j)
                        query_label_list.append(0.0)
                        query_mdsk.append(1.0)
                while len(query_item_list) < self.query_size:
                    query_item_list.append(self.item_num)
                    query_label_list.append(0)
                    query_mdsk.append(0.0)
                data_dict = {
                    'user': u,
                    'item_list': query_item_list,
                    'label_list': query_label_list,
                    'mdsk':query_mdsk                                #making the loss related to padding zero (loss phase)
                }
                query_fill[idx] = data_dict
                idx += 1
                self.idx = idx
            self.support_fill = support_fill
            self.query_fill = query_fill

    '''
    fill:{idx:{'item_lst':...,'label_lst':...,'user_lst':...}}
    '''
    def get_batch(self, idxs):
        users,len_pos,positive_itm,sup_itm,sup_label,query_itm,query_label,sup_mdsk,query_mdsk,positive_mdsk = [],[],[],[],[],[],[],[],[],[]
        for idx in idxs:
            users.append(self.support_fill[idx]['user'])
            len_pos.append(1.0/self.support_fill[idx]['len_pos'])
            positive_itm.append(self.support_fill[idx]['positive_list'])
            sup_itm.append(self.support_fill[idx]['item_list'])
            sup_label.append(self.support_fill[idx]['label_list'])
            sup_mdsk.append(self.support_fill[idx]['mdsk'])
            query_itm.append(self.query_fill[idx]['item_list'])
            query_label.append(self.query_fill[idx]['label_list'])
            query_mdsk.append(self.query_fill[idx]['mdsk'])
            positive_mdsk.append(self.support_fill[idx]['positive_mdsk'])
        return users,torch.tensor(positive_itm).cuda(),torch.tensor(sup_itm).cuda(),torch.tensor(sup_label).cuda(),torch.tensor(sup_mdsk).cuda(),torch.tensor(positive_mdsk).cuda(),\
               torch.tensor(query_itm).cuda(),torch.tensor(query_label).cuda(),torch.tensor(query_mdsk).cuda(),torch.tensor(len_pos).cuda()



