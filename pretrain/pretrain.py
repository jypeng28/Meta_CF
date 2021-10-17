import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.utils.data as data

import argparse
import torch.optim as optim
import random
import os, shutil
from datetime import datetime
import math
import torch.utils.data as data
from copy import deepcopy
import numpy as np
from collections import defaultdict
import data_prepare_pre as data_prepare

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.utils.data as data
import torch.optim as optim
import random
class Logging():
    def __init__(self, log_path):
        self.filename = log_path

    def record(self, str_log):
        now = datetime.now()
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s %s\r\n" % (now.strftime('%Y-%m-%d-%H:%M:%S'), str_log))
            f.flush()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)
str_log_name = './log/pretrain.log'
log = Logging(str_log_name)
## init 0.1-->0.01
class PMFLoss(torch.nn.Module):
    def __init__(self, user_num,item_num,lam_u=0.1):
        super().__init__()
        self.lam_u = lam_u
        self.lossfn = nn.BCEWithLogitsLoss(reduction='sum')
        self.user_embedding = torch.mul(torch.randn(user_num, 32),0.01).cuda()
        self.item_embedding = torch.mul(torch.randn(item_num, 32),0.01).cuda()
        self.user_embedding.requires_grad = True
        self.item_embedding.requires_grad = True
        self.optim = optim.Adam([self.user_embedding, self.item_embedding], lr=0.001)
    def forward(self, user, item, ratings):
        # diag = torch.eye(len(user)).cuda()
        a = torch.tensor([[i for i in range(len(user))]]).cuda()
        # predicted =torch.mul(torch.mm(self.user_embedding[user], self.item_embedding[item].t()),diag).sum(0)
        predicted = torch.mm(self.user_embedding[user], self.item_embedding[item].t()).gather(0,a)[0]
        tmploss = self.lossfn(predicted,ratings)
        u_regularization = self.lam_u * torch.norm(self.user_embedding[user])
        v_regularization = self.lam_u * torch.norm(self.item_embedding[item])
        return tmploss + u_regularization + v_regularization

    def predict(self, user, item):
        predicted = torch.nn.Sigmoid()(torch.mm(self.user_embedding[user], self.item_embedding[item].t()))
        return predicted


def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcgg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics_pmf(model, val_negative_dict, val_negative, top_k):
    HR, NDCG = [], []
    idxs = random.sample(range(user_num), 1000)
    for user in idxs:
        if user in val_negative_dict:
            evaluation_list = val_negative_dict[user]
            for idx in evaluation_list:
                val_item = deepcopy(val_negative[idx])
                gt_item = val_item[0]
                random.shuffle(val_item)
                val_user = torch.LongTensor([user] * len(val_item)).cuda()
                val_item = torch.LongTensor(val_item).cuda()
                # import pdb; pdb.set_trace()
                predictions = model.predict(val_user, val_item)[0]
                new_predictions, indices = torch.topk(predictions, 10)
                recommends = torch.take(val_item, indices).cpu().numpy().tolist()
                HR.append(hit(gt_item, recommends))
                NDCG.append(ndcgg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon_small')
    return parser.parse_args()
if __name__=='__main__':

    args = parse_args()
    print("start to load data")
    user_num,item_num,train_data,  val_negative_dict, val_negative,train_mat,val_mat = data_prepare.load_all(args)
    print('data loaded')
    hrs, ngs,losss = [],[],[]
    pmfloss = PMFLoss(user_num,item_num).cuda()
    Train_data = data_prepare.Test_data(train_data, item_num, train_mat,5)
    Train_data.ng_test_sample()

    hrmax = 0.0
    ndcgmax = 0.0
    for epoch in range(1, 100):
        sampler = data.BatchSampler(data.RandomSampler(range(Train_data.idx)), batch_size=10240, drop_last=False)
        pmfloss.train()
        loss_sum = 0.0
        for u_list in sampler:
            user_list,item_list,label_list = Train_data.get_batch(u_list)
            loss = pmfloss.forward(user_list,item_list,label_list)
            pmfloss.optim.zero_grad()
            loss.backward()
            pmfloss.optim.step()
            loss_sum+=loss.item()
        pmfloss.eval()
        hrtmp,ndcgtmp = metrics_pmf(pmfloss,val_negative_dict,val_negative,10)
        str_log = 'epoch:{}----loss:{}----hr:{}----ndcg{}'.format(epoch,loss_sum,hrtmp,ndcgtmp)
        log.record(str_log)
        if hrtmp> hrmax or ndcgtmp>ndcgmax:
            hrmax = max(hrtmp,hrmax)
            ndcgmax = max(ndcgtmp,ndcgmax)
            np.save('./' + args.dataset + '/' + 'user_pretrained.npy', tensorToScalar(pmfloss.user_embedding))
            np.save('./' + args.dataset + '/' + 'item_pretrained.npy', tensorToScalar(pmfloss.item_embedding))
