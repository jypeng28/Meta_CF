import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from collections import defaultdict
import math
from time import time
import utils.evaluation as evaluation
import random
class model3_linear(torch.nn.Module):
    def __init__(self, item_num,args):
        super(model3_linear, self).__init__()
        self.args = args
        self.item_num = item_num
        self.lam_u = args.regs
        self.item_embeddings = torch.mul(torch.randn(self.item_num+1,32),0.01).cuda()
        item_embeddings = np.load('./pretrain/'+args.dataset+'/item_pretrained.npy')
        self.item_embeddings[:-1] = torch.tensor(item_embeddings).cuda()
        self.item_embeddings[-1] = torch.zeros(32)
        self.item_embeddings.requires_grad = False

        self.generate_layer1 = torch.nn.Linear(32, 32)
        self.local_lr = self.args.local_lr
        torch.nn.init.normal_(self.generate_layer1.weight, 0, 0.01)
        torch.nn.init.normal_(self.generate_layer1.bias, 0, 0.01)

        self.global_lr = self.args.global_lr
        self.lossfunction = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.meta_optim = torch.optim.SGD([
                                           {'params': self.generate_layer1.weight, 'lr': self.global_lr},
                                           {'params': self.generate_layer1.bias, 'lr': self.global_lr*0.1}
                                           ])

    def predict(self, user_embedding, itm_lst):
        return torch.mm(user_embedding, self.item_embeddings[itm_lst].t()).view(-1)


    def loss_func(self, predicted, label_list, user_embedding, item_embedding, mdsk,len_pos):
        loss = torch.mul(self.lossfunction(predicted, label_list), mdsk).sum() + self.lam_u * (user_embedding.norm()*math.sqrt(len_pos.mean()) + item_embedding.norm())
        return loss


    def global_forward(self, positive_x, supp_x, supp_y, sup_mdsk, positive_mdsk, query_x, query_y, query_mdsk, len_pos):
        loss_sum = 0.0
        pos_embedding = torch.mul(self.item_embeddings[positive_x], positive_mdsk.unsqueeze(2).repeat(1, 1, 32)).sum(1)
        pos_embedding = torch.mul(pos_embedding, len_pos.unsqueeze(1))
        user_embedding = self.generate_layer1(pos_embedding)
        user_embedding = user_embedding.unsqueeze(1)
        for epoch in range(self.args.local_epoch):
            predicted = torch.bmm(user_embedding, self.item_embeddings[supp_x].permute(0, 2, 1)).squeeze(1)
            loss = self.loss_func(predicted, supp_y, user_embedding, self.item_embeddings[supp_x], sup_mdsk,len_pos)
            grad_user_embedding = torch.autograd.grad(loss, user_embedding, retain_graph=True)[0]
            user_embedding = user_embedding - torch.mul(grad_user_embedding, self.local_lr)

        q_predicted = torch.bmm(user_embedding, self.item_embeddings[query_x].permute(0, 2, 1)).squeeze(1)
        q_loss = self.loss_func(q_predicted, query_y, user_embedding, self.item_embeddings[query_x], query_mdsk,len_pos)
        self.meta_optim.zero_grad()
        q_loss.backward()
        self.meta_optim.step()

        return q_loss


    def evaluate_test(self, train_data, val_negative_dict, val_negative,sup_dict):
        keep_weight = deepcopy(self.state_dict())
        hr_list, ndcg_list = [],[]
        idxs = range(train_data.idx)
        ts = []
        for idx in idxs:
            positive_x, supp_x, supp_y, sup_mdsk, positive_mdsk, _,_,_, len_pos = train_data.get_batch(
                [idx])
            u = train_data.support_fill[idx]['user']
            pos_embedding = torch.mul(self.item_embeddings[positive_x],
                                      positive_mdsk.unsqueeze(2).repeat(1, 1, 32)).sum(1)
            pos_embedding = torch.mul(pos_embedding, len_pos.unsqueeze(1))

            user_embedding = self.generate_layer1(pos_embedding)
            user_embedding = user_embedding.unsqueeze(1)
            for epoch in range(self.args.local_epoch):
                t1 = time()
                predicted = torch.bmm(user_embedding, self.item_embeddings[supp_x].permute(0, 2, 1)).squeeze(1)
                loss = self.loss_func(predicted, supp_y, user_embedding, self.item_embeddings[supp_x], sup_mdsk,len_pos)
                grad_user_embedding = torch.autograd.grad(loss, user_embedding, retain_graph=True)[0]
                user_embedding = user_embedding - torch.mul(grad_user_embedding, self.local_lr)
                t2 = time()
                ts.append(t2-t1)
            hr, ndcg = evaluation.metrics_meta(self, user_embedding[0], u, val_negative_dict, val_negative,
                                               self.args.topK,self.item_num,train_data)
            hr_list.extend(hr)
            ndcg_list.extend(ndcg)
            self.load_state_dict(keep_weight)
        return hr_list, ndcg_list

    def evaluate(self, train_data, val_negative_dict, val_negative):
        keep_weight = deepcopy(self.state_dict())
        val_hr_list, val_ndcg_list, train_loss, val_loss = [], [], [], []
        if self.args.mod=='train':
            idxs = random.sample(range(len(train_data.support_fill)), 1000)
        else:
            idxs = range(train_data.idx)
        for idx in idxs:
            positive_x, supp_x, supp_y, sup_mdsk, positive_mdsk, _,_,_, len_pos = train_data.get_batch(
                [idx])
            u = train_data.support_fill[idx]['user']
            pos_embedding = torch.mul(self.item_embeddings[positive_x],
                                      positive_mdsk.unsqueeze(2).repeat(1, 1, 32)).sum(1)
            pos_embedding = torch.mul(pos_embedding, len_pos.unsqueeze(1))

            user_embedding = self.generate_layer1(pos_embedding)
            user_embedding = user_embedding.unsqueeze(1)
            for epoch in range(self.args.local_epoch):
                predicted = torch.bmm(user_embedding, self.item_embeddings[supp_x].permute(0, 2, 1)).squeeze(1)
                loss = self.loss_func(predicted, supp_y, user_embedding, self.item_embeddings[supp_x], sup_mdsk,len_pos)
                grad_user_embedding = torch.autograd.grad(loss, user_embedding, retain_graph=True)[0]
                user_embedding = user_embedding - torch.mul(grad_user_embedding, self.local_lr)
            hr, ndcg = evaluation.metrics_meta(self, user_embedding[0], u, val_negative_dict, val_negative,
                                               self.args.topK,self.item_num,train_data)
            val_hr_list.extend(hr)
            val_ndcg_list.extend(ndcg)
            self.load_state_dict(keep_weight)

        return np.mean(val_hr_list), np.mean(val_ndcg_list)
