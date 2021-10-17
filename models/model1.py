import torch
import torch.nn as nn
import utils.evaluation as evaluation
from time import time
import random
from collections import defaultdict
import math
from copy import deepcopy
import numpy as np
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

class user_preference_estimator(torch.nn.Module):
    def __init__(self, item_num,args):
        super(user_preference_estimator,self).__init__()
        self.lam_u = args.regs
        self.item_num = item_num
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.user_embeddings = torch.nn.Parameter(torch.mul(torch.randn(1, 32),0.01))
        self.item_embeddings = torch.mul(torch.randn(self.item_num + 1, 32), 0.01).cuda()
        item_embeddings = np.load('./pretrain/' + args.dataset + '/item_pretrained.npy')
        self.item_embeddings[:-1] = torch.tensor(item_embeddings).cuda()
        self.item_embeddings[-1] = torch.zeros(32)
        self.item_embeddings = torch.nn.Parameter(self.item_embeddings)
        self.item_embeddings.requires_grad = True
        self.local_lr = args.local_lr
        self.user_embeddings.requires_grad = True

    def forward(self, user_embedding_update,item, labels,mdsk):
        predicted = torch.bmm(user_embedding_update, self.item_embeddings[item].permute(0,2,1)).squeeze(1)
        prediction_error = torch.mul(self.loss_func(predicted, labels),mdsk).sum()
        u_regularization = self.lam_u * (user_embedding_update.norm()+self.item_embeddings[item].norm())
        return prediction_error + u_regularization
    def predict(self, item):
        predicted = torch.mm(self.user_embeddings, self.item_embeddings[item].permute(0,2,1))
        return predicted.view(-1)

class model1(torch.nn.Module):
    def __init__(self,item_num,args):
        super(model1, self).__init__()
        self.args = args
        self.item_num = item_num
        self.model = user_preference_estimator(item_num,args)
    def predict(self,user_embedding,itm_lst):
        return torch.mm(user_embedding, self.model.item_embeddings[itm_lst].t()).view(-1)
    def global_forward(self, positive_x,support_x, support_y, sup_mdsk,positive_mdsk,query_x, query_y,query_mdsk,len_pos):
        t1 = time()
        loss_sum = 0.0
        keep_weight = deepcopy(self.model.state_dict())
        user_embedding_init = deepcopy(self.model.user_embeddings).unsqueeze(0).repeat(len(support_x),1,1).detach_()
        user_embedding_init.requires_grad = True
        self.model.load_state_dict(keep_weight)
        for _ in range(self.args.local_epoch):
            if _ == 0:
                loss = self.model.forward(user_embedding_init, support_x,support_y,sup_mdsk)
                grad_user_embedding = torch.autograd.grad(loss, user_embedding_init, retain_graph=True)[0]
                user_embedding_update = user_embedding_init  - torch.mul(grad_user_embedding, self.model.local_lr)
            else:
                loss = self.model.forward(user_embedding_update,support_x, support_y,sup_mdsk)
                grad_user_embedding = torch.autograd.grad(loss, user_embedding_update,retain_graph=True)[0]
                user_embedding_update = user_embedding_update- torch.mul(grad_user_embedding , self.model.local_lr)
        q_loss = self.model.forward(user_embedding_update,query_x, query_y,query_mdsk)
        loss_sum = loss_sum + q_loss
        q_loss.backward()
        tmp_state_dict = self.model.state_dict()
        tmp_state_dict['user_embeddings'] = tmp_state_dict['user_embeddings'] - self.args.global_lr * user_embedding_init.grad.sum(0)
        self.model.load_state_dict(tmp_state_dict)
        self.model.zero_grad()
        return loss_sum

    def evaluate_test(self, train_data, val_negative_dict, val_negative,sup_dict):
        keep_weight = deepcopy(self.model.state_dict())
        hr_list, ndcg_list = [],[]
        tmp_train_loss = []
        ts = []
        if self.args.mod=='train':
            idxs = random.sample(range(len(train_data.support_fill)), 1000)
        else:
            idxs = range(train_data.idx)
        for idx in idxs:
            positive_x, supp_x, supp_y, sup_mdsk, positive_mdsk, query_x, query_y, query_mdsk, len_pos = train_data.get_batch(
                [idx])
            u = train_data.support_fill[idx]['user']
            user_embedding_init = deepcopy(self.model.user_embeddings).unsqueeze(0)
            for _ in range(self.args.local_epoch):
                t1 = time()
                if _ == 0:
                    loss = self.model.forward(user_embedding_init, supp_x,supp_y,sup_mdsk)
                    grad_user_embedding = torch.autograd.grad(loss, user_embedding_init, retain_graph=True)[0]
                    user_embedding_update = user_embedding_init - torch.mul(grad_user_embedding,
                                                                            self.model.local_lr)
                else:
                    loss = self.model.forward(user_embedding_update, supp_x,supp_y,sup_mdsk)
                    grad_user_embedding = torch.autograd.grad(loss, user_embedding_update, retain_graph=True)[0]
                    user_embedding_update = user_embedding_update - torch.mul(grad_user_embedding,self.model.local_lr)
                t2 = time()
            tmp_train_loss.append(tensorToScalar(loss))
            hr, ndcg = evaluation.metrics_meta(self,user_embedding_update[0],u, val_negative_dict, val_negative, self.args.topK,self.item_num,train_data)
            hr_list.extend(hr)
            ndcg_list.extend(ndcg)
            self.model.load_state_dict(keep_weight)
        return hr_list, ndcg_list


    def evaluate(self, train_data, val_negative_dict, val_negative):
        keep_weight = deepcopy(self.model.state_dict())
        hr_list, ndcg_list = [], []
        tmp_train_loss = []
        if self.args.mod=='train':
            idxs = random.sample(range(len(train_data.support_fill)), 1000)
        else:
            idxs = range(train_data.idx)
        for idx in idxs:
            positive_x, supp_x, supp_y, sup_mdsk, positive_mdsk, query_x, query_y, query_mdsk, len_pos = train_data.get_batch(
                [idx])
            u = train_data.support_fill[idx]['user']
            user_embedding_init = deepcopy(self.model.user_embeddings).unsqueeze(0)
            for _ in range(self.args.local_epoch):
                if _ == 0:
                    loss = self.model.forward(user_embedding_init, supp_x,supp_y,sup_mdsk)
                    grad_user_embedding = torch.autograd.grad(loss, user_embedding_init, retain_graph=True)[0]
                    user_embedding_update = user_embedding_init - torch.mul(grad_user_embedding,
                                                                            self.model.local_lr)
                else:
                    loss = self.model.forward(user_embedding_update, supp_x,supp_y,sup_mdsk)
                    grad_user_embedding = torch.autograd.grad(loss, user_embedding_update, retain_graph=True)[0]
                    user_embedding_update = user_embedding_update - torch.mul(grad_user_embedding,self.model.local_lr)
            tmp_train_loss.append(tensorToScalar(loss))
            hr, ndcg = evaluation.metrics_meta(self,user_embedding_update[0],u, val_negative_dict, val_negative, self.args.topK,self.item_num,train_data)
            hr_list.extend(hr)
            ndcg_list.extend(ndcg)
            self.model.load_state_dict(keep_weight)
        return np.mean(hr_list), np.mean(ndcg_list)

