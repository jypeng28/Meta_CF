
from models.model1 import model1
from models.model3_linear import model3_linear

import torch.utils.data as data
import torch
from collections import defaultdict
import time
import numpy as np
from models.model3_neural import model3_neural

from utils.Logging import Logging
from copy import deepcopy
import utils.data_prepare as data_prepare
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()
class meta_learner():
    def __init__(self,args):
        self.args = args
        self.log_str_path = './train_log/' + args.model + str(args.number) + args.mod + args.dataset +'GL'+str(args.global_lr)+'LE'+str(args.local_epoch)+ 'LL'+str(args.local_lr)+'.log'
        self.mod_str_path = './saved_models/' + args.model + str(args.number) + args.mod + args.dataset+'GL'+str(args.global_lr)+'LE'+str(args.local_epoch)+'LL'+str(args.local_lr)
        self.log = Logging(self.log_str_path)

        self.user_num, self.item_num, self.train_data, self.train_support_mat, self.val_negative_dict, self.val_negative, self.train_mat, self.sup_max,self.query_max = data_prepare.load_all(
            self.args)

        self.Train_data = data_prepare.Train_data(self.train_data, self.item_num, self.train_support_mat,self.sup_max,self.query_max,self.args)
        self.Train_data.ng_train_sample()
        eval_ = args.model + "(self.item_num,args).cuda()"
        self.model = eval(eval_)
        self.maml_train_batch_sampler = data.BatchSampler(data.RandomSampler(range(self.Train_data.idx)),
                                                            batch_size=self.args.batch_size,
                                                            drop_last=False)



    def train(self):
        print("data loaded")
        max_hr, max_ndcg = 0.0, 0.0
        val_hr, val_ndcg = self.model.evaluate(self.Train_data, self.val_negative_dict,self.val_negative)
        str_to_log = '---epoch:{},train_query_loss:{},val_hr:{},val_ndcg:{}'.format(0, 0, val_hr, val_ndcg)
        self.log.record(str_to_log)
        for epoch in range(self.args.train_epoch):
            total_loss = 0.0
            for batch_idx_list in self.maml_train_batch_sampler:
                positive_itm,sup_itm,sup_label,sup_mdsk,positive_mdsk,query_itm,query_label,query_mdsk,len_pos= self.Train_data.get_batch(batch_idx_list)
                t1 = time.time()
                step_loss = self.model.global_forward(positive_itm,sup_itm,sup_label,sup_mdsk,positive_mdsk,query_itm,query_label,query_mdsk,len_pos)
                total_loss += step_loss
                t2 = time.time()
            val_hr,val_ndcg = self.model.evaluate(self.Train_data, self.val_negative_dict, self.val_negative)
            str_to_log = '---epoch:{},train_query_loss:{},val_hr:{},val_ndcg:{}'.format(epoch,total_loss,val_hr,val_ndcg)
            self.log.record(str_to_log)
            if val_ndcg>max_ndcg:
                torch.save(self.model.state_dict(),self.mod_str_path+'.mod')
            max_hr = max(max_hr,val_hr)
            max_ndcg = max(max_ndcg,val_ndcg)







