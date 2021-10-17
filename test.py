from collections import defaultdict
from  models.model1 import model1
from  models.model3_linear import model3_linear
import torch
import random
import utils.evaluation as evaluation
from utils.Logging import *
import torch.nn as nn

from copy import deepcopy
from models.model3_neural import model3_neural
import numpy as np
import utils.data_prepare as data_prepare



import utils.parser as parser
def test(args):
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # random.seed(1)
    # np.random.seed(1)
    item_num, test_support_data, test_negative_dict, test_negative, test_mat, sup_max= data_prepare.load_all(
        args)
    if args.dataset=='amazon_small':
        item_num=9449

    elif args.dataset=='amazon_big':
        item_num=57790
    else:
        item_num=3952

    test_data = data_prepare.Test_data(test_support_data, item_num, test_mat,sup_max,args)
    test_data.ng_test_sample()
    log_str_path = './test_log/hr'+str(args.topK)+'/' + args.model + str(args.number) + 'test' + args.dataset +'GL'+str(args.global_lr)+'LE'+str(args.local_epoch)+'LL'+str(args.local_lr)
    mod_str_path = './saved_models/' + args.model + str(args.number) + 'train' + args.dataset+'GL'+str(args.global_lr)+'LE'+str(args.local_epoch)+'LL'+str(args.local_lr)
    log = Logging(log_str_path+'.log')
    eval_ = args.model + "(item_num,args).cuda()"
    model = eval(eval_)
    mod = torch.load(mod_str_path + '.mod')
    model.load_state_dict(mod)
    hrs, ndcgs = model.evaluate_test(test_data, test_negative_dict, test_negative,
                                                                           test_data.sup_dict)

    log.record('------hr:{}-------------ndcg{}'.format(np.mean(hrs), np.mean(ndcgs)))


if __name__=='__main__':
    args = parser.parse_args()
    test(args)


