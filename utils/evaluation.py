from copy import deepcopy
import random
import torch
import numpy as np
import math
from collections import defaultdict
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()



def hr_ndcg(indices_sort_top,index_end_i,top_k):
    hr_topK=0
    ndcg_topK=0
    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg
    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1]
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2)
        count+=1
        if count==top_k:
            break
    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg
    return hr_t,ndcg_t


def metrics_meta(model,user_embedding,user, val_negative_dict, val_negative, top_k,item_num,test_data):
    HR, NDCG = [], []


    # evaluate model

    if model.args.mod=='test':
        gt_len = val_negative_dict[user][1]
        val_item = deepcopy(val_negative[val_negative_dict[user][0]])
        negative_items = []
        for i in range(item_num):
            if (user,i) not in test_data.test_mat and i!=val_item[0] :
                negative_items.append(i)
        val_item.extend(negative_items)
        val_item = torch.LongTensor(val_item).cuda()
        # import pdb; pdb.set_trace()
        predictions = model.predict(user_embedding,val_item)
        new_predictions, indices = torch.topk(predictions, top_k)
        h, n = hr_ndcg(indices, gt_len, top_k)
        HR.append(h)
        NDCG.append(n)



    else:
        val_item = deepcopy(val_negative[val_negative_dict[user][0]])
        for _ in range(model.args.num_negative_evaluation):
            j = np.random.randint(0, item_num)
            while (user,j) in test_data.train_mat:
                j = np.random.randint(0, item_num)
            val_item.append(j)
        val_item = torch.LongTensor(val_item).cuda()
        predictions = model.predict(user_embedding, val_item)
        _, indices = torch.topk(predictions, top_k)
        h, n = hr_ndcg(indices, 1, top_k)
        HR.append(h)
        NDCG.append(n)
    return HR, NDCG


def metrics_test(model, val_negative_dict, val_negative,dict_sup,top_K):
    HRS, NDCGS = defaultdict(list), defaultdict(list)
    for user in val_negative_dict:
        if user%5!=0:
            continue
        sup_num = dict_sup[user]
        gt_len = val_negative_dict[user][1]
        val_item = deepcopy(val_negative[val_negative_dict[user][0]])
        val_user = torch.LongTensor([user] * len(val_item)).cuda()
        val_item = torch.LongTensor(val_item).cuda()
        # import pdb; pdb.set_trace()
        predictions = model.predict(val_user, val_item)[0]
        new_predictions, indices = torch.topk(predictions, top_K)
        h,n = hr_ndcg(indices,gt_len,top_K)
        HRS[sup_num].append(h)
        NDCGS[sup_num].append(n)
    return HRS,NDCGS