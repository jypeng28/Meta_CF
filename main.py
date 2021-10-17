import random
import learner
from utils.parser import *
from utils.Logging import *
from test import *
import numpy as np
import torch

if __name__=='__main__':
    args = parse_args()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    np.random.seed(1)
    gpu = 'cuda:' + str(args.gpu_id)
    if args.mod=='train':
        Learner = learner.meta_learner(args)
        Learner.train()
    elif args.mod=='test':
        test(args)
    else:
        pass


