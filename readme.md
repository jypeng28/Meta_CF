### Basic Information:
This code is released for the paper:

Jingyu Peng, Le Wu, Peijie Sun and Meng Wang. Meta-Learned ID Embeddings for Online Inductive Recommendation
Accepted by CCIR2021.

### Usage:
1. Environment: Python 3.8,torch-1.4.0+cu100
2. To train the meta\_CF models, cd the Meta\_CF directory and execute the command'python main.py --mod=train --model=<model\_name> --dataset=<dataset\_name> --local\_lr=<local\_update\_learning\_rate>  --global_lr=<global\_update\_learning\_rate> --regs=<regularization> --topK=<topK\_used\_when\_evaluation>
3. To test the model trained, use the same parameters as training except the mod replaced by 'test'.

Following this training example:'python main.py --mod=train --model=model3\_linear --dataset=amazon_small --local\_lr=0.01 --global\_lr=0.00001 --local_epoch=5'

Following this test example corresponding to the training example:'python main.py --mod=test --model=model3\_linear --dataset=amazon_small --local\_lr=0.01 --global\_lr=0.00001 --local_epoch=5' 

