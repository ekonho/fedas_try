# things like

python main.py -did 0 -data Cifar100 -nb 100 -m cnn -lbs 16 -gr 40 -ls 5 -algo FedAS -jr 0.1 -nc 20

# 原论文复现（6-layer CNN，gamma=0.001）
python main.py -did 0 -data Cifar100 -nb 100 -m cnn -lbs 16 -gr 40 -ls 5 -algo FedAS -jr 0.1 -nc 20 -lr 0.005 -ldg 0.001

# FedAS with ResNet18 backbone (no maxpool, first conv 3x3 stride=1)
# SGD momentum=0.9 weight_decay=1e-4, grad_clip=10.0, alignment_lr=0.001
python main.py -did 0 -data Cifar100 -nb 100 -m resnet -lbs 128 -gr 60 -ls 5 -algo FedAS -jr 0.1 -nc 20 -lr 0.01 -ldg 0.998


