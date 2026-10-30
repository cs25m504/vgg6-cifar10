## Setup
#Run following commands on google colab terminal
```bash

git clone https://github.com/cs25m504/vgg6-cifar10.git
cd vgg6-cifar10
pip install -r requirements.txt
wandb login #provide API Key when prompted



#For baseline configuration
!python train.py --activation relu --batch_norm True --batch_size 50 --epochs 20  --lr 0.01 --optimizer adam  --weight_decay 0.0005  --seed 42

#Model Performance on Different Configurations
!python run_experiments.py

#Best model
!python train.py --activation gelu --batch_norm True --batch_size 64 --epochs 50  --lr 0.001 --optimizer nadam  --weight_decay 0.0005  --seed 42

#For evaluation
!python evaluate.py
