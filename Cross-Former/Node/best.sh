#
python train.py    --dataset cora     --l 1  --device 0  --lr 0.005  --attention_dropout 0.5
python train.py    --dataset citeseer     --l 1  --device 0  --lr 0.004  --dropout 0.7

python train.py    --dataset cornell     --l 1  --device 3  --lr 0.08
python train.py    --dataset chameleon     --l 1  --device 0  --lr 0.001
python train.py    --dataset texas     --l 1  --device 3  --lr 0.05


python train.py    --dataset wisconsin     --l 1  --device 0  --lr 0.5  --dropout 0.6  --attention_dropout 0.5
python train.py    --dataset squirrel     --l 1  --device 0  --lr 0.001
python train.py    --dataset film     --l 1  --device 3  --lr 0.005    --dropout 0.6


python train.py    --dataset photo     --l 1  --device 3  --lr 0.05 --weight_decay 5e-5 --dropout 0.5 --attention_dropout 0.5
python train.py    --dataset computers     --l 1  --device 0      --lr 0.003  --dropout 0.6   --attention_dropout 0.2