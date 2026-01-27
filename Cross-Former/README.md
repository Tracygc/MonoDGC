# Our Cross-Former for Attributed Graph Embedding

## Environment Settings    
- pytorch 1.9.0
- numpy 1.21.6
- torch-geometric 2.2.0 
- tqdm 4.64.1
- scipy 1.7.3
- seaborn 0.12.2
- scikit-learn 1.0.2


## Node classification on real-world datasets (./Node/data)
We evaluate the performance of Cross-Former against the competitors on 10 real-world datasets.

### Datasets
We provide the datasets in the folder './Node/data' and you can run the code directly, or you can choose not to download the datasets('./Node/data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script directly and this script describes the hyperparameters settings of AOSENet on each dataset.
```sh
cd ./Node
sh best.sh
```
or run the following Command 
+ Citeseer
```sh
python train.py    --dataset citeseer     --l 1  --device 0  --lr 0.004  --dropout 0.7
```
+ Chameleon
```sh
python train.py    --dataset chameleon     --l 1  --device 0  --lr 0.001
```


## Graph classification on real-world datasets (./Graph/data)
We evaluate the performance of Cross-Former against the competitors on 4 real-world datasets.

### Datasets
We provide the datasets in the folder './Graph/data' and you can run the code directly.

### Running the code

You can run the following script directly and this script describes the hyperparameters settings of AOSENet on each dataset.
```sh
cd ./Graph
python train2.py
```


