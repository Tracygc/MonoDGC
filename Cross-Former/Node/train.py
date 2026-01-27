import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits,  set_seed
from Cross_Former import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
from torch_geometric.data import Batch


def RunExp(args, dataset, data,  Net, percls_trn, val_lb,seed):

    def train(model, optimizer, data):#只是训练网络，而非训练集
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # print(out.shape)
        # print(data.y.shape)
        nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #print(nll)
        loss = nll
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):#测试网络得结果
        model.eval()
        logits = model(data)
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):#输入的数据是训练集即是训练loss，是测试集即是测试loss
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            #loss = F.nll_loss(logits[mask], data.y[mask])
            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    #percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    #val_lb = int(round(args.val_rate * len(data.y)))

    # randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, seed)

    tmp_net = Net(dataset, args)
    model, data = tmp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
    # train_loss = np.array(train_loss)
    # np.save('./train_loss/epoch_{}'.format(epoch), train_loss)
    # val_loss = np.array(val_loss)
    # np.save('./val_loss/epoch_{}'.format(epoch), val_loss)
    # tmp_test_loss = np.array(tmp_test_loss)
    # np.save('./tmp_test_loss/epoch_{}'.format(epoch), tmp_test_loss)
    return test_acc, best_val_acc,  time_run



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['DCFormer'], default='DCFormer')
    parser.add_argument('--l', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    args = parser.parse_args()
    #set_seed(args.seed)
    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    Net = Cross_Former

    dataset = DataLoader(args.dataset)
    data = dataset[0]


    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    results = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        args.seed=SEEDS[RP]
        set_seed(args.seed)
        test_acc, best_val_acc, time_run = RunExp(args, dataset, data,Net, percls_trn, val_lb, args.seed)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')


    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    #print(uncertainty*100)
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    f = open("./res/Cross-Former_{}_{}.txt".format(args.dataset,args.l),'a')
    f.write("lr:{}, wd:{} ,hid:{}, drop:{},attention_dropout:{}, acc_test:{},std:{},time1:{},time2:{}".format(args.lr, args.weight_decay,args.hidden,args.dropout,args.attention_dropout,test_acc_mean,uncertainty*100,run_sum/(args.runs),1000*run_sum/epochsss))
    f.write("\n")
    f.close()