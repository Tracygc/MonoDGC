import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset import load
from Cross_Former2 import *
import argparse




def train(dataset, gpu, epoch=40, batch=64):
    nb_epochs = epoch
    batch_size = batch
    patience = 20
    # lr = 0.001
    # l2_coef = 0.0

    lr = args.lr
    l2_coef = args.weight_decay
    # hid_units = 512

    adj, diff, feat, labels, num_nodes = load(dataset)

    feat = torch.FloatTensor(feat).cuda()
    diff = torch.FloatTensor(diff).cuda()
    adj = torch.FloatTensor(adj).cuda()
    labels = torch.LongTensor(labels).cuda()
    # print(labels)

    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]
    num_classes = len(torch.unique(labels))

    model = Cross_Former2(ft_size, max_nodes,num_classes,args,batch_size )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model.cuda()

    cnt_wait = 0
    best = 1e9

    itr = (adj.shape[0] // batch_size) + 1
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        train_idx = np.arange(adj.shape[0])
        np.random.shuffle(train_idx)

        for idx in range(0, len(train_idx), batch_size):
            model.train()
            optimiser.zero_grad()

            batch = train_idx[idx: idx + batch_size]
            mask = num_nodes[idx: idx + batch_size]

            out = model(feat[batch],adj[batch] )


            nll = F.nll_loss(out, labels[batch])
            # print(nll)
            loss = nll

            epoch_loss += loss
            loss.backward()
            optimiser.step()

        epoch_loss /= itr

        # print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, epoch_loss))

        if epoch_loss < best:
            best = epoch_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'{dataset}-{gpu}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    # model.load_state_dict(torch.load(f'{dataset}-{gpu}.pkl'))

    features = feat.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    labels = labels.cuda()

    embeds = model(feat,adj)
    # print(embeds.shape)
    # embeds=embeds.view(-1, args.hidden)
    # out = out
    # int = embeds[0].shape[0]

    x = embeds.cpu().detach().numpy()
    y = labels.cpu().numpy()

    # print(x.shape)
    # print(y.shape)
    #x = x.reshape(x.shape[0], -1)


    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print(np.mean(accuracies), np.std(accuracies))


    f = open("./res/Cross-Former2_{}_{}.txt".format(d, args.l), 'a')
    f.write(
        "lr:{}, wd:{} ,hid:{}, drop:{},attention_dropout:{}, acc_test:{},std:{}".format(args.lr, args.weight_decay,
                                                                                                  args.hidden,
                                                                                                  args.dropout,
                                                                                                  args.attention_dropout,
                                                                                                  np.mean(accuracies),
                                                                                                  np.std(
                                                                                                      accuracies) * 100,
                                                                                                  ))
    f.write("\n")
    f.close()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=1, help='GPU device.')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--l', type=int, default=1,help='Number of Transformer layers')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    args = parser.parse_args()
    # gpu = 1
    # torch.cuda.set_device(gpu)
    batch = [32, 64, 128]
    epoch = [10, 20, 40, 100]

    # batch = [32, 64, 128, 256]
    # epoch = [20, 40, 100, 1000]
    #ds = ['REDDIT-MULTI-5K']
    # 'MUTAG', 'PTC_MR',
    ds = [  'MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI']
    seeds = [123, 132, 321, 312, 231]
    for d in ds:
        print(f'####################{d}####################')
        for b in batch:
                for e in epoch:
                    for i in range(5):
                        seed = seeds[i]
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print(f'Dataset: {d}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        train(d, args.device, e, b)
        print('################################################')
