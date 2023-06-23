from utilsdtiseed import *
from modeltestdtiseed import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity as cos

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure, seed)
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5
dim = 1
lr1 = 1e-6
lr2 = 1e-5
lr3 = 1e-4
lr4 = 5e-3
weight_decay = 1e-10
epochs = 10000

reg_loss_co = 0.0001
fold = 0
dir = "../modelSave"

args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"


def adjust_learning_rate(optimizer, epoch, lr2, lr3, lr4):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if optimizer.param_groups[1]['lr'] > lr1:
        optimizer.param_groups[1]['lr'] = lr2 * (0.7 ** (epoch // 500))
    if optimizer.param_groups[2]['lr'] > lr1:
        optimizer.param_groups[2]['lr'] = lr3 * (0.6 ** (epoch // 600))
    if optimizer.param_groups[3]['lr'] > lr2:
        optimizer.param_groups[3]['lr'] = lr4 * (0.55 ** (epoch // 700))


for name in ["Es"]:
    # for name in ["heter","Es","GPCRs","ICs","Ns","zheng"]:
    dtidata, graph, num, all_meta_paths = load_dataset(name)
    # dataName heter Es GPCRs ICs Ns zheng
    dti_label = torch.tensor(dtidata[:, 2:3]).to(args['device'])

    hd = torch.randn((num[0], in_size))
    hp = torch.randn((num[1], in_size))
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])

    node_feature = [features_d, features_p]

    dti_cl = get_clGraph(dtidata, "dti").to(args['device'])

    cl = dti_cl
    data = dtidata
    label = dti_label


    def main(tr, te, seed):
        all_acc = []
        all_roc = []
        all_f1 = []
        for i in range(len(tr)):
            f = open(f"{i}foldtrain.txt", "w", encoding="utf-8")
            train_index = tr[i]
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = te[i]
            f = open(f"{i}foldtest.txt", "w", encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")

            model = GSLDTI(
                all_meta_paths=all_meta_paths,
                in_size=[hd.shape[1], hp.shape[1]],
                hidden_size=[hidden_size, hidden_size],
                out_size=[out_size, out_size],
                dropout=dropout,
                dim=dim
            ).to(args['device'])
            optimizer = torch.optim.Adam([
                {"params": model.HAN_DTI.parameters(), "lr": lr1},
                {"params": model.ENCODER.parameters(), "lr": lr2},
                {"params": model.GCN.parameters(), "lr": lr3},
                {"params": model.MLP.parameters(), "lr": lr4},
            ],
                weight_decay=weight_decay
            )
            best_acc = 0
            best_f1 = 0
            best_roc = 0
            flag = 0
            for epoch in tqdm(range(epochs), ncols=85):
                adjust_learning_rate(optimizer, epoch, lr2, lr3, lr4)
                loss, train_acc, task1_roc, acc, task1_roc1, task1_pr = train(model, optimizer, train_index, test_index,
                                                                              epoch, i)
                if acc > best_acc:
                    best_acc = acc
                if task1_pr > best_f1:
                    best_f1 = task1_pr
                if task1_roc1 > best_roc:
                    best_roc = task1_roc1
                    flag = epoch
            all_acc.append(best_acc)
            all_roc.append(best_roc)
            all_f1.append(best_f1)
            print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} bestROC is {best_roc:.4f} in {flag} epoch ")

        print(
            f"{name},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f}，")


    def train(model, optim, train_index, test_index, epoch, fold):
        model.train()
        out, d, p = model(graph, node_feature, train_index, data)

        train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)

        task1_roc = get_roc(out, label[train_index])

        reg = get_L2reg(model.parameters())

        loss = F.nll_loss(out, label[train_index].reshape(-1)) + reg_loss_co * reg

        optim.zero_grad()
        loss.backward()
        optim.step()
        te_acc, te_task1_roc1, te_task1_pr = main_test(model, d, p, test_index, epoch, fold)
        return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr


    def main_test(model, d, p, test_index, epoch, fold):
        model.eval()

        out = model(graph, node_feature, test_index, data, iftrain=False, d=d, p=p)

        acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(dtype=float) / len(test_index)

        task_roc = get_roc(out, label[test_index])

        task_pr = get_pr(out, label[test_index])
        return acc1, task_roc, task_pr


    train_indeces, test_indeces = get_cross(dtidata)
    main(train_indeces, test_indeces, seed)
