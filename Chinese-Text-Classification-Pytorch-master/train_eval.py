# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from attack_train import FGSM,PGD,FreeAT,FGM


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    
    if config.attack_train=='fgsm':
        fgsm = FGSM(model=model)
    elif config.attack_train=='pgd':
        pgd = PGD(model=model)
    elif config.attack_train=='FreeAT':
        free_at = FreeAT(model=model)
    elif config.attack_train=='fgm':
        fgm = FGM(model=model)
    
    for epoch in range(config.num_epochs):
        model.train()
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            if config.attack_train=='fgsm':
                fgsm.attack()
                outputs = model(trains)
                loss_adv = F.cross_entropy(outputs, labels)
                loss_adv.backward()
                fgsm.restore()
            
            elif config.attack_train=='pgd':
                pgd_k = 3
                pgd.backup_grad()
                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))

                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    
                    outputs = model(trains)
                    loss_adv = F.cross_entropy(outputs, labels)
                    loss_adv.backward()
                pgd.restore()
            
            elif config.attack_train=='FreeAT':
                m = 5
                free_at.backup_grad()
                for _t in range(m):
                    free_at.attack(is_first_attack=(_t == 0))

                    if _t != m - 1:
                        model.zero_grad()
                    else:
                        free_at.restore_grad()
                    
                    outputs = model(trains)
                    loss_adv = F.cross_entropy(outputs, labels)
                    loss_adv.backward()
                free_at.restore()
            
            elif config.attack_train=='fgm':
                fgm.attack()
                outputs = model(trains)
                loss_adv = F.cross_entropy(outputs, labels)
                loss_adv.backward()
                fgm.restore()
                
            
            optimizer.step()
            model.zero_grad()

        dev_acc, dev_loss = evaluate(config, model, dev_iter)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)

        msg = 'Epoch:{}/{},Val Loss:{},Val Acc:{}'
        print(msg.format(epoch + 1,config.num_epochs, dev_loss, dev_acc))
            
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)