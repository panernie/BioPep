import logging
import math
from Umami.classification.umami_my.configs import fixed_flag
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from Umami.classification.umami_my.model.bert_cnn import BertCNNModel
from Umami.classification.umami_my.frame.dataloaders import att_data_loader

Length = 27



def converts(temp):
    x_train = []
    for i in temp:
        lists = []
        lists[:0] = i
        lists = [int(i) for i in lists]
        x_train.append(lists)
    return x_train


# frame
class AttFrame(nn.Module):
    def __init__(self, batch_size, lr, max_epoch,x_train,y_train,x_test,y_test,k,proportion):
        super().__init__()
        self.model = BertCNNModel().cuda()
        self.is_bert = True
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.k = k
        self.proportion = proportion

        print("------loading data------")
        if self.is_bert:
            self.fixed = fixed_flag
            if self.is_bert:
                self.train_loader = att_data_loader(x_train, y_train,shuffle=True, batch_size=self.batch_size,
                                                    is_bert=True, is_fixed=self.fixed)
                self.test_loader = att_data_loader(x_test, y_test,shuffle=True, batch_size=self.batch_size,
                                                   is_bert=True,
                                                   is_fixed=self.fixed)
            else:
                self.train_loader = att_data_loader(x_train, y_train, shuffle=False, batch_size=self.batch_size,
                                                    is_bert=False)
                self.test_loader = att_data_loader(x_test, y_test, shuffle=True, batch_size=self.batch_size,
                                                   is_bert=False)

            self.loss_func = nn.BCELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')


        else:
            pd_train = pd.read_csv(
                'D:/Projects/DNA_enhancer0/enhancer_my/single/num_organ/num_train_blood_expressed_enhancers.csv')

        print("------finish------")


    # Training Start
    def train_start(self):
        print(f'Fold {self.k + 1} Cross validation starts！')
        best_f1, best_acc, best_recall, best_a = 1e-10, 1e-10, 1e-10, 1e-10
        best_fp, best_tp = 1e-10, 1e-10
        loss_log, acc_log, recall_log, sp_log, sn_log, mcc_log, f1_log, mcm_log = [], [], [], [], [], [], [], []
        best_pre_list, best_gold_list, best_ids_list = [], [], []
        metrics = []
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            is_test = False
            for data in t:
                if self.is_bert:
                    inputs, labels, masks, ids = data
                    masks = masks.cuda()
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    out = self.model(inputs, attention_mask=masks)
                else:
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    out = self.model(inputs, is_test)
                loss = self.loss_func(out, labels.float())
                train_loss += loss.item()
                t.set_postfix(loss=loss)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            self.scheduler.step(avg_loss)
            loss_log.append(avg_loss)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")

            # Validate model
            self.model.eval()
            t = tqdm(self.test_loader)
            pre_num, gold_num, correct_num = 1e-10, 1e-10, 1e-10
            pre_list, gold_list = [], []
            old_pre_list = []
            old_ids_list = []
            pre_list2 = []
            with torch.no_grad():
                all_inputs = []
                all_labels = []
                all_out = []
                for iter_s, batch_samples in enumerate(t):
                    if self.is_bert:
                        inputs, labels, masks, ids = batch_samples
                        masks = masks.cuda()
                        inputs = inputs.cuda()
                        rel_out = self.model(inputs, attention_mask=masks)
                    else:
                        inputs, labels = batch_samples
                        inputs = inputs.cuda()
                        rel_out = self.model(inputs, is_test)
                    # Calculation of evaluation index
                    labels = labels.numpy()
                    ids = np.array(ids)
                    rel_out = rel_out.to('cpu').numpy()
                    all_labels.append(labels)
                    all_out.append(rel_out)
                    idx = inputs.cpu().numpy()
                    all_inputs.append(idx)

                    for pre, gold, id in zip(rel_out, labels, ids):
                        pre_set = np.round(pre)  # 取整
                        pre_set_1 = pre
                        old_pre_list.append(pre_set_1)
                        pre_list2.append(float(pre_set))
                        pre_list.append(float(pre))
                        gold_set = gold
                        gold_list.append(gold_set)
                        pre_num += 1
                        gold_num += 1
                        if pre_set == gold_set:
                            correct_num += 1
                            if gold == 0:
                                old_ids_list.append(id)
            print()
            print('Correct number: %.0f'% (correct_num))
            print('Number of forecasts: %.0f'% (pre_num))
            acc_1 = correct_num / pre_num
            print("Accuracy: %.4f"% (acc_1))

            tn, fp, fn, tp = confusion_matrix(gold_list, pre_list2).ravel()
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            sp, sn = tn / (tn + fp), tp / (tp + fn)
            mcc = (tp * tn - tp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            f1_score = 2 * precision * recall / (precision + recall)
            acc_log.append(acc_1)
            recall_log.append(recall)
            f1_log.append(f1_score)
            sp_log.append(sp)
            sn_log.append(sn)
            mcc_log.append(mcc)
            if best_a < acc_1:
                best_a = acc_1
                best_ids_list = old_ids_list
            if best_f1 < f1_score:
                best_f1, best_acc, best_recall = f1_score, precision, recall
                best_pre_list = pre_list
                best_gold_list = gold_list

            true_pos = fp / (fp + fn)
            false_pos = tp / (tp + fn)
            if best_fp < false_pos and best_tp < true_pos:
                best_fp = false_pos
                best_tp = true_pos

            print('tp %.4f,fp %.4f'% (true_pos,false_pos))
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
            print('best acc: %.4f'% (best_a))
        print(best_f1, best_acc, best_recall, best_a)
        best_sp = max(sp_log)
        best_sn = max(sn_log)
        mcc_log = [elem if not np.isnan(elem) else None for elem in mcc_log]
        while None in mcc_log:
            mcc_log.remove(None)

        best_mcc = max(mcc_log)
        ids_list = list(set(best_ids_list))

        return best_a,best_mcc,best_sp,best_sn,best_gold_list, best_pre_list,ids_list,
