import codecs
import json
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from configs import *
from frame.attframe import AttFrame
from sklearn.metrics import roc_curve, auc

def seed_torch(m_seed=2022):
    random.seed(m_seed)
    np.random.seed(m_seed)
    torch.manual_seed(m_seed)

def plot_roc(labels, predict_prob,proportion,model,acc):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.show()
    plt.savefig('../result/{}/{}/roc{}.svg'.format(proportion,model,acc), dpi=1200)
    plt.savefig('../result/{}/{}/roc{}.jpg'.format(proportion,model,acc))


if __name__ == '__main__':
    #Set parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(seed)
    model = 'Bert-Inception'
    proportion_list = ['1_1','1_2']
    for proportion in proportion_list:
        data = pd.read_csv(f'./data/{proportion}/data.csv')
        #5 fold cross validation
        n_splits = 5
        floder = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        train_files = []
        test_files = []
        for k, (Trindex, Tsindex) in enumerate(floder.split(data)):
            train_files.append(np.array(data)[Trindex].tolist())
            test_files.append(np.array(data)[Tsindex].tolist())
        train_files = train_files
        test_files = test_files

        lacc, lmcc, lsp, lsn, gold_list, pre_list = [], [], [], [], [], []
        for k in range(n_splits):
            x_train, y_train, x_test, y_test, train_ids, test_ids = [], [], [], [], [], []
            for i in train_files[k]:
                seq = i[1]
                x_train.append([j for j in seq])
                y_train.append(i[2])
                train_ids.append(i[0])
            for j in test_files[k]:
                seq = j[1]
                x_test.append([k for k in seq])
                y_test.append(j[2])
                test_ids.append(j[0])
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            framework = AttFrame(batch_size, lr, epoch,x_train,y_train,x_test,y_test,k,proportion)
            best_acc,best_mcc,best_sp,best_sn,best_gold_list, best_pre_list,ids_list = framework.train_start()

            lacc.append(best_acc)
            lmcc.append(best_mcc)
            lsp.append(best_sp)
            lsn.append(best_sn)
            gold_list.append(best_gold_list)
            pre_list.append(best_pre_list)

        for k in range(5):
            df = pd.DataFrame()
            df['label'] = gold_list[k]
            df['pre'] = pre_list[k]
            df.to_csv('../result/{}/{}/roc/{}-fold_roc.csv'.format(proportion, model,k+1))

        mean_acc = float(np.mean(lacc))
        print('----------------mean_acc=',mean_acc,'------------------')
        acc = str(np.mean(lacc)) + '±' + str(np.var(lacc))
        mcc = str(np.mean(lmcc)) + '±' + str(np.var(lmcc))
        sp = str(np.mean(lsp)) + '±' + str(np.var(lsp))
        sn = str(np.mean(lsn)) + '±' + str(np.var(lsn))
        save_metric_dict = {
            'acc': acc,
            'mcc': mcc,
            'sp': sp,
            'sn': sn,
        }
        with codecs.open('../result/{}/{}/'.format(proportion, model) + 'metric.json', 'w', encoding='utf-8') as f:
            json.dump(save_metric_dict, f, indent=4, ensure_ascii=False)
