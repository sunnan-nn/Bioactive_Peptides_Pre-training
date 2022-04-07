from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
import numpy as np

def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v][h] == 1 or y[v][h] == 1:
                union += 1
            if y_hat[v][h] == 1 and y[v][h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v][h] == 1 or y[v][h] == 1:
                union += 1
            if y_hat[v][h] == 1 and y[v][h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v][h] == 1 or y[v][h] == 1:
                union += 1
            if y_hat[v][h] == 1 and y[v][h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v][h] == 1 or y[v][h] == 1:
                union += 1
            if y_hat[v][h] == 1 and y[v][h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def multi_label_metrics(all_pred, all_gold):
    aiming = Aiming(all_pred, all_gold)
    coverage = Coverage(all_pred, all_gold)
    accuracy = Accuracy(all_pred, all_gold)
    absolute_true = AbsoluteTrue(all_pred, all_gold)
    absolute_false = AbsoluteFalse(all_pred, all_gold)  
    results = {"aiming": aiming, "coverage": coverage, "accuracy": accuracy, "absolute_true": absolute_true, "absolute_false": absolute_false}
    return results

def binary_class_metrics(all_prob, all_pred, all_gold, id2label):
    for i, label in id2label.items():
        c_mat = confusion_matrix(y_pred=all_pred[:, i], y_true=all_gold[:, i])
        sn = c_mat[1, 1] / np.sum(c_mat[1, :])    # 预测正确的正样本
        sp = c_mat[0, 0] / np.sum(c_mat[0, :])    # 预测正确的负样本
        mcc = matthews_corrcoef(y_pred=all_pred[:, i], y_true=all_gold[:, i])
        auc = roc_auc_score(y_score=all_prob[:, i], y_true=all_gold[:, i])
        aupr = average_precision_score(y_score=all_prob[:, i], y_true=all_gold[:, i])
        f1 = f1_score(y_pred=all_pred[:, i], y_true=all_gold[:, i])
        acc = accuracy_score(y_pred=all_pred[:, i], y_true=all_gold[:, i])
        print(f"label:{label} - auc: {auc:.4f} - aupr: {aupr:.4f} - f1: {f1:.4f} - acc: {acc:.4f} - sn: {sn:.4f} - acc: {sp:.4f} - mcc: {mcc:.4f}")

# def scores(all_porb, all_pred, all_gold):
#     # y_predlabel=[(0 if item<th else 1) for item in y_pred]
#     tn,fp,fn,tp=confusion_matrix(all_gold, all_pred).flatten()
#     SPE=tn*1./(tn+fp)
#     MCC=matthews_corrcoef(all_gold, all_pred)
#     Recall=recall_score(all_gold, all_pred)
#     Precision=precision_score(all_gold, all_pred)
#     F1=f1_score(all_gold, all_pred)
#     Acc=accuracy_score(all_gold, all_pred)
#     AUC=roc_auc_score(all_gold, all_porb)
#     precision_aupr, recall_aupr, _ = precision_recall_curve(all_gold, all_porb)
#     AUPR = auc(recall_aupr, precision_aupr)
#     # print(f"auc: {AUC:.4f} - aupr: {AUPR:.4f} - f1: {F1:.4f} - acc: {Acc:.4f}")
#     # print([Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR,tp,fn,tn,fp])
#     return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR,tp,fn,tn,fp]
    

