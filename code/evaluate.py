import json
import os
import pdb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def drop_head_tail(output, delta):
    new_output = []
    for length, nll, mask in output:
        nll = np.array(nll[:int(length)+1])
        mask = np.array(mask[:int(length)]+[0])
        nll = nll - mask*delta
        new_output.append(sum(nll))
    return new_output


def evaluate(model, delta=0.1):

    output = "./output"

    normal = json.load(open(os.path.join(output, "{}_prob_0.json".format(model))))
    detour = json.load(open(os.path.join(output, "{}_prob_1.json".format(model))))
    switch = json.load(open(os.path.join(output, "{}_prob_2.json".format(model))))
    ood = json.load(open(os.path.join(output, "{}_prob_3.json".format(model))))

    normal = drop_head_tail(normal, delta)
    detour = drop_head_tail(detour, delta)
    switch = drop_head_tail(switch, delta)
    ood = drop_head_tail(ood, delta)

    normal = np.array(normal)
    detour = np.array(detour)
    switch = np.array(switch)
    ood = np.array(ood)

    score = -np.concatenate((normal, detour))
    label = np.concatenate((np.ones(len(normal)), np.zeros(len(detour))))
    pre, rec, _t = precision_recall_curve(label, score)
    area = auc(rec, pre)
    print("Normal & Detour ROC_AUC: {:.4f}, PR_AUC: {:.4f}".format(roc_auc_score(label, score), area))

    score = -np.concatenate((normal, switch))
    label = np.concatenate((np.ones(len(normal)), np.zeros(len(switch))))
    pre, rec, _t = precision_recall_curve(label, score)
    area = auc(rec, pre)
    print("Normal & Switch ROC_AUC: {:.4f}, PR_AUC: {:.4f}".format(roc_auc_score(label, score), area))

    score = -np.concatenate((ood, detour))
    label = np.concatenate((np.ones(len(ood)), np.zeros(len(detour))))
    pre, rec, _t = precision_recall_curve(label, score)
    area = auc(rec, pre)
    print("OOD & Detour ROC_AUC: {:.4f}, PR_AUC: {:.4f}".format(roc_auc_score(label, score), area))

    score = -np.concatenate((ood, switch))
    label = np.concatenate((np.ones(len(ood)), np.zeros(len(switch))))
    pre, rec, _t = precision_recall_curve(label, score)
    area = auc(rec, pre)
    print("OOD & Switch ROC_AUC: {:.4f}, PR_AUC: {:.4f}".format(roc_auc_score(label, score), area))

evaluate(model="chengdu_test_10.pth")