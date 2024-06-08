import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def read_data():
    with open("train_sentence", "r") as f:
        train_sentence = json.load(f)
    with open("test_sentence", "r") as f:
        test_sentence = json.load(f)
    with open("sarcasm_sentence", "r") as f:
        sarcasm_sentence = json.load(f)


    with open("train_token", "r") as f:
        train_token = json.load(f)
    with open("test_token", "r") as f:
        test_token = json.load(f)
    with open("sarcasm_token", "r") as f:
        sarcasm_token = json.load(f)

    with open('y_train.npy', 'rb') as f:
        y_train = np.load(f)
    with open('y_test.npy', 'rb') as f:
        y_test = np.load(f)
    with open('y_sarcasm.npy', 'rb') as f:
        y_sarcasm = np.load(f)

    return train_sentence, test_sentence, sarcasm_sentence, train_token, test_token, sarcasm_token, y_train, y_test, y_sarcasm

def f1_pos(y_true, y_pred):
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 2 and y_pred[i] == 2:
            tp += 1
    recall = tp / np.sum(y_true == 2)
    precision = tp / np.sum(y_pred == 2)
    f1 = (2*recall*precision)/(recall + precision)
    if np.isnan(f1):
        return 0
    else:
        return f1

def f1_neg(y_true, y_pred):
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
    recall = tp / np.sum(y_true == 1)
    precision = tp / np.sum(y_pred == 1)
    f1 = (2*recall*precision)/(recall + precision)
    if np.isnan(f1):
        return 0
    else:
        return f1

def f1_pn(y_true, y_pred):
    return (f1_pos(y_true, y_pred) + f1_neg(y_true, y_pred))/2



def result(y_test, y_sarcasm, test_pred, sarcasm_pred, show=True):
    '''
    :param test_pred: prediction of test dataset
    :param sarcasm_pred: prediction of sarcasm dataset
    :return:
    '''
    acc = accuracy_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred, average="macro")
    f1 = f1_pn(y_test, test_pred)
    acc_on_sarcasm = accuracy_score(y_sarcasm, sarcasm_pred)
    recall_on_sarcasm = recall_score(y_sarcasm, sarcasm_pred, average="macro")
    f1_on_sarcasm = f1_pn(y_sarcasm, sarcasm_pred)
    if show:
        print(f'Average Accuracy is {acc}')
        print(f'Average Recall is {recall}')
        print(f'F1-Score is {f1}')
        print(f'Average Accuracy on sarcasm dataset is {acc_on_sarcasm}')
        print(f'Average Recall on sarcasm dataset is {recall_on_sarcasm}')
        print(f'F1-Score on sarcasm dataset is {f1_on_sarcasm}')

    return acc, recall, f1, acc_on_sarcasm, recall_on_sarcasm, f1_on_sarcasm



