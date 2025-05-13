from transformers import BertForPreTraining, BertConfig, BertModel
from tokenization import FullTokenizer
import tokenization
import numpy as np

np.set_printoptions(threshold=np.inf)
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV, StratifiedKFold, \
    cross_val_predict
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, recall_score, \
    precision_score, roc_curve, auc
from collections import Counter

import torch

if torch.cuda.is_available():
    print("GPU可用！")
else:
    print("GPU不可用，将使用CPU进行计算。")
import os


def generate_input_features_from_seq_list(seqs, labels, tokenizer, pad_to=1024, return_as_np_array=False):
    tseqs = [[tokenization.CLS_TOKEN] + tokenizer.tokenize(s) for s in seqs]
    input_mask = [[1] * len(ts) + [0] * (pad_to - len(ts)) for ts in tseqs]
    segment_ids = [[0] * pad_to for ts in tseqs]

    if pad_to is not None:
        for ts in tseqs:
            assert len(ts) <= pad_to
            # if len(ts) <= pad_to:
            ts += [tokenization.PAD_TOKEN] * (pad_to - len(ts))
            assert len(ts) == pad_to, ts

    input_ids = [tokenizer.convert_tokens_to_ids(tseq) for tseq in tseqs]

    if return_as_np_array:
        input_dict = {
            'input_ids': np.array(input_ids),
            'input_mask': np.array(input_mask),
            'segment_ids': np.array(segment_ids),
            'labels': np.array(labels)
        }
    else:
        input_dict = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'labels': labels
        }

    return input_dict


def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        # seqs.append(str(seq_record.seq))
        # seqs.append(sequence)
        seqs.append(str(seq_record.seq).replace('F', 'W').replace('M', 'C').replace('H', 'C'))
        # seqs.append(str(seq_record.seq).replace('F', 'W').replace('C', 'M').replace('H', 'M'))
        # seqs.append(str(seq_record.seq).replace('F', 'W').replace('M', 'H').replace('C', 'H'))
        # seqs.append(str(seq_record.seq).replace('W', 'F').replace('M', 'C').replace('H', 'C'))
        # seqs.append(str(seq_record.seq).replace('W', 'F').replace('C', 'M').replace('H', 'M'))
        # seqs.append(str(seq_record.seq).replace('W', 'F').replace('M', 'H').replace('C', 'H'))
    return headers, seqs


def parse_fastas(data_file, prepend_m):
    headers, seqs = fasta_read(data_file)

    # if prepend_m:
    #     for i in range(len(seqs)):
    #         if seqs[i][0] != 'M':
    #             seqs[i] = 'M' + seqs[i]

    seqs = [s if len(s) < 1023 else s[:1022] for s in seqs]
    seqs = [s + '*' for s in seqs]

    # mask = np.array([len(s) for s in seqs]) <= 1023
    # print('Sequences being removed due to length:', np.sum(~mask))
    # print('Sequences being removed:', np.array(headers)[~mask], np.array(seqs)[~mask])
    seqs = list(np.array(seqs))
    headers = list(np.array(headers))

    return seqs, headers


def get_bert_embed(input_dict, m, tok, device, normalize=True, \
                   summary_method="MEAN", tqdm_bar=True, batch_size=64):
    m = m.to(device)
    input_ids = input_dict['input_ids']
    attention_mask = input_dict['input_mask']
    # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            attention_mask_gpu_0 = torch.LongTensor(attention_mask[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0, attention_mask_gpu_0)[1]
                print(embed.shape)
            if summary_method == "MEAN":
                # res = m(input_gpu_0, attention_mask_gpu_0)[0]
                embed = torch.mean(m(input_gpu_0, attention_mask_gpu_0)[0], dim=1)
                # print(embed.shape)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    print(np.concatenate(output_list, axis=0).shape)
    return np.concatenate(output_list, axis=0)


if __name__ == "__main__":
    model = BertModel.from_pretrained("/root/autodl-tmp/aminobert")
    tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')
    # data_file = "D:\Mystudy\code\dataset\dthermoplic_protein\最新2021数据集\Charoenkwan.fasta"
    data_file = "/root/autodl-tmp/Charoenkwan.fasta"
    seqs, headers = parse_fastas(data_file=data_file, prepend_m=True)
    print(seqs)
    input_dict = generate_input_features_from_seq_list(seqs, labels=None, tokenizer=tokenizer, pad_to=1024,
                                                       return_as_np_array=True)
    X = get_bert_embed(input_dict, model, tokenizer,
                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), normalize=True,
                       summary_method="MEAN", tqdm_bar=True, batch_size=64)
    print(X.shape)
    print(X)
    # X = function(X)
    X_train = X[0:2964, :]
    X_test = X[2964:, :]
    y_train = np.array([0] * 1482 + [1] * 1482)
    y_test = np.array([0] * 371 + [1] * 371)
    # X_train = X[0:5928, :]
    # X_test = X[5928:, :]
    # y_train = np.array([0] * 2964 + [1] * 2964)
    # y_test = np.array([0] * 742 + [1] * 742)
    # y = [1] * 914*2 + [0] * 791*2
    # Y = np.array(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    svm_classifier = svm.SVC(kernel='rbf', probability=True)

    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_svm = grid_search.best_estimator_
    print("Best Parameters:", best_params)

    cv_scores = cross_val_score(best_svm, X_train, y_train, cv=5, scoring='accuracy')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化存储每一折结果的列表
    acc_scores = []
    spe_scores = []
    sen_scores = []
    roc_auc_scores = []

    # 执行交叉验证
    for train_index, val_index in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        # 训练模型
        best_svm.fit(X_tr, y_tr)

        # 在验证集上进行预测
        y_pred = best_svm.predict(X_val)
        y_pred_proba = best_svm.predict_proba(X_val)[:, 1]  # 获取正类的预测概率

        # 计算准确率
        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        # 计算特异性和敏感性
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        spe_scores.append(specificity)
        sen_scores.append(sensitivity)

        # 计算ROC AUC
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        roc_auc_scores.append(roc_auc)

        # 打印当前折的结果
        print(
            f"Fold results - Accuracy: {acc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}, ROC AUC: {roc_auc:.4f}")

    # 打印平均结果
    print("\nAverage results across folds:")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f}")
    print(f"Mean Specificity: {np.mean(spe_scores):.4f}")
    print(f"Mean Sensitivity: {np.mean(sen_scores):.4f}")
    print(f"Mean ROC AUC: {np.mean(roc_auc_scores):.4f}")

    y_pred = best_svm.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set: %.4f" % accuracy)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # 计算特异性和敏感性
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Specificity on test set: %.4f" % specificity)
    print("Sensitivity on test set: %.4f" % sensitivity)

    # 计算ROC曲线下面积（AUC）
    y_pred_proba = best_svm.decision_function(X_test)  # 使用decision_function获取概率
    auc = roc_auc_score(y_test, y_pred_proba)
    print("ROC AUC on test set: %.4f" % auc)




