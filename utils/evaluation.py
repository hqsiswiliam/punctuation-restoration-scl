from functools import reduce

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def run_valid(model, loader, device):
    model.eval()
    valid_loss = 0
    all_valid_preds = []
    for data in tqdm(loader):
        text, targets = data
        mask = (targets != -1) + 0
        with torch.no_grad():
            preds = model(text.to(device), mask.to(device))

        word_mask = targets != -1
        preds = preds[word_mask]
        targets = targets[word_mask]

        # loss = criterion(preds.view(-1, num_classes), targets.to(device).view(-1))
        # valid_loss += loss.mean().item()

        all_valid_preds.append(preds.detach().cpu().numpy())
    return all_valid_preds


def evaluate(model, loaders, targets, device):
    if loaders.__class__ is not list:
        loaders = [loaders]
    preds = []
    for loader in loaders:
        valid_preds = run_valid(model, loader, device)
        valid_preds = np.concatenate(valid_preds)
        preds.append(valid_preds)
    preds = reduce(lambda x,y: x+y, preds)
    valid_f1 = f1_score(preds.argmax(axis=1), targets, average='macro', labels=[1, 2, 3])
    valid_f1_sep = f1_score(preds.argmax(axis=1), targets, average=None, labels=[1, 2, 3])
    valid_precision = precision_score(preds.argmax(axis=1), targets, average='macro', labels=[1, 2, 3])
    valid_precision_sep = precision_score(preds.argmax(axis=1), targets, average=None, labels=[1, 2, 3])
    valid_recall = recall_score(preds.argmax(axis=1), targets, average='macro', labels=[1, 2, 3])
    valid_recall_sep = recall_score(preds.argmax(axis=1), targets, average=None, labels=[1, 2, 3])
    return {'total': valid_f1, 'period': valid_f1_sep[0],
            'question': valid_f1_sep[1],
            'comma': valid_f1_sep[2],
            'total_precision': valid_precision,
            'period_precision': valid_precision_sep[0],
            'question_precision': valid_precision_sep[1],
            'comma_precision': valid_precision_sep[2],
            'total_recall': valid_recall,
            'period_recall': valid_recall_sep[0],
            'question_recall': valid_recall_sep[1],
            'comma_recall': valid_recall_sep[2]}, preds
