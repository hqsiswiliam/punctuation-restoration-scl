import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class BertDataset(Dataset):
    def __init__(self, prefix, config, is_train=False, pre_padding=0):
        self.config = config
        self.is_train = is_train
        with open(config.dataset.path + prefix + "_data.pkl", 'rb') as f:
            texts, targets = pickle.load(f)
            self.encoded_texts = [word for t in texts for word in t]
            self.targets = [t for ts in targets for t in ts]
        if pre_padding > 0:
            tokenizer = AutoTokenizer.from_pretrained(config.model.name)
            pad_id = tokenizer.pad_token_id
            cls_id = tokenizer.cls_token_id
            # because first one is cls
            self.encoded_texts = [cls_id] + [pad_id] * pre_padding + self.encoded_texts[1:]
            self.targets = [-1] + [-1] * pre_padding + self.targets[1:]

    def __getitem__(self, idx):
        shift = np.random.randint(self.config.train.seq_shift) - self.config.train.seq_shift // 2 \
            if self.is_train else 0

        start_idx = idx * self.config.model.max_len + shift
        start_idx = max(0, start_idx)
        end_idx = start_idx + self.config.model.max_len
        return torch.LongTensor(self.encoded_texts[start_idx: end_idx]), \
               torch.LongTensor(self.targets[start_idx: end_idx])

    def __len__(self):
        return len(self.encoded_texts) // self.config.model.max_len + 1


def collate_with_pad_id(pad_id, max_len=512):
    def collate_pad_id(batch):
        texts, targets = zip(*batch)
        return_first = False
        if len(texts) == 1 and len(texts[0]) != max_len:
            texts = (texts[0], torch.tensor([pad_id] * max_len, device=texts[0].device))
            targets = (targets[0], torch.tensor([-1] * max_len, device=targets[0].device))
            return_first = True
        # Padding targets
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_id)
        if return_first:
            return padded_texts[:1], padded_targets[:1]
        return padded_texts, padded_targets

    return collate_pad_id


# Make it private
def _collate_pad_debug(batch):
    texts, targets = zip(*batch)
    # Padding targets
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, padded_targets


def get_datasets(config, pre_padding=0):
    train_dataset = BertDataset("train", config, is_train=True)
    valid_dataset = BertDataset("valid", config, pre_padding=pre_padding)
    return train_dataset, valid_dataset


def get_data_loaders(train_dataset, valid_dataset, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                              num_workers=0, collate_fn=collate_with_pad_id(pad_id, max_len=config.model.max_len),
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                              collate_fn=collate_with_pad_id(pad_id, max_len=config.model.max_len))
    return train_loader, valid_loader


def get_test_datasets(config, pre_padding=0):
    test_dataset = BertDataset("test", config, pre_padding=pre_padding)
    testasr_dataset = BertDataset("testasr", config, pre_padding=pre_padding)
    return test_dataset, testasr_dataset


def get_test_data_loaders(test_dataset, testasr_dataset, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    pad_id = tokenizer.pad_token_id
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                             collate_fn=collate_with_pad_id(pad_id, max_len=config.model.max_len))
    testasr_loader = DataLoader(testasr_dataset, batch_size=config.train.batch_size,
                                collate_fn=collate_with_pad_id(pad_id, max_len=config.model.max_len))
    return test_loader, testasr_loader


def get_all_targets(*loaders):
    flatten = lambda t: [item for sublist in t for item in sublist]
    result = []
    for loader in loaders:
        valid_targets = [targets.numpy().tolist() for _, targets in loader]
        valid_targets = flatten(flatten(valid_targets))
        all_valid_target = np.asarray(valid_targets)
        all_valid_target = all_valid_target[all_valid_target != -1]
        result.append(all_valid_target)
    return result
