import torch.nn as nn
import torch
from utils.model_config import *


class PunctuationRestoration(nn.Module):
    def __init__(self, config):
        super(PunctuationRestoration, self).__init__()
        self.config = config
        self.dropout = config.model.dropout
        self.base_model = config.model.name
        self.output_dim = config.model.num_class
        self.tail_enable = config.model.tail.linear
        self.bert_layer_0 = MODELS[self.base_model][0].from_pretrained(self.base_model,
                                                                       add_pooling_layer=False,
                                                                       hidden_dropout_prob=self.dropout,
                                                                       output_hidden_states=True)

        # Freeze bert layers
        if config.model.freeze_base:
            for p in self.bert_layer_0.parameters():
                p.requires_grad = False
        hidden_size = MODELS[self.base_model][2]

        if self.tail_enable:
            self.after_bert_0 = nn.Linear(hidden_size, config.model.tail.linear_dim)
            self.after_bert_1 = nn.BatchNorm1d(config.model.max_len * config.model.tail.linear_dim)
            self.after_bert_2 = nn.Dropout(self.dropout)
            self.after_bert_3 = nn.Linear(config.model.max_len * config.model.tail.linear_dim, config.model.max_len * 4)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=config.model.num_class)
        self.output = nn.Softmax(dim=-1)
        self.device = self.bert_layer_0.device

    def get_tokenizer(self):
        return MODELS[self.base_model][1].from_pretrained(self.base_model)

    def forward(self, x, attn_masks, return_last_state=False):
        self.device = self.bert_layer_0.device
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        y0, _, base_hidden_states = self.bert_layer_0(x, attention_mask=attn_masks)
        x = y0
        if self.tail_enable:
            x_linear = self.after_bert_0(x)
            x_flatten = x_linear.view(x_linear.shape[0], -1)
            x_bn = self.after_bert_1(x_flatten)
            x_bn = self.after_bert_2(x_bn)
            x_flatten = self.after_bert_3(x_bn)
            z = x_flatten.view(-1, self.config.model.max_len, 4)
            if self.config.model.tail.as_scl:
                x = x_linear  # try z?
            # x = self.after_bert(x)

        else:
            x = self.dropout_layer(x)
            z = self.linear(x)
        if return_last_state:
            return z, base_hidden_states[-1:]
        return z

    def freeze_base(self, mode=True, to_last=4):
        if mode:
            for p in self.bert_layer_0.parameters():
                p.requires_grad = False
            for p in self.bert_layer_0.encoder.layer[-to_last:].parameters():
                p.requires_grad = True
        else:
            for p in self.bert_layer_0.parameters():
                p.requires_grad = True
