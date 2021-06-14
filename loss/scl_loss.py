import torch
import torch.nn.functional as F
import torch.nn as nn


def ce_scl_loss(preds, ground_truth, hidden_state,
                lambda_value=0.5, temperature=0.7,
                weight=None,
                pooling=False, device='cuda'):
    if weight is not None:
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
    ce_loss = cross_entropy(preds, ground_truth)
    c_loss = scl_loss(hidden_state, ground_truth, temperature, pooling=pooling)
    loss = torch.tensor(1 - lambda_value, device=device) * ce_loss + torch.tensor(lambda_value, device=device) * c_loss
    return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


global_sup_con_loss = None


def get_sup_con_loss(device='cuda'):
    global global_sup_con_loss
    if global_sup_con_loss is None:
        global_sup_con_loss = SupConLoss(contrast_mode='all')
        global_sup_con_loss = nn.DataParallel(global_sup_con_loss)
        global_sup_con_loss.to(device)
    return global_sup_con_loss


def scl_loss(hidden_states, ground_truth, temperature=0.07, pooling=False):
    sup_con_loss = get_sup_con_loss()
    flatten_hidden_states = []
    for hidden_state in hidden_states:
        hidden_state = F.normalize(hidden_state, dim=1)
        flatten_hidden_state = hidden_state.view(hidden_state.shape[0] * hidden_state.shape[1], 1, -1)
        if pooling:
            pooling = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            pooled = pooling(flatten_hidden_state.permute(2, 1, 0)).permute(2, 1, 0)
            flatten_hidden_state = pooled
        flatten_hidden_states.append(flatten_hidden_state)
    torch_hidden_states = torch.cat(flatten_hidden_states, dim=1)
    if len(ground_truth.shape) == 1:
        flatten_ground_truth = ground_truth
    else:
        flatten_ground_truth = ground_truth.view(ground_truth.shape[0] * ground_truth.shape[1])
    sup_con_loss.temperature = temperature

    loss = sup_con_loss(torch_hidden_states, flatten_ground_truth)
    return loss


if __name__ == '__main__':
    import numpy as np

    for t in np.arange(0, 1, 0.001):
        loss = SupConLoss(contrast_mode='one', temperature=t)
        good_data = torch.tensor([[[0., 0.]], [[1, 1.]], [[2., 2.]], [[0., 0.]], [[1, 1.]], [[2., 2.]]]);
        good_label = torch.tensor([0, 1, 2, 0, 1, 2.])
        good_loss = loss(good_data, good_label)
        bad_data = torch.tensor([[[0., 0.]], [[1, 1.]], [[2., 2.]], [[3., 3.]], [[4, 4.]], [[5., 5.]]]);
        bad_label = torch.tensor([0, 1, 2, 0, 1, 2.])
        bad_loss = loss(bad_data, bad_label)
        print("Temperature: {} Good Loss: {} Bad Loss: {}".format('%.4f' % t, good_loss / len(good_data),
                                                                  bad_loss / len(good_data)))
