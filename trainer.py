import torch
from tqdm import tqdm
import numpy as np
from loss.focal_loss import FocalLoss
from loss.scl_loss import ce_scl_loss
from model.punc_model import PunctuationRestoration
from transformers import AdamW
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from optimizers.lookahead.optimizer import Lookahead
from utils.dataloader import get_datasets, get_test_datasets, get_data_loaders, get_test_data_loaders, get_all_targets
from utils.eprint import eprint, f1_printer
from utils.evaluation import evaluate
from utils.get_gradients import get_total_grad_norm
from utils.save_load_model import save_model

from utils.scheduler import LinearScheduler
import os


def train(config):
    device = config.model.device
    model = PunctuationRestoration(config).to(device)
    model = torch.nn.DataParallel(model)
    model.to(device)

    if config.train.optimizer == 'lookahead':
        optimizer = Adam(model.parameters(), lr=config.train.lr)
        optimizer = Lookahead(optimizer=optimizer, k=6, alpha=0.5)
    elif config.train.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.train.lr)
    else:
        raise ValueError("You need to specify an optimizer!")
    return_last_state = False
    if config.train.loss == 'ce+scl':
        return_last_state = True
    train_ds, valid_ds = get_datasets(config)
    testref_ds, testasr_ds = get_test_datasets(config)
    train_loader, valid_loader = get_data_loaders(train_ds, valid_ds, config)
    testref_loader, testasr_loader = get_test_data_loaders(testref_ds, testasr_ds, config)
    train_targets, valid_targets, testasr_targets, testref_targets = get_all_targets(train_loader, valid_loader,
                                                                                     testasr_loader, testref_loader)
    if config.train.scl.weight:
        _, weights = np.unique(train_targets, return_counts=True)
        weights = weights / weights.sum()
        weights = torch.tensor(weights, device=device, dtype=torch.float)
    else:
        weights = None
    max_dev_metric = {'dev': 0, 'ref': 0, 'asr': 0}
    max_ref_metric = 0
    max_asr_metric = 0
    ce_criterion = torch.nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2)
    scheduler = LinearScheduler(optimizer, config.train.warmup_steps)

    summary_writer = SummaryWriter(comment=config.experiment.name)
    summary_counter = 0
    # Write config to tensorboard
    summary_writer.add_text('config', str(config.toDict()))
    for epoch in range(config.train.epoch):
        eprint("start epoch {}".format(epoch))
        pbar = tqdm(train_loader)
        loss = 0
        for data in pbar:
            model.train()
            optimizer.zero_grad()
            text, targets = data
            if config.train.maskout_subword:
                mask = (targets != -1) + 0
            else:
                mask = torch.ones_like(targets)
            if return_last_state:
                preds, h0 = model(text.to(device), mask.to(device), return_last_state=True)
            else:
                preds = model(text.to(device), mask.to(device))

            not_a_word_mask = (targets == -1).to(device)
            word_mask = ~not_a_word_mask
            targets[not_a_word_mask] = 0
            if config.train.loss == 'ce':
                loss = ce_criterion(preds.reshape(-1, config.model.num_class),
                                    targets.to(device).reshape(-1))

                loss.mean().backward()
            elif config.train.loss == 'focal':
                loss = focal_loss(preds.reshape(-1, config.model.num_class),
                                  targets.to(device).reshape(-1))

                loss.mean().backward()
            elif config.train.loss == 'ce+scl':
                assert return_last_state, 'No last state, no scl!'
                loss = ce_scl_loss(preds.reshape(-1, config.model.num_class),
                                   targets.to(device).reshape(-1), h0,
                                   lambda_value=config.train.scl.lambda_value,
                                   temperature=config.train.scl.temperature,
                                   pooling=config.train.scl.pooling,
                                   weight=weights,
                                   device=device)

                loss.mean().backward()
            # 　No clip actually
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            grads = get_total_grad_norm(model.parameters())
            optimizer.step()
            scheduler.step()
            loss = loss.mean().item()
            pbar.set_description("loss: {} grad: {}".format('%.4f' % loss, '%.4f' % grads))
            summary_writer.add_scalar('train_loss', loss, summary_counter)
            summary_writer.add_scalar('train_grad', grads, summary_counter)
            summary_counter += 1
        # evaluation
        with torch.no_grad():
            model.eval()

            def write_summary(writer, prefix, metrics, counter):
                writer.add_scalar('{}_total_f1'.format(prefix), metrics['total'], counter)
                writer.add_scalar('{}_period_f1'.format(prefix), metrics['period'], counter)
                writer.add_scalar('{}_question_f1'.format(prefix), metrics['question'], counter)
                writer.add_scalar('{}_comma_f1'.format(prefix), metrics['comma'], counter)

            valid_metric, _ = evaluate(model, valid_loader,
                                       valid_targets, device)
            write_summary(summary_writer, 'valid', valid_metric, summary_counter)
            testref_metric, _ = evaluate(model, testref_loader,
                                         testref_targets, device)
            write_summary(summary_writer, 'testref', testref_metric, summary_counter)
            testasr_metric, _ = evaluate(model, testasr_loader,
                                         testasr_targets, device)
            write_summary(summary_writer, 'testasr', testasr_metric, summary_counter)
            # print metrics
            f1_printer('valid', valid_metric)
            f1_printer('testasr', testasr_metric)
            f1_printer('testref', testref_metric)
            # update max f1 to tensorboard
            if max_dev_metric['dev'] < valid_metric['total']:
                max_dev_metric['dev'] = valid_metric['total']
                max_dev_metric['ref'] = testref_metric['total']
                max_dev_metric['asr'] = testasr_metric['total']
                best_epoch = epoch
                summary_writer.add_text('dev_valid_max_f1', str(max_dev_metric['dev']), summary_counter)
                summary_writer.add_text('ref_valid_max_f1', str(max_dev_metric['ref']), summary_counter)
                summary_writer.add_text('asr_valid_max_f1', str(max_dev_metric['asr']), summary_counter)
            if max_asr_metric < testasr_metric['total']:
                max_asr_metric = testasr_metric['total']
                summary_writer.add_text('asr_max_f1', str(max_asr_metric), summary_counter)
            if max_ref_metric < testref_metric['total']:
                max_ref_metric = testref_metric['total']
                summary_writer.add_text('ref_max_f1', str(max_ref_metric), summary_counter)
            summary_writer.add_scalar('dev_valid_max_f1', max_dev_metric['dev'], summary_counter)
            summary_writer.add_scalar('ref_valid_max_f1', max_dev_metric['ref'], summary_counter)
            summary_writer.add_scalar('asr_valid_max_f1', max_dev_metric['asr'], summary_counter)
            summary_writer.add_scalar('asr_max_f1', max_asr_metric, summary_counter)
            summary_writer.add_scalar('ref_max_f1', max_ref_metric, summary_counter)

            os.makedirs('saved_model/{}'.format(config.experiment.name), exist_ok=True)
            path = 'saved_model/{}/{:03d}.pt'.format(config.experiment.name, epoch)
            save_model(path, model)
    return model, summary_writer
