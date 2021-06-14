import argparse
import torch
from utils.config import get_config
from model.punc_model import PunctuationRestoration
from utils.dataloader import get_test_data_loaders, get_test_datasets, get_all_targets
from utils.evaluation import evaluate
# Add args
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='config path')
parser.add_argument('-ckpt', '--checkpoint', type=str, help='checkpoint path')
args = parser.parse_args()
config_file = args.config
model_checkpoint = args.checkpoint

# Initialize model
punc_set = {'': 0, ',': 3, '.': 1, '?': 2}
id_to_punc = {0: 'O', 3: 'COMMA', 1: 'PERIOD', 2: 'QUESTION'}
device = 'cuda'
config = get_config(config_file)
model = PunctuationRestoration(config).to(device)
model = torch.nn.DataParallel(model)
model.to(device)
checkpoint = torch.load(model_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

testref_ds, testasr_ds = get_test_datasets(config)
testref_loader, testasr_loader = get_test_data_loaders(testref_ds, testasr_ds, config)
testasr_targets, testref_targets = get_all_targets(testasr_loader, testref_loader)


with torch.no_grad():
    model.eval()
    testref_metric, _ = evaluate(model, [testref_loader],
                                 testref_targets, device)

print(testref_metric['comma_precision'])
print(testref_metric['comma_recall'])
print(testref_metric['comma'])
print(testref_metric['period_precision'])
print(testref_metric['period_recall'])
print(testref_metric['period'])
print(testref_metric['question_precision'])
print(testref_metric['question_recall'])
print(testref_metric['question'])
print(testref_metric['total_precision'])
print(testref_metric['total_recall'])
print(testref_metric['total'])