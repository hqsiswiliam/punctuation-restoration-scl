import csv


def write_calc_metric_result(result, filename, best_epoch=''):
    asr = result['asr']
    ref = result['ref']
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([asr['f1']['total'], ref['f1']['total'], asr, ref, best_epoch])
