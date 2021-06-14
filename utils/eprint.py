from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def f1_printer(prefix, f1_scores):
    eprint("============{}============".format(prefix))
    eprint("F1 Total:    {}".format(f1_scores['total']))
    eprint("F1 Comma:    {}".format(f1_scores['comma']))
    eprint("F1 Period:   {}".format(f1_scores['period']))
    eprint("F1 Question: {}".format(f1_scores['question']))
