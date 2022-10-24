import argparse
import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

STSB = 'stsb' 
COLA = 'cola' 
MNLI = 'mnli' 
MRPC = 'mrpc' 
QNLI = 'qnli' 
QQP = 'qqp' 
RTE = 'rte' 
SST2 ='sst2' 
WNLI ='wnli'

TASKS = [
    STSB,
    COLA,
    MNLI,
    MRPC,
    QNLI,
    QQP,
    RTE,
    SST2,
    WNLI,
]
TASK_TITLES = {
    STSB: 'STSB',
    COLA: 'COLA',
    MNLI: 'MNLI',
    MRPC: 'MRPC',
    QNLI: 'QNLI',
    QQP: 'QQP',
    RTE: 'RTE',
    SST2: 'SST2',
    WNLI: 'WNLI',
}

@dataclass
class Score:
    first: float
    second: float = None

TASK_METRICS = {
    STSB: 'eval_pearson',
    COLA: 'eval_matthews_correlation',
    MNLI: 'eval_accuracy',
    MRPC: ('eval_f1','eval_accuracy'),
    QNLI: 'eval_accuracy',
    QQP: ('eval_f1','eval_accuracy'),
    RTE: 'eval_accuracy',
    SST2: 'eval_accuracy',
    WNLI: 'eval_accuracy',
}

from os.path import (
    join,
    exists
)
import json

def load_text(path):
    with open(path) as file:
        return file.read()

def load_json(path):
    text = load_text(path)
    return json.loads(text)

def metrics_score(task, metrics):
    key = TASK_METRICS[task]
    if isinstance(key, tuple):
        first, second = key
        first, second = metrics[first], metrics[second]
    else:
        first = metrics[key]
        second = None
    return Score(first, second)

def main(args):
    parser = argparse.ArgumentParser(prog='bench.py')
    parser.add_argument("--file",default="results.csv", help="write report to FILE")
    parser.add_argument("--path",help="don't print status messages to hfgstdout")
    args = parser.parse_args(args)
    print(args.file)
    print(args.path)

    jiant_task_scores = []
    JSON = r'eval_results.json'
    #JIANT_EVAL_DIR = r'/content/data/'
    #MODELS = ['bert_uncased_pruned_1','bert_uncased_1','bert_base']
    MODELS = os.listdir(args.path)
    #MODELS = ['bert_base_1','robert_base_1']
    for model in MODELS:
        for task in TASKS:
            path = join(args.path, model, task, JSON)
            if not exists(path):
                continue

            data = load_json(path)
            score = metrics_score(task, data)
            speed = data['eval_samples_per_second']
            size_of = data['size_of'] / 1024 /1024
            used_cpu = data['used_cpu']
            used_cpumem = data['used_cpumem']
            used_gpu = data['used_gpu']
            used_gpumem = data['used_gpumem']
            
            jiant_task_scores.append([model, task, score.first, size_of, speed, 
                used_cpu, used_cpumem, used_gpu, used_gpumem])

    table = pd.DataFrame(jiant_task_scores, columns=['model', 'task', 'score',
                'size(MB)','SPS', 'used_cpu', 'used_cpu_mem', 'used_gpu', 'used_gpu_mem'])
    left_table = table.pivot(index='model', columns='task', values='score').copy()
    left_table.columns.name = None
    left_table.index.name = None
    left_table = left_table.reindex(index=MODELS, columns=TASKS)

    table_res = table.drop(['task'], axis=1).groupby('model').agg(np.mean)
    table_ans = table_res.join(left_table)
    round(table_ans,3).to_csv(args.file)
    round(table_ans,3).to_csv(sys.stdout)

if __name__ == '__main__':
    main(sys.argv[1:])
