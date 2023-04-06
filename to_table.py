import argparse
import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

TASKS_GLUE = [
    'stsb',
    'cola',
    'mnli',
    'mrpc',
    'qnli',
    'qqp',
    'rte',
    'sst2',
    'wnli',
    ]

TASKS_SUPERGLUE = [
    'copa',
    'multirc',
    'rte',
    'boolq',
    'cb',
    'wic',
    'wsc',
    'record'
    ]


@dataclass
class Score:
    first: float
    second: float = None

TASK_METRICS = {
    'stsb': 'eval_pearson',
    'cola': 'eval_matthews_correlation',
    'mnli': 'eval_accuracy',
    'mrpc': ('eval_f1','eval_accuracy'),
    'qnli': 'eval_accuracy',
    'qqp': ('eval_f1','eval_accuracy'),
    'rte': 'eval_accuracy',
    'sst2': 'eval_accuracy',
    'wnli': 'eval_accuracy',

    'copa': 'eval_accuracy',
    'multirc': 'eval_accuracy',
    'boolq': 'eval_accuracy',
    'cb': 'eval_accuracy',
    'wic': 'eval_accuracy',
    'wsc': 'eval_accuracy',
    'record': 'eval_accuracy',
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

def OverallTable(path_to, file_to, bench_name):
    if bench_name == 'glue':
        TASKS = TASKS_GLUE
    else:
        TASKS = TASKS_SUPERGLUE

    jiant_task_scores = []

    MODELS = os.listdir(path_to)

    for model in MODELS:
        csv_iter = "inference_memory.csv"
        path_csv = join(path_to, model, csv_iter)
        if not exists(path_csv):
            continue
        df_inf = pd.read_csv(path_csv)
        data_inf = df_inf.iloc[0].squeeze()

        csv_train_time = "train_time.csv"
        path_csv_train_time = join(path_to, model, csv_train_time)
        df_train_time = pd.read_csv(path_csv_train_time)
        data_train_t = df_train_time.iloc[-1].squeeze()

        csv_inf_time = "inference_time.csv"
        path_csv_inf_time = join(path_to, model, csv_inf_time)
        df_inf_time = pd.read_csv(path_csv_inf_time)
        data_inf_t = df_inf_time.iloc[-1].squeeze()

        for task in TASKS:
            json_iter = "bench_0_results.json"
            path = join(path_to, model, task, json_iter)
            

            if not exists(path):
                continue

            data = load_json(path)
            score = metrics_score(task, data)
            speed = data['eval_samples_per_second']
            try:
                size_of = data['size_of'] / 1024 /1024
            except:
                size_of = 0

            try:
                param_of = data['param_of']
            except:
                param_of = 0

            used_cpu = data_inf['used_cpu']
            used_cpumem = data_inf['used_cpumem']
            used_gpu = data_inf['used_gpu']
            used_gpumem = data_inf['used_gpumem']

            t_train = data_train_t['result']
            t_inf = data_inf_t['result']

        
            jiant_task_scores.append([model, task, json_iter, score.first, size_of, param_of, t_train, t_inf,
                used_cpu, used_cpumem, used_gpu, used_gpumem])

    table = pd.DataFrame(jiant_task_scores, columns=['model', 'task', 'json_iter', 'score',
                'size(MB)','size(param)', 'train speed', 'inf speed', 'used_cpu', 'used_cpu_mem', 'used_gpu', 'used_gpu_mem'])

    table_new = table.drop(['json_iter'], axis=1).groupby(['model','task']).agg(np.mean).reset_index()
    print(table_new)
    left_table = table_new.pivot(index='model', columns='task', values='score').copy()

    left_table.columns.name = None
    left_table.index.name = None
    left_table = left_table.reindex(index=MODELS, columns=TASKS)

    table_res = table.drop(['task','json_iter'], axis=1).groupby('model').agg(np.mean)
    table_ans = table_res.join(left_table)
    round(table_ans,4).to_csv(file_to)
    round(table_ans,4).to_csv(sys.stdout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",help="path to generated GLUE folder")
    parser.add_argument("--output",default="results.csv", help="write report to file")
    parser.add_argument("--dataset_name",default="glue", help="GLUE or SuperGLUE")
    args = parser.parse_args(sys.argv[1:])
    print(args.output)
    print(args.path)
    OverallTable(args.path, args.output, args.dataset_name)
