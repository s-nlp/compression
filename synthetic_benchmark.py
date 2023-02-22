from utils.hf_bench.benchmark import PyTorchBenchmark
from utils.hf_bench.benchmark_args import PyTorchBenchmarkArguments
from exps.models import MODEL_NAMES

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime

#Dont work with HF argparser
#https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py
def synth_bench():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",default="bert-base-uncased", help="model from HF")
    parser.add_argument("--run_name")
    parser.add_argument("--output_dir")
    parser.add_argument('--batch_sizes', type=int, nargs="+", default=(1,64), help='batch sizes for bench')
    parser.add_argument('--sequence_lengths', type=int, nargs="+", default=(16,128), help='sequence sizes for bench')
    parser.add_argument('--max_bench_iter', help='Seconds for warmup blocked_autorange benchmark', type=int, default=1)
    parser.add_argument("--do_bench", help="do bench",action="store_true")
    parser.add_argument("--bench_on_train", help="do bench train",action="store_true")
    parser.add_argument("--bench_on_eval", help="do bench test",action="store_true")
    parser.add_argument("--comp_func", help="Compression function to be used on finetuned model",default="none")  
    parser.add_argument("--rank", type=int, default=100, help="rank to compress")
    parser.add_argument("--tt_ranks", type=int, nargs="+", default=(10,10,10), help="ranks of TT decomposition") 
    parser.add_argument("--tt_input_dims", type=int, nargs="+", default=(4,6,8,4), help="input_dims in TTMatrix") 
    parser.add_argument("--tt_output_dims", type=int, nargs="+", default=(8,8,6,8), help="output_dims in TTMatrix")
    args_bench, unknown = parser.parse_known_args()
    print(args_bench)
    

    if args_bench.do_bench:
        save_to = args_bench.output_dir + args_bench.run_name+r'/'
        isExist = os.path.exists(save_to)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(save_to)

        args_full = PyTorchBenchmarkArguments(models=[args_bench.model_name_or_path], batch_sizes=args_bench.batch_sizes, 
                                 sequence_lengths=args_bench.sequence_lengths,
                                 training=args_bench.bench_on_train, inference=args_bench.bench_on_eval, cuda=True,
                                 multi_process=True, verbose=False, trace_memory_line_by_line=False, repeat=args_bench.max_bench_iter,
                                 inference_time_csv_file=save_to+r'inference_time.csv',
                                 inference_memory_csv_file=save_to+r'inference_memory.csv',
                                 train_time_csv_file=save_to+r'train_time.csv',
                                 train_memory_csv_file=save_to+r'train_memory.csv',
                                 env_info_csv_file=save_to+r'os_info.csv',
                                 save_to_csv=True, env_print=False, exp_name=args_bench.comp_func, rank=args_bench.rank,
                                 tt_ranks=args_bench.tt_ranks,
                                 tt_input_dims=args_bench.tt_input_dims, tt_output_dims=args_bench.tt_output_dims
                                 )
        benchmark = PyTorchBenchmark(args_full)
        benchmark.run()


if __name__ == '__main__':
    synth_bench()