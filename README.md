# compression
Project on transformers compression


## Quick Start

This benchmark is mostly based on the following code [HF-run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py). So most of args should work for this script too. 

Benchmark based on GLUE which is made up of a total of 9 different tasks. Here is how to run the script:

```bash
python bench_glue_AIO.py \
    --model_name_or_path gpt2 \
    --run_name gpt2-3epoch \
    --task_name sst2 \
    --save_strategy "epoch" \
    --do_train \
    --logging_strategy no \
    --save_strategy no \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 72 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy 'epoch' \
    --seed 1337 \
    --output_dir ./data/ \
    --overwrite_output_dir
```
This script will train and eval gpt2 model for GLUE, and then output the results with GPU and CPU utilization.

```
model,score,size(MB),SPS,used_cpu,used_cpu_mem,used_gpu,used_gpu_mem,stsb,cola,mnli,mrpc,qnli,qqp,rte,sst2,wnli
gpt2-3epoch,0.731,486.706,113.125,0.42,3491.0,36.0,15473.438,0.845,0.318,0.819,0.847,0.885,0.861,0.632,0.909,0.465
```

## Details

For evaluate gpu/cpu metrics we use pynvml library. Model only evaluate during evalatuation process.

You can also get scores without training model by directly using bench.py script.
```
python bench.py --path ./data/ --file ./ans.csv
```

Data folder contains evaluation results for gpt2/bert-base-cased/distilbert/pruned-bert on 3090. This can be used as example.

Due to the specifics of gpus, the benchmark should only be performed on immutable environment. The same script running on different gpus will give different results. 

## Experiments

exps folder contains various model compression experiments:

1. head pruning, based on [16vs1head paper](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology)

2. more to come

## to-do

1. SuperGlue, RussianSuperGlue, BigBench
2. Better initialization
3. Best-of-5 evaluation
