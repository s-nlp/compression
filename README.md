# compression
Project on transformers compression


## Quick Start

This benchmark is mostly based on the following code [HF/run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) and [HF/benchmark.py](https://github.com/huggingface/transformers/tree/ebfd7229d2e49ad038115d0edc43c3d12158a17f/src/transformers/benchmark) So most of args should work for this script too. 

Benchmark based on GLUE which is made up of a total of 9 different tasks. Here is how to run the script:

```bash
model="distilgpt2"
python bench_glue_AIO.py \
	--model_name_or_path $model \
	--run_name $model-3epoch \
	--save_strategy "epoch" \
	--logging_strategy no --save_strategy no \
	--do_bench --bench_on_eval --bench_on_train \
	--batch_sizes [1,16,32] \
	--sequence_lengths [128] \
	--max_bench_iter 1 \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 1 \
	--learning_rate 5e-5 \
	--num_train_epochs 3 \
	--evaluation_strategy 'epoch' \
	--seed 1337 \
	--output_dir ./data_eval/ \
	--overwrite_output_dir \
	--do_train --do_eval 
```
This script will train and eval distil-gpt2 model for GLUE, and then output the results with GPU and CPU utilization:

```
model,score,size(MB),SPS,used_cpu,used_cpu_mem,used_gpu,used_gpu_mem,stsb,cola,mnli,mrpc,qnli,qqp,rte,sst2,wnli
distilgpt2-3epoch,0.703,318.478,157.787,0.39,2263.0,9.0,1699.438,0.807,0.268,0.803,0.864,0.86,0.85,0.65,0.903,0.324
```

## Details

For evaluate gpu/cpu metrics we use pynvml library. Model only evaluate during evalatuation process.

You can also get scores without training model by directly using bench.py script.
```
python bench.py --path ./data/ --file ./ans.csv
```

data_eval folder contains evaluation results for distil-gpt2/distilbert/distilroberta on RTX 3090. This can be used as example.

Due to the specifics of gpus, the benchmark should only be performed on immutable environment. The same script running on different gpus will give different results. 

## Experiments

exps folder contains various model compression experiments:

1. head pruning, based on [16vs1head paper](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology)

2. more to come

## to-do

1. SuperGlue, RussianSuperGlue, BigBench
2. Better initialization
3. Best-of-5 evaluation
