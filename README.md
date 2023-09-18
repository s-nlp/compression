# compression
Project on transformers compression


## Quick Start

This benchmark is mostly based on the following code [HF-run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py). So most of args should work for this script too. 

Benchmark based on GLUE which is made up of a total of 9 different tasks. Here is how to run the script:

```bash
model = "bert-base-uncased"
random = 814084
python bench_glue_AIO.py \
				--model_name_or_path $model  \
				--run_name $model-full-$random \
				--comp_func 'none'
				--save_strategy "epoch" \
				--logging_strategy no \
				--do_bench --bench_on_eval --bench_on_train \
				--max_bench_iter 1 \
				--batch_sizes 1 16 32 \
				--sequence_lengths 128 \
				--max_seq_length 128 \
				--per_device_train_batch_size 32 \
				--per_device_eval_batch_size 128 \
				--learning_rate 5e-5 \
				--num_train_epochs 2 \
				--evaluation_strategy 'epoch' \
				--seed $random \
				--output_dir ./data_eval/ \
				--overwrite_output_dir \
				--do_train --do_eval 
```
This script will train and eval bert-base model for GLUE, and then output the results with GPU and CPU utilization.

for TTM
```bash
for model in "bert-base-uncased"
do
	for ranks in 10 60 110 
	do
		for random in 39512 
		do
			CUDA_VISIBLE_DEVICES=0 python glue_trainer.py \
				--model_name_or_path $model  \
				--run_name $model-svd_ffn_w_T-$ranks-$random \
				--comp_func 'ttm_ffn' --rank $ranks \
				--save_strategy "no" \
				--logging_strategy "no" \
				--do_bench --bench_on_eval --bench_on_train \
				--max_bench_iter 1 \
				--batch_sizes 1 16 32 \
				--sequence_lengths 128 \
				--max_seq_length 128 \
				--per_device_train_batch_size 32 \
				--per_device_eval_batch_size 128 \
				--learning_rate 5e-5 \
				--tt_ranks $ranks $ranks $ranks \
				--tt_input_dims 12 2 2 16 \
				--tt_output_dims 32 3 2 16 \
				--num_train_epochs 2 \
				--evaluation_strategy 'epoch' \
				--seed $random \
				--output_dir './bert-base-uncased-ttm_ffn/' \
				--do_train --do_eval \
				--overwrite_output_dir
		done
	done
done
```

| model                  | score    | size(MB) | size(M param) | SPS       | train speed | inf speed | used_cpu | used_cpu_mem | used_gpu | used_gpu_mem |
|------------------------|----------|----------|---------------|-----------|-------------|-----------|----------|--------------|----------|--------------|
| bert | 0.79508  | 417.6553 | 109.483778    | 513.44118 | 0.21948     | 0.078     | 35.40032 | 2644.8       | 44.9     | 1599         |
| **stsb**               | **cola** | **mnli** | **mrpc**      | **qnli**  | **qqp**     | **rte**   | **sst2** | **wnli**     |          |              |
| 0.88816                | 0.57574  | 0.84928  | 0.90352       | 0.91338   | 0.87682     | 0.67508   | 0.92432  | 0.5493       |          |              |

To train model with compression function you can change script to:
```bash
	--comp_func 'svd_ffn_w'  \
    --rank 210 \
    --double_train \
```
where __svd_ffn_w__ is SVD (all models available at /exps/models.py) and __rank__ is SVD rank. __double_train__ is the function for additional training after svd compression, in some cases gives better results.


## Details

For evaluate gpu/cpu metrics we use pynvml library. Model only evaluate during evalatuation process.

You can also get scores without training model by directly using __synthetic_benchmark.py__ script.
```
synthetic_benchmark.py \
				--model_name_or_path 'gpt2'  \
				--run_name 'gpt2-full' \
				--comp_func 'none' \
				--do_bench --bench_on_eval --bench_on_train \
				--max_bench_iter 1 \
				--batch_sizes 1 16 32 \
				--sequence_lengths 128 \
				--max_seq_length 128 \
				--seed 42 \
				--output_dir ./data_eval_gpt/
```

Data folder contains evaluation results for gpt2/bert-base-cased/distilbert/pruned-bert on 3090. This can be used as example.

Due to the specifics of gpus, the benchmark should only be performed on immutable environment. The same script running on different gpus will give different results. 

## Experiments

exps folder contains various model compression experiments using ```--comp_func``` :

1. head pruning, based on [16vs1head paper](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology) ```random_head```

2. Standart SVD ```our_ffn```
3. Weighted SVD ```svd_ffn_w_T or svd_ffn_w```
4. TTM ```ttm_ffn```

Additional models can be found in ./exps/models.py

## to-do

1. RussianSuperGlue, BigBench
2. Better initialization
3. Best-of-5 evaluation
