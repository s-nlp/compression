for rank in 10 20 40 80
do
   echo "doing $model"
   CUDA_VISIBLE_DEVICES=3,4,5 python bench_glue_AIO.py \
	--model_name_or_path bert-base-uncased \
	--run_name ttm-checkpointed-einsum-forward-bert-base-uncased-rank-$rank-rerun --exp_name 'ttm_ffn' \
    --rank $rank \
	--save_strategy "epoch" \
	--logging_strategy no --save_strategy no \
	--do_bench --bench_on_eval --bench_on_train \
	--batch_sizes [1,16] \
	--sequence_lengths [128] \
	--max_bench_iter 1 \
	--max_seq_length 128 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 1 \
	--learning_rate 5e-5 \
	--num_train_epochs 3 \
	--evaluation_strategy 'epoch' \
	--seed 42 \
	--output_dir ./ttm_proj_results/ \
	--overwrite_output_dir \
	--do_train --do_eval \
#	--fp16 --fp16_full_eval 

done
