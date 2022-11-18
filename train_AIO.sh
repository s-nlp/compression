for model in bert-base-uncased
do
   echo "doing $model "
   CUDA_VISIBLE_DEVICES=0 python bench_glue_AIO.py \
	--model_name_or_path $model \
	--run_name $model-bert-3epoch_our_fnn600 --exp_name 'our_fnn' --rank 600 \
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
	--seed 42 \
	--output_dir ./data_eval_v2/ \
	--overwrite_output_dir \
	--do_train --do_eval \
#	--fp16 --fp16_full_eval 

done