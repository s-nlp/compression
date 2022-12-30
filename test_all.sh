for model in "bert-base-uncased" #"facebook/bart-base" #bert-base-uncased "facebook/bart-base" #
do
	for ranks in 210
	do
		for random in 814084
		do
			python bench_glue_AIO.py \
				--model_name_or_path $model  \
				--run_name $model-svd-fnn-w-$ranks-$random \
				--comp_func 'none'  --rank $ranks\
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
				--output_dir ./data_eval_weight_ALL_7/ \
				--overwrite_output_dir \
				--do_train --do_eval 
		done
	done
done
