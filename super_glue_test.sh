for model in "bert-base-uncased" #"facebook/bart-base" #bert-base-uncased "facebook/bart-base" #
do
	for ranks in 410 #10 60 110 160 210 260 310 360 410 460 510 #560 610 660 710 760
	do
		for random in 39512 #297104 585534 743580 814084
		do
			CUDA_VISIBLE_DEVICES=0 python superglue_trainer.py \
				--model_name_or_path $model  \
				--run_name $model-model-$ranks-$random \
				--comp_func 'none'  --rank $ranks \
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
				--num_train_epochs 2 \
				--evaluation_strategy 'epoch' \
				--seed $random \
				--output_dir './simple_superglue/' \
				--do_eval \
				--overwrite_output_dir
		done
	done
done