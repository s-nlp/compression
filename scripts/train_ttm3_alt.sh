for model in "bert-base-uncased"
do
	for ranks in 10 34 46 56 64 71 78 84 90 96 105 115
	do
		for random in 297104
		do
			CUDA_VISIBLE_DEVICES=0 python glue_trainer.py \
				--model_name_or_path $model  \
				--run_name $model-ttm_ffn_w-$ranks-$random \
				--comp_func 'ttm_ffn_w'  --rank $ranks \
				--save_strategy "no" \
				--logging_strategy "no" \
				--do_bench --bench_on_eval --bench_on_train \
				--max_bench_iter 1 \
				--tt_input_dims 8 12 8 \
				--tt_output_dims 12 16 16 \
				--tt_ranks $ranks $ranks \
				--batch_sizes 1 16 32 \
				--sequence_lengths 128 \
				--max_seq_length 128 \
				--per_device_train_batch_size 32 \
				--per_device_eval_batch_size 128 \
				--learning_rate 5e-5 \
				--num_train_epochs 2 \
				--evaluation_strategy 'epoch' \
				--seed $random \
				--resume_from_checkpoint /path/to/pretrained/models  \
				--output_dir './ttm3/' \
                --do_train --do_eval --double_train  \
				--overwrite_output_dir
		done
	done
done

#10 60 110 160 210 260 310 360 410 460 510
#10 34 46 56 64 71 78 84 90 96 105 115
#10 76 128 192 250 310 372 432 492 552 612