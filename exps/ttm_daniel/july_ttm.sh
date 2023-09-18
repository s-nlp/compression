for model in "bert-base-uncased"
do
	for ranks in 78 #10 34 46 56 64 71 78 84 90 96 105 115
	do
		for random in 743580 814084 39512 #297104 585534 743580 814084 39512
		do
			CUDA_VISIBLE_DEVICES=3 python glue_trainer.py \
				--model_name_or_path $model  \
				--run_name $model-alt_ttm_ffn_3-$ranks-$random \
				--comp_func 'ttm_ffn_alt'  --rank $ranks \
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
				--per_device_eval_batch_size 512 \
				--learning_rate 5e-5 \
				--num_train_epochs 2 \
				--evaluation_strategy 'epoch' \
				--seed $random \
				--resume_from_checkpoint /home/pletenev/compression_final_v2/data_bert_random_v2_512/bert-base-uncased-clean-$random  \
				--output_dir 'daniil_original/' \
                                --do_train --do_eval \
				--overwrite_output_dir
		done
	done
done
