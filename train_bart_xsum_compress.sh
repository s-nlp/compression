export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

for ranks in 10 #210 460 84 90 105 #10 34 46 56 64 71 78 84 90 96 105
	do
		for random in 585534 #585534 297104 743580 814084 39512
		do
			python ./xsum_trainer.py \
			--model_name_or_path ./experiments_BART_FWSVD_DT_predict_2/BART_XSUM_FWSVD_$random \
			--tokenizer_name ./experiments_BART_FWSVD_DT_predict_2/BART_XSUM_FWSVD_$random  \
			--run_name bart-xsum-wttm-dt-$ranks \
			--dataset_name xsum \
			--dataset_config "3.0.0" \
            --do_train \
			--do_predict \
            --max_train_samples 1000 \
            --max_eval_samples 100 \
            --max_predict_samples 100 \
            --tt_input_dims 8 12 8 \
            --tt_output_dims 12 16 16 \
            --tt_ranks $ranks $ranks \
            --comp_func "ttm_ffn_bart" \
			--rank $ranks \
            --predict_with_generate \
			--seed $random \
			--evaluation_strategy "epoch" \
			--max_source_length 512 \
			--per_device_train_batch_size 16 \
			--per_device_eval_batch_size 32 \
			--learning_rate 3e-5 \
			--num_train_epochs 1 \
			--overwrite_output_dir \
			--save_strategy "no" \
			--source_prefix "" \
			--output_dir ./experiments_BART_TTM/BART_XSUM_dt_ttm_ffn_bart-$ranks-$random
		done
	done