python synthetic_benchmark.py \
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