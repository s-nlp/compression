for method in "row-sum-weighted-svd"
do
    for rank in 450 600
    do
        echo "doing $method rank $rank "
        CUDA_VISIBLE_DEVICES=5 python bench_glue_AIO.py \
        --model_name_or_path bert-base-uncased \
        --run_name bert-base-uncased-3epoch_svd_${rank}_full --exp_name 'fwsvd_ffn' --rank $rank \
        --use_baseline \
        --low_rank_method ${method} \
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
        --seed 44 \
        --output_dir ./nla_proj_results/ \
        --overwrite_output_dir \
        --do_train --do_eval \
    #	--fp16 --fp16_full_eval 
    done
done
