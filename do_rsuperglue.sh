task_names=(muserc)

export CUDA_VISIBLE_DEVICES=2
export NVIDIA_VISIBLE_DEVICES=2

for ((i = 0; i < 1; i++)); do
    python run_trainer.py \
        --model_type bert \
        --model_name_or_path cointegrated/rubert-tiny2 \
        --task_name ${task_names[i]} \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 1 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --gradient_accumulation_steps 1 \
        --seed 42 \
        --output_dir superglue_models/ \
        --overwrite_output_dir \
        --log_evaluate_during_training \
        --save_only_best \
        --logging_steps 500
done
