# task_names=(boolq cb copa multirc record wic wsc)
# task_folders=(BoolQ CB COPA MultiRC ReCoRD WiC WSC)

task_names=(multirc)
task_folders=(MultiRC)

export CUDA_VISIBLE_DEVICES=2
export NVIDIA_VISIBLE_DEVICES=2

for ((i = 0; i < 1; i++)); do
    echo "task_name: ${task_names[i]}, task_folder: data/${task_folders[i]}"
    python superglue_trainer.py \
        --data_dir data/${task_folders[i]} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name ${task_names[i]} \
        --output_dir logs/ \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --seed 42 \
        --output_dir superglue_models/bert-base-uncased/${task_names[i]} \
        --overwrite_output_dir \
        --log_evaluate_during_training \
        --save_only_best \
        --evaluate_test #
done
