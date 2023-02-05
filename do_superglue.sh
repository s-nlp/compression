# task_names=(boolq cb copa multirc record wic wsc)
# task_folders=(BoolQ CB COPA MultiRC ReCoRD WiC WSC)

task_names=(boolq cb copa rte wic wsc multirc record)
task_folders=(BoolQ CB COPA RTE WiC WSC MultiRC ReCoRD)

export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

for ((i = 0; i < 1; i++)); do
        echo "task_name: ${task_names[i]}, task_folder: data/${task_folders[i]}"
        python superglue_trainer.py \
                --data_dir data/${task_folders[i]} \
                --model_type bert \
                --model_name_or_path bert-base-uncased \
                --task_name ${task_names[i]} \
                --do_train \
                --double_train \
                --comp_func 'our_ffn' \
                --rank 310 \
                --do_eval \
                --per_gpu_train_batch_size 16 \
                --per_gpu_eval_batch_size 16 \
                --learning_rate 5e-5 \
                --num_train_epochs 3 \
                --gradient_accumulation_steps 1 \
                --seed 42 \
                --output_dir superglue_models/ \
                --overwrite_output_dir \
                --log_evaluate_during_training \
                --save_only_best
done
