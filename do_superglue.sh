# task_names=(boolq cb copa multirc record wic wsc)
# task_folders=(BoolQ CB COPA MultiRC ReCoRD WiC WSC)

task_names=(multirc)
task_folders=(MultiRC)

export CUDA_VISIBLE_DEVICES=1
export NVIDIA_VISIBLE_DEVICES=1

for ((i = 0; i < 1; i++)); do
        echo "task_name: ${task_names[i]}, task_folder: data/${task_folders[i]}"
        python superglue_trainer.py \
                --data_dir data/${task_folders[i]} \
                --model_type roberta \
                --model_name_or_path roberta-base \
                --task_name ${task_names[i]} \
                --do_train \
                --do_eval \
                \
                \
                \
                --per_gpu_train_batch_size 16 \
                --per_gpu_eval_batch_size 1 \
                --learning_rate 5e-6 \
                --num_train_epochs 3 \
                --gradient_accumulation_steps 1 \
                --seed 42 \
                --output_dir superglue_models/ \
                --overwrite_output_dir \
                --log_evaluate_during_training \
                --logging_steps 500 \
                --save_only_best # --double_train \
        # --comp_func 'our_ffn' \
        # --rank 310 \
done
