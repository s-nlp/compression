# task_names=(lidirus rcb danetqa parus rwsd terra rcb rucos russe)

task_names=(rcb)

export CUDA_VISIBLE_DEVICES=1
export NVIDIA_VISIBLE_DEVICES=1
export WANDB_DISABLED=True

for ((i = 0; i < 1; i++)); do
    python run_trainer.py \
        --model_name_or_path DeepPavlov/rubert-base-cased-conversational \
        --task_name ${task_names[i]} \
        --do_cofi \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 5 \
        --gradient_accumulation_steps 1 \
        --seed 42 \
        --output_dir superglue_models/CoFi \
        --overwrite_output_dir \
        --log_evaluate_during_training \
        --save_only_best \
        --logging_steps 100
done
