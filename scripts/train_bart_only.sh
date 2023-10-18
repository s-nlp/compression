export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

for random in 585534 #585534 297104 743580 814084 39512
do
    python ./xsum_trainer.py \
    --model_name_or_path "facebook/bart-base" \
    --tokenizer_name "facebook/bart-base"  \
    --run_name bart-xsum-wttm-dt-$ranks \
    --dataset_name xsum \
    --dataset_config "3.0.0" \
    --do_train \
    --do_predict \
    --max_train_samples 1000 \
    --max_eval_samples 100 \
    --max_predict_samples 100 \
    --comp_func "none" \
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
    --output_dir ./experiments_BART_FWSVD_DT_predict_2/BART_XSUM_FWSVD_$random
done