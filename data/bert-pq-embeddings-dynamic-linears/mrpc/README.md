---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: mrpc
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mrpc

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.4584
- eval_accuracy: 0.7990
- eval_f1: 0.8493
- eval_combined_score: 0.8241
- eval_runtime: 11.5361
- eval_samples_per_second: 35.367
- eval_steps_per_second: 0.607
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 64
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.8.0
- Tokenizers 0.12.1
