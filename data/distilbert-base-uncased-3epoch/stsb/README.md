---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: stsb
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE STSB
      type: glue
      args: stsb
    metrics:
    - name: Spearmanr
      type: spearmanr
      value: 0.8648629466161909
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stsb

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the GLUE STSB dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5666
- Pearson: 0.8682
- Spearmanr: 0.8649
- Combined Score: 0.8665

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
- train_batch_size: 72
- eval_batch_size: 1
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Pearson | Spearmanr | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:---------:|:--------------:|
| No log        | 1.0   | 80   | 0.5834          | 0.8607  | 0.8582    | 0.8595         |
| No log        | 2.0   | 160  | 0.5565          | 0.8695  | 0.8665    | 0.8680         |
| No log        | 3.0   | 240  | 0.5666          | 0.8682  | 0.8649    | 0.8665         |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1
