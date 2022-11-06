---
language:
- en
license: mit
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
      value: 0.842858632517954
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stsb

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the GLUE STSB dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6759
- Pearson: 0.8455
- Spearmanr: 0.8429
- Combined Score: 0.8442

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
| No log        | 1.0   | 80   | 0.8483          | 0.7991  | 0.7974    | 0.7982         |
| No log        | 2.0   | 160  | 0.7659          | 0.8374  | 0.8376    | 0.8375         |
| No log        | 3.0   | 240  | 0.6759          | 0.8455  | 0.8429    | 0.8442         |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1
