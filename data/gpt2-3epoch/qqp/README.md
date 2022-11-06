---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
- f1
model-index:
- name: qqp
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE QQP
      type: glue
      args: qqp
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8948800395745733
    - name: F1
      type: f1
      value: 0.8610929533272322
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qqp

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the GLUE QQP dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2570
- Accuracy: 0.8949
- F1: 0.8611
- Combined Score: 0.8780

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

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|:--------------:|
| No log        | 1.0   | 5054  | 0.2815          | 0.8763   | 0.8399 | 0.8581         |
| No log        | 2.0   | 10108 | 0.2526          | 0.8909   | 0.8566 | 0.8738         |
| No log        | 3.0   | 15162 | 0.2570          | 0.8949   | 0.8611 | 0.8780         |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1
