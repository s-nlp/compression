---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
- f1
model-index:
- name: mrpc
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE MRPC
      type: glue
      args: mrpc
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8137254901960784
    - name: F1
      type: f1
      value: 0.8733333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mrpc

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4516
- Accuracy: 0.8137
- F1: 0.8733
- Combined Score: 0.8435

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
- num_epochs: 1.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:--------------:|
| No log        | 1.0   | 51   | 0.4516          | 0.8137   | 0.8733 | 0.8435         |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1
