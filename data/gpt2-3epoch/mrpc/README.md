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
      value: 0.7720588235294118
    - name: F1
      type: f1
      value: 0.8467874794069192
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# mrpc

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4841
- Accuracy: 0.7721
- F1: 0.8468
- Combined Score: 0.8094

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

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:--------------:|
| No log        | 1.0   | 51   | 0.5580          | 0.7034   | 0.8169 | 0.7602         |
| No log        | 2.0   | 102  | 0.5087          | 0.7426   | 0.8377 | 0.7902         |
| No log        | 3.0   | 153  | 0.4841          | 0.7721   | 0.8468 | 0.8094         |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1