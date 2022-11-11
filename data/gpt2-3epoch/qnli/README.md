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
model-index:
- name: qnli
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE QNLI
      type: glue
      args: qnli
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8854109463664653
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qnli

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the GLUE QNLI dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2872
- Accuracy: 0.8854

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

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 1455 | 0.3328          | 0.8556   |
| No log        | 2.0   | 2910 | 0.3072          | 0.8667   |
| No log        | 3.0   | 4365 | 0.2872          | 0.8854   |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1