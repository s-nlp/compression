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
model-index:
- name: wnli
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE WNLI
      type: glue
      args: wnli
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.4647887323943662
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wnli

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the GLUE WNLI dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6952
- Accuracy: 0.4648

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
| No log        | 1.0   | 9    | 0.6903          | 0.5634   |
| No log        | 2.0   | 18   | 0.6942          | 0.5493   |
| No log        | 3.0   | 27   | 0.6952          | 0.4648   |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.6.1
- Tokenizers 0.13.1