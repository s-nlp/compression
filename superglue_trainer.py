#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

from utils.hf_bench.benchmark import PyTorchBenchmark
from utils.hf_bench.benchmark_args import PyTorchBenchmarkArguments
from exps.models import MODEL_NAMES

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime

import datasets
import numpy as np
from datasets import load_dataset
from tasksource import load_task

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    GlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.utils import add_start_docstrings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from to_table import OverallTable
from synthetic_benchmark import synth_bench

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "multirc": ("sentence1", None),
    "boolq": ("sentence1", None),
    "rte": ("premise", "hypothesis"),
    "cb": ("sentence1", "sentence2"),
    "wic": ("sentence1", "sentence2"),
    "axg": ("sentence1", "sentence2"),
    "copa": ("not", "using"),
    "wsc": ("text", None),
}

logger = logging.getLogger(__name__)

class CustomTrainerBert(Trainer):
    def make_grad_bank(self, model):
        self.grad_bank_out_epoch = []
        self.grad_bank_int_epoch = []
        self.grad_bank_out = {i:torch.zeros((model.config.hidden_size,model.config.intermediate_size)) for i in range(model.config.num_hidden_layers)}
        self.grad_bank_int = {i:torch.zeros((model.config.intermediate_size,model.config.hidden_size)) for i in range(model.config.num_hidden_layers)}
        self.grad_bank_out_2 = {i:torch.zeros((model.config.hidden_size,model.config.intermediate_size)) for i in range(model.config.num_hidden_layers)}
        self.grad_bank_int_2 = {i:torch.zeros((model.config.intermediate_size,model.config.hidden_size)) for i in range(model.config.num_hidden_layers)}
        self.avg_counter = 0
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        loss.backward()

        for layer in range(model.config.num_hidden_layers):

            self.grad_bank_out[layer] += model.bert.encoder.layer[layer].output.dense.weight.grad.detach().cpu() #(model.bert.encoder.layer[layer].intermediate.dense.weight.grad.detach().cpu() **2).sum(0)
            self.grad_bank_int[layer] += model.bert.encoder.layer[layer].intermediate.dense.weight.grad.detach().cpu() #(model.bert.encoder.layer[layer].output.dense.weight.grad.detach().cpu() **2).sum(0)
        
            self.grad_bank_out_2[layer] += model.bert.encoder.layer[layer].output.dense.weight.grad.detach().cpu()**2 #(model.bert.encoder.layer[layer].intermediate.dense.weight.grad.detach().cpu() **2).sum(0)
            self.grad_bank_int_2[layer] += model.bert.encoder.layer[layer].intermediate.dense.weight.grad.detach().cpu()**2 #(model.bert.encoder.layer[layer].output.dense.weight.grad.detach().cpu() **2).sum(0)
        self.avg_counter+=1
        return loss.detach()

class ProcessorWSC:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize(self, tokens, ss, se, os, oe):
        sents = []

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if i_t == ss:
                new_ss = len(sents)
                tokens_wordpiece = ['@'] + tokens_wordpiece
            if i_t == se:
                tokens_wordpiece = tokens_wordpiece + ['@']
            if i_t == os:
                new_os = len(sents)
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if i_t == oe:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            sents.extend(tokens_wordpiece)

        return ' '.join(sents).replace(' ##','')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default='cb',
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    do_bench: bool = field(
        default=False, metadata={"help": "NVML benchmarking"}
    )
    max_bench_iter: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Seconds for warmup blocked_autorange benchmark"
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(self.task_name, "Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a SuperGLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    comp_func: str = field(
        default='none',
        metadata={"help": "Compression function to be used on finetuned model"}
    )
    rank: int = field(
        default=150, 
        metadata={"help": "rank to compress"})

    tt_ranks: list[int] = field(
        default_factory=lambda:[10,10],
        metadata={"help": "Ranks of TTm decomposition of weights"}
    )
    tt_input_dims: list[int] = field(
        default_factory=lambda:[8,12,8],
        metadata={"help": "Input dimensions in TTMatrix representation of weights"}
    )
    tt_output_dims: list[int] = field(
        default_factory=lambda:[12,16,16],
        metadata={"help": "Output dimensions in TTMatrix representation of weights"}
    )

    double_train: bool = field(
        default=False,
        metadata={"help": "Train model after compression"},
    )

def main(tasks_):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    data_args.task_name = tasks_
    print(data_args.task_name)
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name, tasks_)
    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = os.path.join(training_args.resume_from_checkpoint, tasks_)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name in ['copa', 'rte', 'wsc']:
        raw_datasets = load_dataset(
            "super_glue",
            data_args.task_name if data_args.task_name != 'wsc' else 'wsc.fixed',
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif data_args.task_name is not None:
        raw_datasets = load_task(
            'super_glue/'+data_args.task_name)
        raw_datasets = raw_datasets.rename_column('labels','label')

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = config.eos_token_id

    if data_args.task_name == 'copa':
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    def preprocess_function_copa(examples):
        first_sentences = []
        second_sentences = []
        for indx, line in enumerate(examples['question']):
            if line == 'effect':
                first_sentences += [examples['premise'][indx], examples['premise'][indx]]
                second_sentences += [examples['choice1'][indx], examples['choice2'][indx]]
            else:
                first_sentences += [examples['choice1'][indx], examples['choice2'][indx]]
                second_sentences += [examples['premise'][indx], examples['premise'][indx]]

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)

        return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

    def preprocess_function_wsc(examples):
        first_sentence = WSC_preprocess.tokenize(tokens=examples['text'].split(' '),
                   ss=examples['span1_index'], se=(examples['span1_index']+len(examples['span1_text'].split(' '))-1),
                   os=examples['span2_index'], oe=(examples['span2_index']+len(examples['span2_text'].split(' '))-1)
                   )
    
        return {'text':first_sentence}

    #TODO:bad optimization code, need to be batched
    if data_args.task_name == 'wsc':
        WSC_preprocess = ProcessorWSC(tokenizer)
        raw_datasets = raw_datasets.map(preprocess_function_wsc, batched=False)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function if data_args.task_name != 'copa' else preprocess_function_copa,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if False: #data_args.task_name is not None:
        metric = evaluate.load("super_glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_ = np.squeeze(preds_) if is_regression else np.argmax(preds_, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds_, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds_ - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds_ == p.label_ids).astype(np.float32).mean().item()}

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
    from typing import Optional, Union
    @dataclass
    class DataCollatorForMultipleChoice:
        """
        Data collator that will dynamically pad the inputs for multiple choice received.
        """

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
            ]
            flattened_features = sum(flattened_features, [])

            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if data_args.task_name == 'copa':
        data_collator = DataCollatorForMultipleChoice(tokenizer)


    # Initialize our Trainer
    if not model_args.comp_func in ['none', None]:
        trainer = CustomTrainerBert(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.make_grad_bank(model)

        import pickle
        with open(os.path.join(training_args.resume_from_checkpoint,'weight_dict.pickle'), 'rb') as ff:
            mass = pickle.load(ff)
        trainer.grad_bank_int_2 = mass['weight_int']
        trainer.grad_bank_out_2 = mass['weight_out']
        trainer.avg_counter = mass['weight_count']


    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.model.to('cuda')

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if training_args.save_strategy != 'no':
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            import pickle
            full_dict = {'weight_int':[],
                        'weight_out':[],
                        'weight_count':0}
            full_dict['weight_int'] = trainer.grad_bank_int_2
            full_dict['weight_out'] = trainer.grad_bank_out_2
            full_dict['weight_count'] = trainer.avg_counter

            with open(os.path.join(trainer.args.output_dir, 'weight_dict.pickle'), 'wb') as handle:
                pickle.dump(full_dict, handle)


    # EVALUATION
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if not model_args.comp_func in ['none', None]:
            def_class = MODEL_NAMES[model_args.comp_func]
            class_module = __import__("exps.models", fromlist=[def_class])
            model_def = getattr(class_module, def_class)
            if model_args.comp_func in ["ttm_ffn", "ttm_ffn_alt"]:
                trainer.model = model_def(trainer.model, model_args.tt_ranks, 
                                          model_args.tt_input_dims, model_args.tt_output_dims)
            elif model_args.comp_func in ["ttm_ffn_w_inv","ttm_ffn_w", "ttm_ffn_alt_w"]:
                trainer.model = model_def(trainer.model, model_args.tt_ranks, 
                                          model_args.tt_input_dims, model_args.tt_output_dims,
                                          weight_int=trainer.grad_bank_int_2, 
                                          weight_out=trainer.grad_bank_out_2, 
                                          weight_count=trainer.avg_counter)
            else:
                trainer.model = model_def(trainer.model, model_args.rank, 
                                          weight_int=trainer.grad_bank_int_2, weight_out=trainer.grad_bank_out_2, 
                                          weight_count=trainer.avg_counter)
                #trainer.model = model_def(trainer.model, model_args.rank)
            trainer.model.to('cuda')

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    # BENCHMARKING
    if data_args.do_bench:
        logger.info("*** Benchmarking ***")
        if not model_args.comp_func in ['none', None] and model_args.double_train:
            trainer2 = Trainer(
                model=trainer.model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,)
            trainer2.train()

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}
        for eval_dataset, task in zip(eval_datasets, tasks):

            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)
                
            size_of = trainer.model.get_memory_footprint()
            param_of = sum(p.numel() for p in trainer.model.parameters())
            metrics.update({'size_of':size_of})
            metrics.update({'param_of':param_of})

            
            if task is not None and "mnli" in task:
                combined.update({'size_of':size_of})
                combined.update({'param_of':param_of})
                
            trainer.log_metrics("bench_", metrics)
            trainer.save_metrics("bench_0", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

        #training_args.output_dir
    #return training_args.output_dir

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

##BAD MANNERS
if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    tasks_ = ['copa', 'multirc', 'rte', 'boolq', 'cb', 'wic', 'wsc'] #  ,'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'wnli'
    synth_bench()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name")
    parser.add_argument("--output_dir")
    args_alt, _ = parser.parse_known_args()

    for task_ in tasks_:
        path_to = main(task_)
    OverallTable(os.path.join(args_alt.output_dir, args_alt.run_name,'..'), 'results.csv', 'superglue')