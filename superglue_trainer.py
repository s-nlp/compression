# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on SuperGLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import logging
import os
import pickle
import random
import warnings

import numpy as np
import torch
import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    TrainingArguments,
)
from transformers import logging as trlogging

from exps.models import MODEL_NAMES
from utils.custom_trainer import SuperGlueTrainer, SuperGlueTrainerBasic
from utils.data_utils import SuperGLUEDataset
from utils.metrics import (
    boolq_metric,
    cb_metric,
    copa_metric,
    multirc_metric,
    record_metric,
    rte_metric,
    wic_metric,
    wsc_metric,
)
from utils.modeling import (
    BertForSpanClassification,
    DistilBertForSpanClassification,
    RobertaForSpanClassification,
)
from utils.super_glue_data_utils import (
    superglue_convert_examples_to_features as convert_examples_to_features,
)
from utils.super_glue_data_utils import superglue_output_modes as output_modes
from utils.super_glue_data_utils import superglue_processors as processors
from utils.super_glue_data_utils import superglue_tasks_num_spans as task_spans

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        BertTokenizer,
        {
            "classification": BertForSequenceClassification,
            "span_classification": BertForSpanClassification,
        },
    ),
    "roberta": (
        RobertaConfig,
        RobertaTokenizer,
        {
            "classification": RobertaForSequenceClassification,
            "span_classification": RobertaForSpanClassification,
        },
    ),
    "distilbert": (
        DistilBertConfig,
        DistilBertTokenizer,
        {
            "classification": DistilBertForSequenceClassification,
            "span_classification": DistilBertForSpanClassification,
        },
    ),
}

TASK2FILENAME = {
    "boolq": "BoolQ.jsonl",
    "cb": "CB.jsonl",
    "copa": "COPA.jsonl",
    "multirc": "MultiRC.jsonl",
    "record": "ReCoRD.jsonl",
    "rte": "RTE.jsonl",
    "wic": "WiC.jsonl",
    "wsc": "WSC.jsonl",
}

TASK2METRIC = {
    "boolq": boolq_metric,
    "cb": cb_metric,
    "copa": copa_metric,
    "multirc": multirc_metric,
    "record": record_metric,
    "rte": rte_metric,
    "wic": wic_metric,
    "wsc": wsc_metric,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_examples(args, task, tokenizer, split="train"):
    if args.local_rank not in [-1, 0] and split not in ["dev", "test"]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]

    # no cached tensors, process data from scratch
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if split == "dev":
        get_examples = processor.get_dev_examples
    elif split == "test":
        get_examples = processor.get_test_examples
    elif split == "train":
        get_examples = processor.get_train_examples
    examples = get_examples(args.data_dir)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=args.model_type in ["xlnet"],
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    logger.info("\tFinished creating features")
    if args.local_rank == 0 and split not in ["dev", "train"]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    logger.info("Converting features into tensors")
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode in ["classification", "span_classification"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if output_mode in ["span_classification"]:
        # all_starts = torch.tensor([[s[0] for s in f.span_locs] for f in features], dtype=torch.long)
        # all_ends = torch.tensor([[s[1] for s in f.span_locs] for f in features], dtype=torch.long)
        all_spans = torch.tensor([f.span_locs for f in features])
        print(all_labels.shape, all_spans.shape)
        dataset = SuperGLUEDataset(
            input_ids=all_input_ids,
            attention_masks=all_attention_mask,
            token_type_ids=all_token_type_ids,
            labels=all_labels,
            span_cl=True,
            spans=all_spans,
            guids=all_guids,
        )
    else:
        # print('creating dataset')
        dataset = SuperGLUEDataset(
            input_ids=all_input_ids,
            attention_masks=all_attention_mask,
            token_type_ids=all_token_type_ids,
            labels=all_labels,
            span_cl=False,
            guids=all_guids,
        )
    logger.info("\tFinished converting features into tensors")
    if args.task_name != "record" or split not in ["dev", "test"]:
        return dataset
    answers = processor.get_answers(args.data_dir, split)
    return dataset, answers


def main():  # sourcery skip: low-code-quality, remove-unnecessary-cast
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--log_evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta1",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer. Currently not used. ",
    )
    parser.add_argument(
        "--adam_beta2",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer. Currently not used. ",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0,
        type=float,
        help="Linear warmup over warmup_steps as a float.",
    )

    parser.add_argument(
        "--log_energy_consumption",
        action="store_true",
        help="Whether to track energy consumption",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--eval_and_save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_only_best",
        action="store_true",
        help="Save only when hit best validation score.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--evaluate_test", action="store_true", help="Evaluate on the test splits."
    )
    parser.add_argument(
        "--skip_evaluate_dev",
        action="store_true",
        help="Skip final evaluation on the dev splits.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--use_gpuid", type=int, default=-1, help="Use a specific GPU only"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Will enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--comp_func",
        type=str,
        help="Compression function to be used on finetuned model",
        default=None,
    )
    parser.add_argument("--rank", type=int, default=150, help="Rank size to compress.")
    parser.add_argument(
        "--double_train", action="store_true", help="train model after compression"
    )
    # parser.add_argument("--tt_ranks", type=)
    # tt_ranks: Tuple[int, ...] = field(
    #     default=(10,10,10),
    #     metadata={"help": "Ranks of TTm decomposition of weights"}
    # )
    # tt_input_dims: Tuple[int, ...] = field(
    #     default=(4,6,8,4),
    #     metadata={"help": "Input dimensions in TTMatrix representation of weights"}
    # )
    # tt_output_dims: Tuple[int, ...] = field(
    #     default=(8,8,6,8),
    #     metadata={"help": "Output dimensions in TTMatrix representation of weights"}
    # )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Setup CUDA, GPU & distributed training
    if args.use_gpuid > -1:
        device = args.use_gpuid
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Do all the stuff you want only first process to do
    # e.g. make sure only the first process will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir or None, num_labels=num_labels
    )

    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]
        model = MODEL_CLASSES[args.model_type][-1][
            "span_classification"
        ].from_pretrained(
            args.model_name_or_path, config=config, cache_dir=args.cache_dir or None
        )
    else:
        model = MODEL_CLASSES[args.model_type][-1]["classification"].from_pretrained(
            args.model_name_or_path, config=config, cache_dir=args.cache_dir or None
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir or None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    if args.do_train:
        train_dataset = load_examples(args, args.task_name, tokenizer)
        print(train_dataset[0])

    if args.do_eval:
        if args.task_name == "record":
            eval_dataset, _ = load_examples(args, args.task_name, tokenizer, split="dev")
            print(eval_dataset[0])
        else:
            eval_dataset = load_examples(args, args.task_name, tokenizer, split="dev")

    train_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.model_name_or_path}_{args.comp_func}_{args.rank}/{args.task_name}",
        overwrite_output_dir=args.overwrite_output_dir or False,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="steps",
        do_predict=args.evaluate_test or False,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=0 if args.num_train_epochs else args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_and_save_steps,
        save_total_limit=args.save_only_best,
        seed=args.seed,
        no_cuda=args.no_cuda,
        include_inputs_for_metrics=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"{args.model_name_or_path}_{args.comp_func}_{args.rank}_{args.task_name}",
    )

    if args.comp_func in ("svd_ffn_w_inv", "svd_ffn_w"):
        trainer = SuperGlueTrainer(
            model=model,
            # tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=TASK2METRIC[args.task_name],
        )

        trainer.make_grad_bank(model)

    else:
        trainer = SuperGlueTrainerBasic(
            model=model,
            # tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=TASK2METRIC[args.task_name],
        )

    # Training
    if args.do_train:
        trainer.model.to("cuda")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        if train_args.save_strategy != "no":
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            if args.comp_func in ("svd_ffn_w_inv", "svd_ffn_w"):
                full_dict = {"weight_int": [], "weight_out": [], "weight_count": 0}
                full_dict["weight_int"] = trainer.grad_bank_int_2
                full_dict["weight_out"] = trainer.grad_bank_out_2
                full_dict["weight_count"] = trainer.avg_counter

                with open(
                    os.path.join(trainer.args.output_dir, "weight_dict.pickle"), "wb"
                ) as handle:
                    pickle.dump(full_dict, handle)

    # Evaluation
    if args.do_eval:
        logger.info("***** Evaluate *****")

        eval_result = trainer.evaluate()
        trainer.save_metrics("eval", eval_result)
        logger.info(eval_result)

    if args.evaluate_test:
        if args.task_name == "record":
            test_dataset, _ = load_examples(
                args, args.task_name, tokenizer, split="test"
            )
        else:
            test_dataset = load_examples(args, args.task_name, tokenizer, split="test")

        trainer.predict(test_dataset)

    if args.comp_func not in ("none", None):
        class_module = __import__("exps.models", fromlist=[MODEL_NAMES[args.comp_func]])
        model_def = getattr(class_module, MODEL_NAMES[args.comp_func])
        if args.comp_func == "ttm_ffn":
            trainer.model = model_def(
                trainer.model, (10, 10, 10), (4, 6, 8, 4), (8, 8, 6, 8)
            )
        elif args.comp_func == "svd_ffn":
            trainer.model = model_def(trainer.model, args.rank)
        elif args.comp_func == "our_svd":
            trainer.model = model_def(trainer.model, args.rank)
        trainer.model.to("cuda")
        eval_result = trainer.evaluate()
        trainer.save_metrics("eval", eval_result)
        logger.info(eval_result)

    if args.double_train:
        logger.info("*** Second train loop ***")
        trainer2 = SuperGlueTrainerBasic(
            model=trainer.model,
            args=train_args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=TASK2METRIC[args.task_name],
            tokenizer=tokenizer,
        )
        trainer2.train()
        logger.info("*** Evaluate after retraining ***")
        eval_result = trainer2.evaluate(eval_dataset=eval_dataset)
        trainer2.log_metrics("eval", eval_result)
        trainer2.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
