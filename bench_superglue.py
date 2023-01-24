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
import glob
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.modeling import (
    BertForSpanClassification,
    RobertaForSpanClassification,
    DistilBertForSpanClassification,
)
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
    BertTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from super_glue_data_utils import (
    superglue_convert_examples_to_features as convert_examples_to_features,
    superglue_output_modes as output_modes,
    superglue_processors as processors,
    superglue_tasks_metrics as task_metrics,
    superglue_tasks_num_spans as task_spans,
)

from utils.metrics import superglue_compute_metrics as compute_metrics

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    # if args.local_rank in [-1, 0]:
    #    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:  # number of training steps = number of epochs * number of batches
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
    num_warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
        logger.info("Training with fp16.")

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    # )
    train_iterator = range(epochs_trained, int(args.num_train_epochs))

    set_seed(args)  # Added here for reproductibility
    best_val_metric = None
    for epoch_n in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_n}",
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                results = None
                logs = {}
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    if (
                        args.local_rank == -1
                        and args.log_evaluate_during_training
                        and results is None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = evaluate(
                            args, args.task_name, model, tokenizer, use_tqdm=False
                        )
                        for key, value in results.items():
                            eval_key = f"eval_{key}"
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["avg_loss_since_last_log"] = loss_scalar
                    logging_loss = tr_loss

                    # for key, value in logs.items():
                    #    tb_writer.add_scalar(key, value, global_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))
                    logging.info(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.eval_and_save_steps > 0
                    and global_step % args.eval_and_save_steps == 0
                ):
                    # evaluate
                    results, _, _ = evaluate(
                        args, args.task_name, model, tokenizer, use_tqdm=False
                    )
                    for key, value in results.items():
                        logs[f"eval_{key}"] = value
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # save
                    if args.save_only_best:
                        output_dirs = []
                    else:
                        output_dirs = [
                            os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        ]
                    curr_val_metric = results[task_metrics[args.task_name]]
                    if best_val_metric is None or curr_val_metric > best_val_metric:
                        # check if best model so far
                        logger.info("Congratulations, best model so far!")
                        output_dirs.append(
                            os.path.join(args.output_dir, "checkpoint-best")
                        )
                        best_val_metric = curr_val_metric

                    for output_dir in output_dirs:
                        # in each dir, save model, tokenizer, args, optimizer, scheduler
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", output_dir)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info("\tSaved model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            # train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, task_name, model, tokenizer, split="dev", prefix="", use_tqdm=True):

    results = {}
    if task_name == "record":
        eval_dataset, eval_answers = load_and_cache_examples(
            args, task_name, tokenizer, split=split
        )
    else:
        eval_dataset = load_and_cache_examples(args, task_name, tokenizer, split=split)
        eval_answers = None

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = (
        tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    )
    # debug_idx = 0
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        guids = batch[-1]

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
            ex_ids.append(guids.detach().cpu().numpy())
        # debug_idx += 1
        # if debug_idx > 10:
        #    break

    ex_ids = np.concatenate(ex_ids, axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode in [
        "classification",
        "span_classification",
    ] and args.task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    # logging.info(f"predictions: {preds}")
    if split != "test":
        # don't have access to test labels, so skip evaluating on them
        # NB(AW): forcing evaluation on ReCoRD on test (no labels) will error
        result = compute_metrics(
            task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers
        )
        results |= result
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        # K O C T bI Jl b
        os.makedirs(os.path.join(args.output_dir, prefix), exist_ok=True)

        with open(output_eval_file, "w") as writer:
            logger.info(f"***** {split} results: {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids


def load_and_cache_examples(args, task, tokenizer, split="train"):
    if args.local_rank not in [-1, 0] and split not in ["dev", "test"]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_tensors_file = os.path.join(
        args.data_dir,
        f'tensors_{split}_{list(filter(None, args.model_name_or_path.split("/"))).pop()}_{str(args.max_seq_length)}_{str(task)}',
    )

    if os.path.exists(cached_tensors_file) and not args.overwrite_cache:
        logger.info("Loading tensors from cached file %s", cached_tensors_file)
        start_time = time.time()
        dataset = torch.load(cached_tensors_file)
        logger.info("\tFinished loading tensors")
        logger.info(f"\tin {time.time() - start_time}s")

    else:
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
            dataset = TensorDataset(
                all_input_ids,
                all_attention_mask,
                all_token_type_ids,
                all_labels,
                all_spans,
                all_guids,
            )
        else:
            dataset = TensorDataset(
                all_input_ids,
                all_attention_mask,
                all_token_type_ids,
                all_labels,
                all_guids,
            )
        logger.info("\tFinished converting features into tensors")
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_tensors_file)
            torch.save(dataset, cached_tensors_file)
            logger.info("\tFinished saving tensors")

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

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = MODEL_CLASSES[args.model_type][-1][args.output_mode].from_pretrained(
            args.output_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [os.path.join(args.output_dir, "checkpoint-best")]
        if args.eval_all_checkpoints:
            checkpoints = [
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(f"{args.output_dir}/**/{WEIGHTS_NAME}", recursive=True)
                )
            ]

            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]  # if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )
            model = MODEL_CLASSES[args.model_type][-1][args.output_mode].from_pretrained(
                args.output_dir
            )
            model.to(args.device)

            if not args.skip_evaluate_dev:
                result, _, _ = evaluate(
                    args, args.task_name, model, tokenizer, prefix=prefix
                )
                result = {f"{k}_{global_step}": v for k, v in result.items()}

            if args.evaluate_test:
                # Hack to handle diagnostic datasets
                eval_task_names = (
                    ("rte", "ax-b", "ax-g")
                    if args.task_name == "rte"
                    else (args.task_name,)
                )
                for eval_task_name in eval_task_names:
                    result, preds, ex_ids = evaluate(
                        args,
                        eval_task_name,
                        model,
                        tokenizer,
                        split="test",
                        prefix=prefix,
                    )
                    processor = processors[eval_task_name]()
                    if args.task_name == "record":
                        answers = processor.get_answers(args.data_dir, "test")
                        processor.write_preds(
                            preds, ex_ids, args.output_dir, answers=answers
                        )
                    else:
                        processor.write_preds(preds, ex_ids, args.output_dir)


if __name__ == "__main__":
    main()
