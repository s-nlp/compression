import json
import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd
import transformers.utils.logging
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from utils.data_collators import FieldDataCollatorWithPadding
from utils.dataset_configs import (
    COLUMNS_TO_DROP,
    TASK_NUM_CLASSES,
    TASK_TO_CONFIG,
    TASK_TO_NAME,
    TASK_TYPES,
    load_data,
)
from utils.russian_superglue_models import EntityChoiceModel, SpanClassificationModel


class NumpyEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle NumPy data types.

    This class extends the `json.JSONEncoder` class to handle NumPy data types such as
    `np.integer` and `np.floating`. It can also handle NumPy arrays by converting them to
    Python lists. All other data types are handled by the superclass's `default` method.

    Example usage:

        import json
        import numpy as np

        data = {"int": np.int64(42), "float": np.float64(3.14), "array": np.array([1, 2, 3])}
        encoded = json.dumps(data, cls=NumpyEncoder)

    Attributes:
        None

    Methods:
        default(obj): Override the superclass's method to handle NumPy data types and arrays.

    """

    def default(self, obj):
        """
        Override the `default` method to handle NumPy data types and arrays.

        Args:
            obj: A Python object to be encoded as JSON.

        Returns:
            A JSON-serializable Python object.

        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj.tolist() if isinstance(obj, np.ndarray) else super().default(obj)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model(args) -> PreTrainedModel:
    """
    Initializes a correct model for a given task.

    Args:
        args: An object that contains the following fields:
            - model_name: A string with the name of the pre-trained model.
            - task_name: A string with the name of the task.
            - model_name_or_path" A string for model name from HF hub or path to the weights.

    Returns:
        An instance of a pre-trained model for the given task.
    """
    task_type = TASK_TYPES[args.task_name]
    num_classes = TASK_NUM_CLASSES[args.task_name]

    if task_type == "span_classification":
        return SpanClassificationModel(
            backbone=AutoModel.from_pretrained(args.model_name_or_path),
            num_labels=num_classes,
        )
    elif task_type == "entity_choice":
        return EntityChoiceModel(
            backbone=AutoModel.from_pretrained(args.model_name_or_path)
        )
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_labels=num_classes
        )


def main(args):
    if args.model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.task_name in ["russe", "rwsd"]:
        data_collator = FieldDataCollatorWithPadding(
            tokenizer,
            fields_to_pad=(("e1_mask", 0, 0), ("e2_mask", 0, 0)),
            pad_to_multiple_of=8,
        )
    elif args.task_name == "rucos":
        data_collator = FieldDataCollatorWithPadding(
            tokenizer=tokenizer,
            fields_to_pad=(("entity_mask", 0, 1), ("labels", -1, None)),
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    if args.task_name == "rucos":
        # we're using a custom dataset, because the HF hub version has no information about entity indices
        train = Dataset.from_json("data/combined/RuCoS/train.jsonl")
        val = Dataset.from_json("data/combined/RuCoS/val.jsonl")
        test = Dataset.from_json("data/combined/RuCoS/test.jsonl")
        dataset = DatasetDict(train=train, validation=val, test=test)
    elif args.task_name == "rucola":
        train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
            pd.read_csv,
            (
                "combined/RuCoLA/in_domain_train.csv",
                "combined/RuCoLA/in_domain_dev.csv",
                "combined/RuCoLA/out_of_domain_dev.csv",
                "combined/RuCoLA/test.csv",
            ),
        )

        # concatenate datasets to get aggregate metrics
        dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))
        train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
        dataset = DatasetDict(train=train, validation=dev, test=test)
    else:
        dataset = load_data(task_name=args.task_name)

    config = TASK_TO_CONFIG[args.task_name](dataset)

    processed_dataset = dataset.map(
        partial(
            config.process_data,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        ),
        num_proc=32,
        keep_in_memory=True,
        batched=args.task_name != "rwsd",
        remove_columns=COLUMNS_TO_DROP[args.task_name],
    )

    transformers.utils.logging.enable_progress_bar()
    model_name = (
        f"{args.model_name_or_path}_{args.task_name}_{args.rank or None}_{args.seed}"
    )
    dev_metrics_per_run, predictions_per_run = [], []
    set_seed(args.seed)

    model = get_model(args)
    run = wandb.init(project="russian_superglue", name=model_name)
    run.config.update({"task": TASK_TO_NAME[args.task_name], "model": model_name})

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{model_name}/{TASK_TO_NAME[args.task_name]}",
        overwrite_output_dir=args.overwrite_output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        save_strategy="epoch",
        seed=args.seed,
        fp16=args.fp16,
        group_by_length=True,
        report_to="wandb",
        run_name=model_name,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=config.best_metric,
    )

    if args.task_name != "lidirus":

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            compute_metrics=partial(
                config.compute_metrics,
                split="validation",
                processed_dataset=processed_dataset["validation"],
            ),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        print("train", train_result.metrics)
        trainer.log(train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        dev_predictions = trainer.predict(test_dataset=processed_dataset["validation"])
        print("dev", dev_predictions.metrics)
        trainer.log(dev_predictions.metrics)
        trainer.save_metrics("dev", dev_predictions.metrics)

        run.summary.update(dev_predictions.metrics)

    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=processed_dataset,
            compute_metrics=partial(
                config.compute_metrics,
                processed_dataset=processed_dataset,
            ),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        dev_predictions = trainer.predict(test_dataset=processed_dataset)
        print(dev_predictions.metrics)
        trainer.save_metrics("dev", dev_predictions.metrics)

    wandb.finish()
    dev_metrics_per_run.append(dev_predictions.metrics[f"test_{config.best_metric}"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type (BERT or RoBERTa)",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name to HF hub",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list",
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
    args = parser.parse_args()

    main(args)
