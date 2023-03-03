import json
import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Type, Union

import datasets
import evaluate
import numpy as np
import pandas as pd
import pymorphy2
from datasets import Dataset, DatasetDict, load_metric
from fuzzysearch import find_near_matches
from Levenshtein import distance
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1_score_sklearn
from transformers import EvalPrediction, PreTrainedTokenizer

f1_metric = evaluate.load("f1")


def _quantize_max_length(max_length):
    return max_length - max_length % 8


class DatasetConfig:
    """
    Configuration for a dataset, including metadata and methods for data processing and evaluation.

    This dataclass stores metadata about a dataset, including the name of the best metric for evaluation and the number
    of classes in the dataset. It also provides methods for processing data, computing evaluation metrics, and
    processing model predictions.

    Attributes:
        best_metric (str): The name of the best metric for evaluating the dataset.
        num_classes (int): The number of classes in the dataset.
        data (DatasetDict): A dictionary of the dataset, containing its inputs, labels, and other metadata.

    Methods:
        process_data(examples, tokenizer, max_length): A static method for processing data examples before training.
        compute_metrics(p, split): A method for computing evaluation metrics on model predictions.
        process_predictions(p, split, **kwargs): A method for processing model predictions after inference.

    """

    best_metric: str
    num_classes: int

    def __init__(self, dataset: DatasetDict):
        self.data = dataset

    @staticmethod
    def process_data(
        examples: List[str], tokenizer: PreTrainedTokenizer, max_length: int
    ):
        pass

    def compute_metrics(self, predictions: EvalPrediction, split: str):
        pass

    def process_predictions(
        self, predictions: np.ndarray, split: str, **kwargs
    ) -> List[Dict]:
        pass


class RCBConfig(DatasetConfig):
    best_metric: str = "f1"
    num_classes: int = 3

    _index_to_label = {0: "entailment", 1: "contradiction", 2: "neutral"}

    @staticmethod
    def process_data(
        examples: List[str], tokenizer: PreTrainedTokenizer, max_length: int
    ):
        _label_to_index = {"entailment": 0, "contradiction": 1, "neutral": 2}

        result = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation="longest_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        if isinstance(examples["label"], list):
            result["labels"] = [_label_to_index[x] for x in examples["label"]]
        elif isinstance(examples["label"], str):
            result["labels"] = _label_to_index[examples["label"]]
        return result

    def compute_metrics(self, predictions: EvalPrediction, split: str, **kwargs):
        preds = predictions.predictions.argmax(axis=1)

        return {
            "accuracy": accuracy_score(predictions.label_ids.astype(np.float32), preds),
            "f1": f1_score_sklearn(
                predictions.label_ids.astype(np.float32),
                preds,
                labels=[0, 1, 2],
                average="macro",
            ),
        }

    def process_predictions(self, predictions: np.ndarray, split: str, **kwargs):
        predicted_labels = predictions.argmax(axis=1).tolist()

        return [
            {"idx": i, "label": self._index_to_label[label]}
            for i, label in enumerate(predicted_labels)
        ]


class TerraConfig(DatasetConfig):
    """
    Configuration class for the Terra dataset, which inherits from the DatasetConfig class.

    Attributes:
        best_metric (str): The evaluation metric used to determine the best model during training.
        num_classes (int): The number of classes in the dataset.
        _index_to_label (Dict[int, str]): Mapping from integer labels to string labels.

    Methods:
        process_data: Tokenizes and preprocesses the data for the model training.
        compute_metrics: Computes the evaluation metrics for the predictions made by the model.
        process_predictions: Converts the predictions into a more readable format.

    Usage:
        config = TerraConfig()

    """

    best_metric = "accuracy"
    num_classes = 2
    _index_to_label = {0: "entailment", 1: "not_entailment"}

    @staticmethod
    def process_data(
        examples: List[str], tokenizer: PreTrainedTokenizer, max_length: int
    ):
        """
        Tokenizes a list of examples and returns a dictionary with input IDs,
        token type IDs, and labels.

        Args:
            examples (List[Dict]): A list of dictionaries where each dictionary
                contains a "premise" field with the text of a premise and a
                "hypothesis" field with the text of a hypothesis. Each dictionary may
                also contain a "label" field with a boolean value or a list of boolean
                values indicating whether the corresponding premise entails the
                corresponding hypothesis.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the
                input text.
            max_length (int): The maximum length of the tokenized input sequences.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing the input IDs,
            token type IDs, and labels. The "input_ids" and "token_type_ids" fields
            contain lists of input IDs and token type IDs for each example, and the
            "labels" field contains a list of binary label values for each example.

        """
        _label_to_index = {"entailment": 0, "not_entailment": 1}

        result = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            return_token_type_ids=True,
            padding=False,
        )
        if isinstance(examples["label"], list):
            result["labels"] = [_label_to_index[x] for x in examples["label"]]
        elif isinstance(examples["label"], str):
            result["labels"] = _label_to_index[examples["label"]]
        return result

    def compute_metrics(self, predictions: EvalPrediction, split: str, **kwargs):
        """
        Computes and returns the accuracy of the model predictions.

        Args:
            predictions (EvalPrediction): The model predictions.
            split (str): The split name.

        Returns:
            Dict[str, float]: A dictionary containing the accuracy of the model predictions.
        """
        return {
            "accuracy": accuracy_score(
                y_pred=predictions.predictions.argmax(axis=1),
                y_true=predictions.label_ids.astype(np.float32),
            )
        }

    def process_predictions(self, predictions: np.ndarray, split: str, **kwargs):

        preds_list = predictions.argmax(axis=1).tolist()

        return [
            {"idx": idx, "label": self._index_to_label[predicted_class]}
            for idx, predicted_class in enumerate(preds_list)
        ]


class LiDiRusConfig(TerraConfig):
    """
    A configuration class for LiDiRus diagnostic dataset.
    """

    best_metric: str = "accuracy"
    num_classes: int = 2

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        _label_to_index = {"not_entailment": 0, "entailment": 1}
        result = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            return_token_type_ids=True,
            padding=False,
        )
        if isinstance(examples["label"], list):
            result["labels"] = [_label_to_index[x] for x in examples["label"]]
        else:
            result["labels"] = _label_to_index[examples["label"]]
        return result

    def compute_metrics(self, predictions: EvalPrediction, **kwargs):
        """
        Computes and returns the accuracy of the model predictions.

        Args:
            predictions (EvalPrediction): The model predictions.
            split (str): The split name.

        Returns:
            Dict[str, float]: A dictionary containing the accuracy of the model predictions.
        """
        return {
            "accuracy": accuracy_score(
                y_pred=predictions.predictions.argmax(axis=1),
                y_true=predictions.label_ids.astype(np.float32),
            )
        }


class DaNetQAConfig(DatasetConfig):
    """
    Configuration for the DaNetQA dataset. Inherits from the DatasetConfig class.

    Attributes:
        best_metric (str): The evaluation metric used to determine the best model during training.
        num_classes (int): The number of classes in the dataset.

    Methods:
        process_data: Tokenizes and preprocesses the data for the model training.
        compute_metrics: Computes the evaluation metrics for the predictions made by the model.
        process_predictions: Converts the predictions into a more readable format.

    Usage:
        config = DaNetQAConfig()
    """

    best_metric = "accuracy"
    num_classes = 2

    @staticmethod
    def process_data(
        examples: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, max_length: int
    ) -> Dict[str, List[int]]:
        """
        Tokenizes a list of examples and returns a dictionary with input IDs,
        token type IDs, and labels.

        Args:
            examples (List[Dict]): A list of dictionaries where each dictionary
                contains a "passage" field with the text of a passage and a
                "question" field with the text of a question. Each dictionary may
                also contain a "label" field with a boolean value or a list of boolean
                values indicating whether the corresponding passage answers the
                corresponding question.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the
                input text.
            max_length (int): The maximum length of the tokenized input sequences.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing the input IDs,
            token type IDs, and labels. The "input_ids" and "token_type_ids" fields
            contain lists of input IDs and token type IDs for each example, and the
            "labels" field contains a list of binary label values for each example.
        """
        if "label" not in examples:
            raise ValueError("Missing label field in examples")

        if isinstance(examples["label"], list):
            labels = [int(x) for x in examples["label"]]
        elif isinstance(examples["label"], str):
            labels = int(examples["label"])
        else:
            raise ValueError("Label field should be a list or a string")

        encoded_inputs = tokenizer(
            examples["passage"],
            examples["question"],
            truncation="only_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        encoded_inputs["labels"] = labels
        return encoded_inputs

    def compute_metrics(
        self, predictions: EvalPrediction, split: str, **kwargs
    ) -> Dict[str, float]:
        """
        Computes and returns the accuracy of the model predictions.

        Args:
            predictions (EvalPrediction): The model predictions.
            split (str): The split name.

        Returns:
            Dict[str, float]: A dictionary containing the accuracy of the model predictions.

        """
        preds = predictions.predictions.argmax(axis=1)

        return {
            "accuracy": accuracy_score(
                y_pred=preds, y_true=predictions.label_ids.astype(np.float32)
            )
        }

    def process_predictions(
        self, preds: np.ndarray, split: str, **kwargs
    ) -> List[Dict[str, Union[int, str]]]:
        """
        Processes the predictions returned by the model and returns a list of dictionaries containing the
        prediction index and the predicted label.

        Args:
            p (np.ndarray): The model predictions.
            split (str): The split name.

        Returns:
            List[Dict[str, Union[int, str]]]: A list of dictionaries containing the prediction index and
                the predicted label.

        """
        preds_list = preds.argmax(axis=1).tolist()

        return [
            {"idx": idx, "label": str(bool(prediction)).lower()}
            for idx, prediction in enumerate(preds_list)
        ]


class PARusConfig(DatasetConfig):
    """
    Configuration for the PARus dataset. Inherits from the DatasetConfig class.

    Attributes:
        best_metric (str): The evaluation metric used to determine the best model during training.
        num_classes (int): The number of classes in the dataset.

    Methods:
        process_data: Tokenizes and preprocesses the data for the model training.
        compute_metrics: Computes the evaluation metrics for the predictions made by the model.
        process_predictions: Converts the predictions into a more readable format.

    Usage:
        config = PARusConfig()
    """

    best_metric = "accuracy"
    num_classes = 2

    @staticmethod
    def process_data(
        examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int
    ) -> Dict[str, np.ndarray]:
        """
        Tokenizes the examples in the PARus dataset and returns a dictionary
        containing the tokenized inputs and labels.

        Args:
            examples (Dict[str, List[str]]): The examples to process.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_length (int): The maximum sequence length.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the tokenized inputs and labels.
        """
        first_texts = []
        second_texts = []
        labels = []

        for (premise, choice1, choice2, question, label) in zip(
            examples["premise"],
            examples["choice1"],
            examples["choice2"],
            examples["question"],
            examples["label"],
        ):
            if question == "cause":
                first_texts.extend([choice1, choice2])
                second_texts.extend([premise, premise])
            elif question == "effect":
                first_texts.extend([premise, premise])
                second_texts.extend([choice1, choice2])

            if label == 0:
                # could've used [1-label, label], but they mean different things
                labels.extend([1, 0])
            else:  # includes case of -1 for test data, but we remove it later anyway
                labels.extend([0, 1])

        assert len(first_texts) == len(second_texts) == len(labels)

        result = tokenizer(
            first_texts,
            second_texts,
            truncation="longest_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        result["labels"] = labels
        return result

    def _get_per_instance_preds(self, flattened_preds: np.ndarray) -> np.ndarray:
        """
        Computes the per-instance predictions from the flattened model predictions.

        Args:
            flattened_preds (np.ndarray): The flattened model predictions.

        Returns:
            np.ndarray: The per-instance predictions.
        """
        true_probs = softmax(flattened_preds, axis=1)[:, 1]
        probs_per_instance = true_probs.reshape((true_probs.shape[0] // 2, 2))
        return np.argmax(probs_per_instance, axis=1)

    def compute_metrics(
        self, predictions: EvalPrediction, split: str, **kwargs
    ) -> Dict[str, float]:
        """
        Computes the accuracy metric for the PARus dataset.

        Args:
            predictions (EvalPrediction): The model predictions.
            split (str): The split name.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        per_instance_preds = self._get_per_instance_preds(predictions.predictions)
        label_ids = predictions.label_ids.reshape(
            (per_instance_preds.shape[0], 2)
        ).argmax(
            axis=1
        )  # [0, 1] -> 1, [1, 0] -> 0

        return {
            "accuracy": accuracy_score(
                y_pred=per_instance_preds, y_true=label_ids.astype(np.float32)
            )
        }

    def process_predictions(
        self, predictions: np.ndarray, split: str, **kwargs
    ) -> List[Dict[str, Union[int, str]]]:
        """
        Processes model predictions for a given split,
        returns a list of dicts with the index of each instance
        and its predicted label.

        Args:
            p (np.ndarray): The predictions output by the model for the given split.
            split (str): The name of the split, e.g., "train", "dev", or "test".
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries with the following keys:
                - "idx": The index of the instance in the original dataset.
                - "label": The predicted label for the instance. Possible values are "0" or "1".
        """
        preds = self._get_per_instance_preds(predictions).tolist()

        return [{"idx": idx, "label": is_same} for idx, is_same in enumerate(preds)]


class MuSeRCConfig(DatasetConfig):
    """Configuration class for the MuSeRC dataset."""

    best_metric: str = "f1"
    num_classes: int = 2

    @staticmethod
    def process_data(
        examples: Dict[str, Union[List[str], List[List[str]], List[int]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> Dict[str, Union[List[int], List[List[int]]]]:
        """
        Preprocesses the data.

        Args:
            examples: The examples to preprocess.
            tokenizer: The tokenizer to use.
            max_length: The maximum length of the sequences.

        Returns:
            The preprocessed examples.
        """
        questions_with_answers = [
            question + answer
            for question, answer in zip(examples["question"], examples["answer"])
        ]
        result = tokenizer(
            examples["paragraph"],
            questions_with_answers,
            truncation="only_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )
        result["labels"] = examples["label"]
        return result

    @staticmethod
    def _get_paragraph_metrics(prediction_pair):
        exact_match = int(
            (prediction_pair["prediction"] == prediction_pair["labels"]).all()
        )
        f1_result = f1_score_sklearn(
            y_pred=prediction_pair["prediction"],
            y_true=prediction_pair["labels"],
            average="binary",
        )
        return pd.Series(
            {
                "f1": f1_result,
                "em": exact_match,
            }
        )

    def compute_metrics(
        self, predictions: EvalPrediction, split: str, **kwargs
    ) -> Dict[str, float]:
        """
        Computes the evaluation metrics for the predictions.

        Args:
            predictions: The predictions to evaluate.
            split: The name of the split being evaluated.
            **kwargs: Additional arguments.

        Returns:
            A dictionary of metric names to metric values.
        """
        preds = predictions.predictions.argmax(axis=-1)
        labels = predictions.label_ids.astype(np.float32)

        split_idx_df = pd.DataFrame(self.data[split]["idx"])

        split_idx_df["prediction"] = preds
        split_idx_df["labels"] = labels

        return (
            split_idx_df.groupby("paragraph")
            .apply(self._get_paragraph_metrics)
            .mean()
            .to_dict()
        )

    @staticmethod
    def _unravel_answers(answers_df: pd.DataFrame) -> pd.Series:
        """
        Computes the evaluation metrics for a single paragraph.

        Args:
            x: The data for the paragraph.

        Returns:
            A Pandas Series containing the metrics for the paragraph.
        """
        answers_df_unraveled = answers_df[["answer", "prediction"]].rename(
            columns={"answer": "idx", "prediction": "label"}
        )
        return {
            "answers": answers_df_unraveled.to_dict("records"),
            "idx": answers_df.name,
        }

    @staticmethod
    def _unravel_preds(x):
        questions = x.groupby("question").apply(MuSeRCConfig._unravel_answers).to_list()
        return {"idx": x.name, "passage": {"questions": questions}}

    def process_predictions(self, predictions: np.ndarray, split: str, **kwargs):
        split_idx_df = pd.DataFrame(self.data[split]["idx"])
        preds = predictions.argmax(axis=1)

        split_idx_df["prediction"] = preds

        return split_idx_df.groupby("paragraph").apply(self._unravel_preds).to_list()


class RUSSEConfig(DatasetConfig):
    best_metric: str = "accuracy"
    num_classes: int = 2
    _index_to_label = {0: "false", 1: "true"}
    PUNCTUATION_TO_REMOVE = (
        " «»—-…\xa0" + string.punctuation + "".join(map(str, range(10)))
    )
    morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def _get_leading_trailing_spaces(x):
        start_len = len(x)
        len_lstrip = len(x.lstrip(RUSSEConfig.PUNCTUATION_TO_REMOVE))
        len_rstrip = len(x.rstrip(RUSSEConfig.PUNCTUATION_TO_REMOVE))

        return start_len - len_lstrip, start_len - len_rstrip

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        for key in ["sentence1", "sentence2"]:
            examples[key] = [x.replace("\xa0", " ") for x in examples[key]]

        result = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation="longest_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        e1_masks, e2_masks = [], []

        for example_index, (word, start1, start2, end1, end2, input_ids) in enumerate(
            zip(
                examples["word"],
                examples["start1"],
                examples["start2"],
                examples["end1"],
                examples["end2"],
                result["input_ids"],
            )
        ):

            assert all(char not in word for char in RUSSEConfig.PUNCTUATION_TO_REMOVE)

            e1_mask, e2_mask = np.zeros((len(input_ids),), dtype=bool), np.zeros(
                (len(input_ids),), dtype=bool
            )

            # fill parts of mask corresponding to the target entity
            for seq_index, (mask, start, end) in enumerate(
                zip((e1_mask, e2_mask), (start1, start2), (end1, end2))
            ):
                sentence_idx = "sentence1" if seq_index == 0 else "sentence2"

                # fix error 1: end is longer than the length of the sentence
                input_sentence = examples[sentence_idx][example_index]
                end = min(end, len(input_sentence))

                # fix error 2: spans include spaces
                input_string = input_sentence[start:end]
                (
                    leading_spaces,
                    trailing_spaces,
                ) = RUSSEConfig._get_leading_trailing_spaces(input_string)
                start += leading_spaces
                end -= trailing_spaces

                # fix error 3: spans are too short
                while (
                    start > 0
                    and input_sentence[start - 1]
                    not in RUSSEConfig.PUNCTUATION_TO_REMOVE
                ):
                    start -= 1
                while (
                    end < len(input_sentence)
                    and input_sentence[end] not in RUSSEConfig.PUNCTUATION_TO_REMOVE
                ):
                    end += 1
                input_string_strip = input_sentence[start:end]

                # fix error 4: the word is different from the substring given by indices
                if distance(input_string_strip.lower(), word) > 1 and all(
                    distance(parse_result.normal_form, word_parse_result.normal_form) > 1
                    for parse_result in RUSSEConfig.morph.parse(input_string_strip)
                    for word_parse_result in RUSSEConfig.morph.parse(word)
                ):
                    print(
                        f"Levenshtein distance exceeds 1 for {input_sentence} "
                        f"({word}!={input_string_strip}, {start}, {end}). "
                        f"Resorting to fuzzy search"
                    )

                    parse_result = RUSSEConfig.morph.parse(word)[0]

                    best_match = None
                    best_match_diff = 999

                    for lexeme in parse_result.lexeme:
                        matches = find_near_matches(
                            lexeme.word, input_sentence, max_l_dist=0
                        )

                        for match in matches:
                            if abs(match.start - start) + abs(
                                match.start - start
                            ) < best_match_diff or (
                                abs(match.start - start) + abs(match.start - start)
                                == best_match_diff
                                and len(match.matched) > len(best_match.matched)
                            ):
                                best_match = match
                                best_match_diff = abs(match.start - start) + abs(
                                    match.start - start
                                )
                    if best_match is not None:
                        print(
                            f"Found {best_match.start}:{best_match.end} ({best_match.matched})"
                        )
                        start = best_match.start
                        end = best_match.end
                    else:
                        print(
                            f"!!! Could not find a suitable match for {examples['idx'][example_index]}. Giving up"
                        )

                entity_start = result.char_to_token(
                    example_index, start, sequence_index=seq_index
                )
                assert entity_start is not None, (start, end, seq_index, input_sentence)

                # end-1 because [start:end] denotes a slice and we need the last character
                entity_end = result.char_to_token(
                    example_index, end - 1, sequence_index=seq_index
                )
                assert entity_end is not None, (
                    start,
                    end,
                    input_sentence,
                    input_ids,
                    example_index,
                    examples["idx"][example_index],
                    seq_index,
                )

                entity_tokens = result["input_ids"][example_index][
                    entity_start : entity_end + 1
                ]
                # here we verify that the indexing is correct
                assert (
                    tokenizer.decode(entity_tokens) in input_sentence
                ), tokenizer.decode(entity_tokens)

                mask[entity_start : entity_end + 1] = 1

            e1_masks.append([int(x) for x in e1_mask])
            e2_masks.append([int(x) for x in e2_mask])

        result["e1_mask"] = e1_masks
        result["e2_mask"] = e2_masks

        if isinstance(examples["label"], list):
            result["labels"] = [int(x) for x in examples["label"]]
        else:
            result["labels"] = int(examples["label"])

        return result

    def compute_metrics(self, predictions: EvalPrediction, split: str, **kwargs):
        preds = np.argmax(predictions.predictions, axis=1)
        return {
            "accuracy": accuracy_score(
                y_true=predictions.label_ids.astype(np.float32), y_pred=preds
            )
        }

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [
            {"idx": idx, "label": self._index_to_label[predicted_class]}
            for idx, predicted_class in enumerate(preds_list)
        ]


def normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute normalized token level F1
    From official ReCoRD eval script
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text

    Returns:
        The F1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    From official ReCoRD eval script"""
    """
    Compute normalized exact match score

    Args:
        prediction: The predicted text
        ground_truth: The ground truth text

    Returns:
        True if the predicted text exactly matches the ground truth, False otherwise.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(
    metric_fn: callable, prediction: str, ground_truths: List[str]
) -> float:
    """
    Compute max metric between prediction and each ground truth
    From official ReCoRD eval script
    Args:
        metric_fn: The metric function to compute
        prediction: The predicted text
        ground_truths: A list of ground truth texts

    Returns:
        The maximum score between prediction and each ground truth.
    """
    return max(metric_fn(prediction, gt) for gt in ground_truths)


class RuCoSConfig(DatasetConfig):
    """
    Configuration class for the RuCoS dataset.

    Attributes:
    ----------
    best_metric : str
        The best metric to use for model selection during training.
    num_classes : int
        The number of classes in the dataset.
    PUNCTUATION_TO_REMOVE : str
        A string of punctuation characters to remove from text.
    """

    best_metric: str = "f1"
    num_classes: int = 2

    PUNCTUATION_TO_REMOVE = " " + string.punctuation

    @staticmethod
    def _get_leading_trailing_spaces(text):
        """
        Returns the number of leading and trailing spaces in a string.

        Parameters:
        ----------
        text : str
            The input string.

        Returns:
        -------
        Tuple[int, int]
            A tuple of two integers, the number of leading and trailing spaces in the input string.
        """
        start_len = len(text)
        len_lstrip = len(text.lstrip(RuCoSConfig.PUNCTUATION_TO_REMOVE))
        len_rstrip = len(text.rstrip(RuCoSConfig.PUNCTUATION_TO_REMOVE))

        return start_len - len_lstrip, start_len - len_rstrip

    def process_data(self, examples, tokenizer, max_length):
        query_answers = []
        query_texts = []
        query_ids = []
        passage_entities = []
        passage_texts = []

        entity_masks = []
        target_masks = []

        for passage, qas in zip(examples["passage"], examples["qas"]):
            assert len(qas) == 1, examples["qas"]
            passage_text = passage["text"].replace("\u200b", " ")
            passage_texts.append(passage_text)

            entities = []
            for entity in passage["entities"]:
                entity_text = passage_text[entity["start"] : entity["end"]]
                leading_spaces, trailing_spaces = self._get_leading_trailing_spaces(
                    entity_text
                )
                entity["start"] += leading_spaces
                entity["end"] -= trailing_spaces
                if entity["start"] < entity["end"]:
                    entity["text"] = passage_text[entity["start"] : entity["end"]]
                    entities.append(entity)
                else:
                    print(
                        f'Discarding entity "{entity_text}" ({entity}) due to it consisting entirely of punctuation'
                    )

            qa = qas[0]
            query_texts.append(qa["query"])
            query_ids.append(qa["idx"])

            if "answers" in qa:
                for answer in qa["answers"]:  # `default` handles the test set
                    answer_text = passage_text[answer["start"] : answer["end"]]
                    assert answer_text == answer["text"]

                    leading_spaces, trailing_spaces = self._get_leading_trailing_spaces(
                        answer_text
                    )
                    answer["start"] += leading_spaces
                    answer["end"] -= trailing_spaces

                    assert answer["start"] < answer["end"]

                    if answer not in entities:
                        entities.append(answer)

                query_answers.append(qa["answers"])

            passage_entities.append(entities)

        result = tokenizer(
            passage_texts,
            query_texts,
            truncation="only_first",
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        entities, lengths = [], []

        for example_index, (example_entities, input_ids) in enumerate(
            zip(passage_entities, result["input_ids"])
        ):
            input_length = len(input_ids)
            lengths.append(input_length)

            example_entity_masks = []
            example_target_masks = []
            example_correct_entities = []

            for entity in example_entities:
                start = entity["start"]
                end = entity["end"]

                entity_start = result.char_to_token(
                    example_index, start, sequence_index=0
                )

                # end-1 because [start:end] denotes a slice, we need the last character
                entity_end = result.char_to_token(
                    example_index, end - 1, sequence_index=0
                )

                if entity_start is not None and entity_end is not None:
                    entity_mask = np.zeros((input_length,), dtype=int)
                    entity_mask[entity_start : entity_end + 1] = 1
                    example_entity_masks.append(entity_mask)

                    if query_answers:
                        if entity in query_answers[example_index]:
                            example_target_masks.append(1)
                        else:
                            example_target_masks.append(0)

                    example_correct_entities.append(entity)
                else:
                    print(
                        examples["passage"][example_index]["text"],
                        examples["passage"][example_index]["entities"],
                    )

            entity_masks.append(np.stack(example_entity_masks))
            target_masks.append(np.array(example_target_masks))
            entities.append(example_correct_entities)

        result["entity_mask"] = entity_masks
        result["entities"] = entities
        result["idx"] = examples["idx"]
        result["length"] = lengths

        if query_answers:
            result["answers"] = query_answers
            result["labels"] = target_masks

        return result

    def compute_metrics(
        self, predictions: EvalPrediction, processed_dataset: datasets.Dataset, **kwargs
    ) -> Dict[str, float]:
        """
        Compute F1 and EM metrics for the given predictions and dataset.
        Args:
            predictions (EvalPrediction): Object that contains the predictions and labels
            processed_dataset (Dataset): Dataset with the entities and answers

        Returns:
            Dict[str, Any]: Dictionary with the F1 and EM metrics
        """

        predicted_entities = np.argmax(predictions.predictions, axis=1)
        text_entities = processed_dataset["entities"]
        text_answers = processed_dataset["answers"]

        f1_values: List[float] = []
        exact_match_values: List[float] = []

        for pred_idx, entities, targets in zip(
            predicted_entities, text_entities, text_answers
        ):
            prediction = entities[pred_idx]["text"]
            target_texts = [answer["text"] for answer in targets]

            f1_values.append(
                metric_max_over_ground_truths(f1_score, prediction, target_texts)
            )

            exact_match_values.append(
                metric_max_over_ground_truths(
                    exact_match_score, prediction, target_texts
                )
            )

        return {"f1": np.mean(f1_values), "em": np.mean(exact_match_values)}

    def process_predictions(
        self, p: np.ndarray, processed_dataset: datasets.Dataset, **kwargs
    ) -> List[Dict]:
        preds_list = p.argmax(axis=1).tolist()

        text_entities = processed_dataset["entities"]

        return [
            {"idx": idx, "label": text_entities[idx][predicted_entity]["text"]}
            for idx, predicted_entity in enumerate(preds_list)
        ]


class RuCoLAConfig(DatasetConfig):
    """
    Configuration class for the RuCoLA dataset.
    """

    best_metric: str = "mcc"
    num_classes: int = 2

    @staticmethod
    def process_data(
        examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer, max_length: int
    ) -> Dict[str, Any]:
        """
        Tokenizes the input examples and returns a dictionary with the tokenized inputs
        and optionally the labels.

        Args:
            examples (Dict[str, List[Any]]): A dictionary containing the input examples.
            tokenizer (PreTrainedTokenizerBase): A tokenizer to use for tokenization.
            max_length (int): The maximum length of the tokenized input.

        Returns:
            A dictionary containing the tokenized inputs and optionally the labels.
        """

        result = tokenizer(
            examples["sentence"],
            return_token_type_ids=True,
            max_length=_quantize_max_length(max_length),
            padding=False,
        )

        if "acceptable" in examples:
            result["labels"] = examples["acceptable"]
        return result

    def compute_metrics(self, predictions: EvalPrediction, split: str, **kwargs):
        preds = predictions.predictions.argmax(1)
        """
        Computes the accuracy and Matthews correlation coefficient for the model
        predictions and returns them as a dictionary.

        Args:
            p (EvalPrediction): The predictions to evaluate.
            split (str): The split on which the predictions were made.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing the computed metrics.
        """
        return {
            "accuracy": accuracy_score(
                y_pred=preds, y_true=predictions.label_ids.astype(np.float32)
            ),
            "mcc": matthews_corrcoef(
                y_pred=preds, y_true=predictions.label_ids.astype(np.float32)
            ),
        }

    def process_predictions(self, predictions: np.ndarray, split: str, **kwargs):
        """
        Processes the model predictions and returns them as a list of dictionaries.

        Args:
            p (np.ndarray): The model predictions to process.
            split (str): The split on which the predictions were made.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries containing the processed predictions.
        """
        preds_list = predictions.argmax(axis=1).tolist()

        return [
            {"id": idx, "acceptable": predicted_class}
            for idx, predicted_class in enumerate(preds_list)
        ]


class RWSDConfig(DatasetConfig):
    """
    Configuration class for the RWSD dataset.
    NOTE: the implementation of span masking is incomplete.
    """

    best_metric: str = "accuracy"
    num_classes: int = 2

    @staticmethod
    def process_data(examples, tokenizer: PreTrainedTokenizer, max_length: int):
        text = examples["text"].translate(
            str.maketrans({a: None for a in string.punctuation})
        )
        result = tokenizer(
            text,
            truncation="longest_first",
            return_token_type_ids=True,
            max_length=max_length,
            padding=False,
        )

        e1_mask = np.zeros_like(result["input_ids"], dtype=int)
        e2_mask = np.zeros_like(result["input_ids"], dtype=int)

        e1_span = examples["target"]["span1_text"]
        e2_span = examples["target"]["span2_text"]
        # Find the start and end indices of the spans in the input text
        e1_start, e1_end = find_sub_list(e1_span.split(), text.split())
        e2_start, e2_end = find_sub_list(e2_span.split(), text.split())

        if isinstance(e1_start, int) and isinstance(e1_end, int):
            e1_mask[e1_start:e1_end] = 1

        if isinstance(e2_start, int) and isinstance(e2_end, int):
            e2_mask[e2_start:e2_end] = 1

        result["e1_mask"] = e1_mask
        result["e2_mask"] = e2_mask

        if isinstance(examples["label"], list):
            result["labels"] = [int(x) for x in examples["label"]]
        else:
            result["labels"] = int(examples["label"])

        return result

    def compute_metrics(self, predictions: EvalPrediction, split: str, **kwargs):
        preds = np.argmax(predictions.predictions, axis=1)
        return {
            "accuracy": accuracy_score(
                y_true=predictions.label_ids.astype(np.float32), y_pred=preds
            )
        }


def find_sub_list(sublist: List[str], main_list: List[str]):
    start, end = None, None
    sublist_length = len(sublist)
    for idx, word in enumerate(main_list):
        if distance(word, sublist[0]) < 2:
            start = idx
            if main_list[idx : idx + sublist_length] == sublist:
                end = idx + sublist_length
                return start, end
        if distance(word, sublist[-1]) < 2:
            end = idx

    if end is None:
        end = start + len(sublist)
    if start is None:
        start = end - 1
    return (start, end + 1) if start == end else (start, end)


def load_data(task_name: str, data_path: str = "data/combined/") -> DatasetDict:
    """
    Loads data for a given task from JSON files.

    Args:
        task_name (str): The name of the task for which to load data.
        data_path (str, optional): The directory containing the JSON files. Defaults to "data/combined/".

    Returns:
        A `DatasetDict` object containing the loaded training, validation, and test datasets.

    Raises:
        FileNotFoundError: If any of the required JSON files cannot be found.

    """
    if task_name != "lidirus":
        task_path = os.path.join(data_path, TASK_TO_NAME[task_name])
        train_file = os.path.join(task_path, "train.jsonl")
        val_file = os.path.join(task_path, "val.jsonl")

        if not all(os.path.isfile(p) for p in [train_file, val_file]):
            raise FileNotFoundError(
                f"Could not find required files for task '{task_name}' in directory '{data_path}'"
            )

        if task_name != "muserc":
            train_dataset = Dataset.from_json(train_file)
            val_dataset = Dataset.from_json(val_file)

        else:
            with open(train_file, encoding="utf-8") as f:
                train_list = [_row_to_dict(json.loads(line)) for line in f]
            with open(val_file, encoding="utf-8") as f:
                val_list = [_row_to_dict(json.loads(line)) for line in f]
            train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_list))
            val_dataset = Dataset.from_pandas(pd.DataFrame(data=val_list))

        return DatasetDict(train=train_dataset, validation=val_dataset)
    else:
        task_path = os.path.join(data_path, "LiDiRus/LiDiRus.jsonl")
        return Dataset.from_json(task_path)


def _cast_label(label: Union[str, bool, int]) -> str:
    """Converts the label into the appropriate string version."""
    if isinstance(label, str):
        return label
    elif isinstance(label, bool):
        return "True" if label else "False"
    elif isinstance(label, int):
        assert label in (0, 1)
        return str(label)
    else:
        raise ValueError("Invalid label format.")


def _row_to_dict(row):
    paragraph = row["passage"]
    for question in paragraph["questions"]:
        for answer in question["answers"]:
            label = -1 if answer.get("label") is None else int(answer["label"])
            return {
                "paragraph": paragraph["text"],
                "question": question["question"],
                "answer": answer["text"],
                "label": label,
                "idx": {
                    "paragraph": row["idx"],
                    "question": question["idx"],
                    "answer": answer["idx"],
                },
            }


TASK_TO_CONFIG: Dict[str, Type[DatasetConfig]] = {
    "rcb": RCBConfig,
    "terra": TerraConfig,
    "danetqa": DaNetQAConfig,
    "lidirus": LiDiRusConfig,
    "parus": PARusConfig,
    "muserc": MuSeRCConfig,
    "russe": RUSSEConfig,
    "rucos": RuCoSConfig,
    "rucola": RuCoLAConfig,
    "rwsd": RWSDConfig,
}

TASK_TO_NAME: Dict[str, str] = {
    "rcb": "RCB",
    "terra": "TERRa",
    "danetqa": "DaNetQA",
    "rwsd": "RWSD",
    "lidirus": "LiDiRus",
    "parus": "PARus",
    "muserc": "MuSeRC",
    "russe": "RUSSE",
    "rucos": "RuCoS",
    "rucola": "RuCoLA",
}

TASK_TYPES: Dict[str, str] = {
    "rcb": "classification",
    "terra": "classification",
    "danetqa": "classification",
    "lidirus": "classification",
    "parus": "classification",
    "muserc": "classification",
    "russe": "span_classification",
    "rucos": "entity_choice",
    "rucola": "classification",
    "rwsd": "span_classification",
}

TASK_NUM_CLASSES: Dict[str, int] = {
    "rcb": 3,
    "terra": 2,
    "danetqa": 2,
    "rwsd": 2,
    "lidirus": 2,
    "parus": 2,
    "muserc": 2,
    "russe": 2,
    "rucos": 2,
    "rucola": 2,
}


COLUMNS_TO_DROP: Dict[str, List[str]] = {
    "rcb": ["premise", "label", "hypothesis", "verb", "negation", "genre"],
    "terra": ["premise", "hypothesis", "label"],
    "danetqa": ["question", "passage", "label"],
    "parus": ["premise", "choice1", "choice2", "question", "label", "idx"],
    "muserc": ["paragraph", "question", "answer", "label"],
    "russe": [
        "idx",
        "word",
        "sentence1",
        "sentence2",
        "start1",
        "end1",
        "start2",
        "end2",
        "label",
        "gold_sense1",
        "gold_sense2",
    ],
    "rwsd": ["text", "target", "idx", "label"],
    "rucos": ["passage", "qas"],
    "lidirus": [
        "idx",
        "label",
        "sentence1",
        "sentence2",
        "knowledge",
        "logic",
        "lexical-semantics",
        "predicate-argument-structure",
    ],
}
