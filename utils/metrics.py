import re
import string
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from scipy.special import softmax
from datasets import load_metric
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import EvalPrediction

from .super_glue_data_utils import RecordProcessor


def simple_accuracy(preds: List[int], labels: List[int]) -> float:
    return (preds == labels).mean()


def acc_and_f1(
    preds: List[int], labels: List[int], f1_avg: str = "binary"
) -> Dict[str, float]:
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds: List[int], labels: List[int]) -> Dict[str, float]:
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def normalize_answer(s: str) -> str:
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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths) -> int:
    """Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script"""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _record_f1_score(prediction, ground_truth):
    """Compute normalized token level F1
    From official ReCoRD eval script"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _record_em_score(prediction, ground_truth):
    """Compute normalized exact match
    From official ReCoRD eval script"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def accuracy_score(p: EvalPrediction) -> Dict[str, float]:
    preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_ = np.argmax(preds_, axis=1)
    return {"accuracy": (preds_ == p.label_ids).astype(np.float32).mean().item()}


def acc_f1(p: EvalPrediction, average: str = "macro") -> Dict[str, float]:
    preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_ = np.argmax(preds_, axis=1)
    f1 = f1_score(y_true=p.label_ids.astype(np.float32), y_pred=preds_, average=average)
    result = accuracy_score(p)
    result["f1"] = f1
    return result


def boolq_metric(p: EvalPrediction):
    return accuracy_score(p)


def copa_metric(p: EvalPrediction):
    return accuracy_score(p)


def rte_metric(p: EvalPrediction):
    return accuracy_score(p)


def wic_metric(p: EvalPrediction):
    return accuracy_score(p)


def cb_metric(p: EvalPrediction):
    return acc_f1(p)


def wsc_metric(p: EvalPrediction):
    return acc_f1(p)


def multirc_metric(p: EvalPrediction):

    metric = load_metric("super_glue", "multirc")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=-1)
    guids = p.guids
    labels = p.label_ids.astype(np.float32)

    predictions = [
        {
            "idx": {"answer": idx[2], "paragraph": idx[0], "question": idx[1]},
            "prediction": pred,
        }
        for idx, pred in zip(guids, preds)
    ]
    result = metric.compute(predictions=predictions, references=labels)
    print(result)

    return result


def record_metric(p: EvalPrediction):
    processor = RecordProcessor()
    answers = processor.get_answers("data/ReCoRD/", set_type="dev")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=-1)
    guids = p.guids
    labels = p.label_ids.astype(np.float32)

    assert len(guids) == len(preds), "Different number of predictions and IDs!"
    qst2ans = defaultdict(list)
    # print(preds, labels, guids)
    # iterate over examples and aggregate statistics
    for idx, pred, label in zip(guids, preds, labels):
        qst_idx = (idx[0], idx[1])
        qst2ans[qst_idx].append((idx[2], pred))
    # print(qst2ans)
    f1s, ems = [], []
    for qst, idxs_and_prds in qst2ans.items():
        cands, golds = answers[qst]

        idxs_and_prds.sort(key=lambda x: x[0])
        logits = np.vstack([i[1] for i in idxs_and_prds])

        # take the most probable choice as the prediction
        pred_idx = np.argmax(softmax(logits, axis=1)[:, -1]).item()
        pred = cands[pred_idx]

        # compute metrics
        f1s.append(metric_max_over_ground_truths(_record_f1_score, pred, golds))
        ems.append(metric_max_over_ground_truths(_record_em_score, pred, golds))

    avg_f1 = sum(f1s) / len(f1s)
    avg_em = sum(ems) / len(ems)
    em_and_f1 = (avg_em + avg_f1) / 2
    print("f1", avg_f1, "em", avg_em, "em_and_f1", em_and_f1)
    return {"f1": avg_f1, "em": avg_em, "em_and_f1": em_and_f1}
