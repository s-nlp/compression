import argparse
import json
import os
from typing import Dict

import pandas as pd


def read_json(file_path: str) -> Dict[str, float]:
    with open("file_path/eval_results.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        return {
            k: v
            for k, v in data.items()
            if k
            in ["eval_f1_a", "eval_f1_m", "eval_exact_match", "eval_accuracy", "eval_f1"]
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        required=True,
        default="superglue_models/bert-base-uncased",
        type=str,
    )
    parser.add_argument(
        "--output_filename", default="superglue_parsed_results.csv", type=str
    )

    args = parser.parse_args()

    # Define the list of task directories and metrics for each task
    tasks = [
        ("boolq", ["acc"]),
        ("cb", ["acc", "f1"]),
        ("copa", ["acc"]),
        ("multirc", ["em", "f1a", "f1em"]),
        ("record", ["em", "f1a", "f1em"]),
        ("rte", ["acc"]),
        ("wic", ["acc"]),
        ("wsc", ["acc", "f1"]),
    ]

    # Initialize an empty dictionary to store the results
    results = {}

    # Iterate over the tasks and load the metrics from the JSON files
    for task, metrics in tasks:
        results[task] = {metric: None for metric in metrics}
        for eval_file in [
            "eval_results.json",
            "eval2_results.json",
            "eval3_results.json",
        ]:
            eval_path = os.path.join(task, eval_file)
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    eval_results = json.load(f)
                    for metric in metrics:
                        if isinstance(metric, list):
                            for sub_metric in metric:
                                if sub_metric in eval_results:
                                    results[task][sub_metric] = eval_results[sub_metric]
                        elif metric in eval_results:
                            results[task][metric] = eval_results[metric]

    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results).T
    df.index.name = "Task"

    # Calculate the overall averages
    averages = df.mean().round(2)

    # Add a row for the averages
    df.loc["AVG"] = averages

    # Reorder the columns
    df = df[["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]]

    # Reorder the metrics within each column
    column_order = {
        "boolq": ["acc"],
        "cb": ["acc", "f1"],
        "copa": ["acc"],
        "multirc": ["em", "f1a", "f1em"],
        "record": ["em", "f1a", "f1em"],
        "rte": ["acc"],
        "wic": ["acc"],
        "wsc": ["acc", "f1"],
    }

    for col in df.columns:
        df[col] = df[col][column_order[col]]

    # Add a blank row between the tasks and the averages
    df.loc[""] = ""

    # Create a multi-level column index
    columns = pd.MultiIndex.from_product([["AVG", *df.columns], df.columns[0]])
    df.columns = columns

    # Print the DataFrame
    print(df)

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter("superglue_metrics.xlsx") as writer:
        df.to_excel(writer, index=True)
