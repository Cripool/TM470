"""
Final model metrics and per-image error analysis for the TM470 EMA.

This script compares the final primary versions of:

- Baseline CNN
- Corrected ResNet18
- MobileNetV2

The analysis uses the saved per-image probability exports from the
shared test dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "Baseline CNN" : (
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "baseline_ema_predictions.csv"
    ),

    "Resnet18_Corrected" : (
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "resnet18_corrected_ema_predictions.csv"
    ),

    "MobileNetV2" : (
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "mobilenetv2_ema_predictions.csv"
    ),
}

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "ema_final_metrics_analysis"
)

def inspect_input_files() -> None:
    """
    Confirm that each prediction CSV exists and inspect its structure.
    """

    print("Checking final model prediction files...")

    for model_name, csv_path in MODEL_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Prediction CSV for {model_name} not found at {csv_path}"
            )

        data = pd.read_csv(csv_path)

        print("\n" + "=" * 80)
        print(f"model: {model_name}")
        print(
            "File: "
            f"{csv_path.relative_to(PROJECT_ROOT)}"
        )

        print(f"Rows: {len(data):,}")
        print(f"Columns: {len(data.columns)}")
        print("Column names:")

        for column_name in data.columns:
            print(f" - {column_name}")


def load_and_validate_model_data() -> dict[str, pd.DataFrame]:
    """
    Load the predicition files and confirm that all models used 
    the same test images an true labels.
    """

    model_data: dict[str, pd.DataFrame] = {}

    required_columns = {
        "relative_image_path",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "inference_time_sec",
    }

    reference_images: pd.Series | None = None
    reference_labels: pd.Series | None = None
    reference_model_name: str | None = None

    print("\nValidating shared test data...")

    for model_name, csv_path in MODEL_FILES.items():
        data = pd.read_csv(csv_path)

        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            raise ValueError(
                f"{model_name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        data = data.sort_values(
            "relative_image_path"
        ).reset_index(drop=True)


        duplicate_count = int(
            data["relative_image_path"].duplicated().sum()
        )

        if duplicate_count > 0:
            raise ValueError(
                f"{model_name} has {duplicate_count:,} duplicate "
                "image paths in the predictions CSV."
            )

        if reference_images is None:
            reference_images = data["relative_image_path"]
            reference_labels = data["true_label"]
            reference_model_name = model_name

        else:
            if not data["relative_image_path"].equals(
                reference_images
            ):
                raise ValueError(
                    f"{model_name} has a different set of test images "
                    f"than {reference_model_name}"
            )

            if not data ["true_label"].equals(reference_labels):
                raise ValueError(
                    f"{model_name} has a different set of true labels "
                    f"than {reference_model_name}"
                )

        model_data[model_name] = data

        print(
            f"Validated {model_name}: "
            f"{len(data):,} unique test images, "
            f"{data['true_label'].nunique():,} unique true labels"
        )

    print(
        "\nAll three models have the same test images and true labels."
    )

    return model_data

        

def calculate_overall_metrics(
        model_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate consistent overall performance metrics for each model.
    """

    metric_rows: list[dict[str, object]] =[]

    print("\nCalculating overall metrics...")

    for model_name, data in model_data.items():
        true_labels = data["true_label"]
        predicted_labels = data["predicted_label"]

        accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _,
        ) = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )


        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )

        correct_predictions = int(
            (true_labels == predicted_labels).sum()
        )

        incorrect_predictions = int(
            len(data) - correct_predictions
        )

        metric_rows.append({
            "model": model_name,
            "test_images": len(data),
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "average_inference_time_ms": (
                data["inference_time_sec"].mean() * 1000
            ),
            "median_inference_time_ms": (
                data["inference_time_sec"].median() * 1000
            ),
        }
    )

    metrics_table = pd.DataFrame(metric_rows)

    metrics_table = metrics_table.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = (OUTPUT_DIRECTORY / "overall_metrics.csv")

    metrics_table.to_csv(output_path, index=False)

    print("\nOverall model metrics:")
    print(
        metrics_table.round(4).to_string(index=False)
    )

    print("\nSaved overall metrics to: "
          f"{output_path.relative_to(PROJECT_ROOT)}")

    return metrics_table


def calculate_per_class_metrics(
        model_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate precision, recall, F1-score and support
    for each mushroom class for every model."""


    metric_rows: list[dict[str, object]] = []

    print("\nCalculating per-class metrics...")

    for model_name, data in model_data.items():
        class_names = sorted(data["true_label"].unique()
        )

        (
            precision_values,
            recall_values,
            f1_values,
            support_values,
        ) = precision_recall_fscore_support(
            data["true_label"],
            data["predicted_label"],
            labels=class_names,
            zero_division=0,
            average=None,
        )

        for (

            class_name,
            precision,
            recall,
            f1_score,
            support,
        ) in zip(
            class_names,
            precision_values,
            recall_values,
            f1_values,
            support_values,
        ):
            metric_rows.append({
                "model": model_name,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": int(support),
            }

        )

        model_results = pd.DataFrame(
            [
                row
                for row in metric_rows
                if row["model"] == model_name
            ]
        )

        strongest_class = model_results.loc[
            model_results["f1_score"].idxmax()
        ]
        weakest_class = model_results.loc[
            model_results["f1_score"].idxmin()
        ]

        print(
            f"\n{model_name}:"
        )
        print(
            " strongest class: "
            f"{strongest_class['class_name']} "
            f"(F1={strongest_class['f1_score']:.4f})"
        )
        print(
            "Weakest class: "
            f"{weakest_class['class_name']} "
            f"(F1={weakest_class['f1_score']:.4f})"

        )

    per_class_table = pd.DataFrame(metric_rows)

    per_class_table = per_class_table.sort_values(
        ["model", "class_name"]
    ).reset_index(drop=True)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    long_output_path = (
        OUTPUT_DIRECTORY / "per_class_metrics_long.csv"
    )

    per_class_table.to_csv(long_output_path, index=False,

    )

    f1_comparison = per_class_table.pivot(
        index="class_name",
        columns="model",
        values="f1_score",
    ).reset_index()

    comparison_output_path = (
        OUTPUT_DIRECTORY / "per_class_f1_comparison.csv"
    )

    f1_comparison.to_csv(comparison_output_path, index=False)

    print("\nSaved per-class metrics to: "
          f"{long_output_path.relative_to(PROJECT_ROOT)}")

    print(
        "Saved per-class F1 comparison to: "
        f"{comparison_output_path.relative_to(PROJECT_ROOT)}"
    )

    return per_class_table

def calculate_common_miscassifications(
    model_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Identify and rank the most common true-class to predicted-class
    errors for each model.
    """

    all_error_tables: list[pd.DataFrame] = []

    print("\nCalculating common misclassification pairs...")

    for model_name, data in model_data.items():
        errors = data[
            data["true_label"] != data["predicted_label"]
        ].copy()

        error_counts = (
            errors.groupby(
                ["true_label", "predicted_label"]
            )
            .size()
            .reset_index(name="error_count")
        )

        error_counts = error_counts.sort_values(
            "error_count",
            ascending=False,
        ).reset_index(drop=True)

        error_counts.insert(
            0,
            "model",
            model_name,
        )

        error_counts.insert(
            1,
            "rank",
            range(1, len(error_counts) + 1),
        )

        error_counts["percentage_of_model_errors"] = (
            error_counts["error_count"]
            / len(errors)
            * 100
        )

        all_error_tables.append(error_counts)

        print(f"\n{model_name}:")
        print(
            f" Total incorrect predictions: "
            f"{len(errors):,}"
        )
        print(" Five most common error pairs:")

        top_five = error_counts.head(5)

        for _, row in top_five.iterrows():
            print(
                f" {int(row['rank'])}. "
                f"{row['true_label']} predicted as "
                f"{row['predicted_label']}: "
                f"{int(row['error_count'])} images "
                f"({row['percentage_of_model_errors']:.2f}% "
                "of model errors)"
            )

    # This section runs after all three models
    misclassification_table = pd.concat(
        all_error_tables,
        ignore_index=True,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "common_misclassifications.csv"
    )

    misclassification_table.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved common misclassifications to: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )

    return misclassification_table

def analyse_prediction_confidence(
        model_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse confidence for correct and incorrect predictions and 
    identify the highest-confidence mistakes made by each model.
    """

    high_confidence_tables: list[pd.DataFrame] = []
    confidence_rows: list[dict[str, object]] = []

    print("\nAnalysing prediction confidence...")

    for model_name, data in model_data.items():
        data = data.copy()

        data["prediction_result"] = np.where(
            data["true_label"] == data["predicted_label"],
            "Correct",
            "Incorrect",
        )

        for prediction_result in ["Correct", "Incorrect"]:
            result_data = data[
                data["prediction_result"] == prediction_result
            ]

            confidence_rows.append(
                {
                    "model": model_name,
                    "prediction_result": prediction_result,
                    "image_count": len(result_data),
                    "average_confidence": (
                        result_data["confidence"].mean()
                    ),
                    "median_confidence": (
                        result_data["confidence"].median()
                    ),
                    "minimum_confidence": (
                        result_data["confidence"].min()
                    ),
                    "maximum_confidence": (
                        result_data["confidence"].max()
                    ),
                }
            )

        errors = data[
            data["prediction_result"] == "Incorrect"
        ].copy()

        high_confidence_errors = errors.sort_values(
            "confidence",
            ascending=False,
        ).head(20)

        high_confidence_errors.insert(
            0,
            "model",
            model_name,
        )

        high_confidence_errors.insert(
            1,
            "error_rank",
            range(1, len(high_confidence_errors) + 1),
        )

        selected_columns = [
            "model",
            "error_rank",
            "relative_image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "inference_time_sec",
        ]

        high_confidence_tables.append(
            high_confidence_errors[selected_columns]
        )

        highest_error = high_confidence_errors.iloc[0]

        print(f"\n{model_name}:")
        print(
            " Highest-confidence error: "
            f"{highest_error['true_label']} predicted as "
            f"{highest_error['predicted_label']} "
            f"with confidence "
            f"{highest_error['confidence']:.4f}"
        )

        confidence_summary = pd.DataFrame(
            confidence_rows
        )

        high_confidence_error_table = pd.concat(
            high_confidence_tables,
            ignore_index=True,
        )

        confidence_output_path = (
            OUTPUT_DIRECTORY
             / "confidence_summary.csv"
    )

    errors_output_path = (
        OUTPUT_DIRECTORY
        / "high_confidence_errors.csv"
    )

    confidence_summary.to_csv(
        confidence_output_path,
        index=False,
    )

    high_confidence_error_table.to_csv(
        errors_output_path,
        index=False,
    )

    confidence_chart_data = confidence_summary.pivot(
        index="model",
        columns="prediction_result",
        values="average_confidence",
    )

    confidence_chart_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title(
        "Average Confidence for Correct and Incorrect Predictions"
    )
    plt.xlabel("Model")
    plt.ylabel("Average confidence")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    confidence_chart_path = (
        OUTPUT_DIRECTORY
        / "correct_incorrect_confidence_comparison.png"
    )

    plt.savefig(
        confidence_chart_path,
        dpi=300,
    )

    plt.close()

    inference_rows: list[dict[str, object]] = []

    for model_name, data in model_data.items():
        inference_rows.append(
            {
                "model": model_name,
                "average_inference_time_ms": (
                    data["inference_time_sec"].mean() * 1000
                ),
                "median_inference_time_ms": (
                    data["inference_time_sec"].median() * 1000
                ),
            }
        )

    inference_table = pd.DataFrame(
        inference_rows
    ).set_index("model")

    inference_table.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title("Model Inference-Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Inference time in milliseconds")
    plt.xticks(rotation=0)
    plt.tight_layout()

    inference_chart_path = (
        OUTPUT_DIRECTORY
        / "inference_time_comparison.png"
    )

    plt.savefig(
        inference_chart_path,
        dpi=300,
    )

    plt.close()

    print(
        "\nSaved confidence summary to: "
        f"{confidence_output_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        "Saved high-confidence errors to: "
        f"{errors_output_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        "Saved confidence chart to: "
        f"{confidence_chart_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        "Saved inference-time chart to: "
        f"{inference_chart_path.relative_to(PROJECT_ROOT)}"
    )

    return confidence_summary, high_confidence_error_table


def verify_analysis_outputs() -> None:
    """
    Confirm that all expected final-analysis outputs exist and contain data.
    """

    expected_outputs = [
        "overall_metrics.csv",
        "per_class_metrics_long.csv",
        "per_class_f1_comparison.csv",
        "common_misclassifications.csv",
        "confidence_summary.csv",
        "high_confidence_errors.csv",
        "correct_incorrect_confidence_comparison.png",
        "inference_time_comparison.png",
    ]

    print("\nVerifying final analysis outputs...")

    missing_files: list[str] = []
    empty_files: list[str] =[]

    for filename in expected_outputs:
        output_path = OUTPUT_DIRECTORY / filename

        if not output_path.exists():
            missing_files.append(filename)
            continue

        if output_path.stat().st_size == 0:
            empty_files.append(filename)
            continue

        print(
            f" Verified: {filename} "
            f"({output_path.stat().st_size:,} bytes)"
        )


    if missing_files:
        raise FileNotFoundError(
            "Missing analysis outputs: "
            f"{missing_files}"
        )

    if empty_files:
        raise ValueError(
            "Empty analysis outputs: "
            f"{empty_files}"
        )

    print(
        "\nAll eight final-analysis outputs were created "
        "successfully."
    )

def main() -> None:
    inspect_input_files()
    model_data = load_and_validate_model_data()

    calculate_overall_metrics(model_data=model_data)
    calculate_per_class_metrics(model_data=model_data)
    calculate_common_miscassifications(model_data=model_data)
    analyse_prediction_confidence(model_data=model_data)
    verify_analysis_outputs()

if __name__ == "__main__":
    main()


