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

def main() -> None:
    inspect_input_files()
    model_data = load_and_validate_model_data()

    calculate_overall_metrics(model_data=model_data)

if __name__ == "__main__":
    main()


