from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize



# ----------------------------------------------------
# PROJECT SETTINGS
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "Baseline CNN":(
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "baseline_ema_predictions.csv"
    ),
    "ResNet18 Corrected": (
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "resnet18_corrected_ema_probabilities.csv"
    ),
     "MobileNetV2": (
        PROJECT_ROOT
        / "results"
        / "ema_probabilities"
        / "mobilenetv2_ema_predictions.csv"
    ),
}

RUN_ID = time.strftime("%d%m%Y_%H%M%S")

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "ema_roc_analysis"
    / f"run_{RUN_ID}"
)

COMMON_FPR = np.linspace(
        0.0,
        1.0,
        1000,
    )

# --------------------------------------------------
# DATA LOADING AND VALIDATION
# --------------------------------------------------

def load_model_data(
        model_name: str,
        csv_path: Path,
) -> tuple[pd.Dataframe, list[str], np.ndarry, np.ndarray]:
    """
    Load one model's probability CSV and validate that it contains 
    the expected rows, labels and class-probability columns.    
    """

    print (f"\nLoading {model_name}")
    print(f"File: {csv_path.relative_to(PROJECT_ROOT)}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Probability CSV does not exist: {csv_path}"
        )

    data = pd.read_csv(csv_path)

    required_columns = {
        "true_label",
        "predicted_label",
        "confidence", 
        "model_name",
        "split",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"{model_name} is missing required columns: "
            f"{sorted({missing_columns})}"
        )

    probability_columns = [
        column
        for column in data.columns
        if column.startswith("prob_")
    ]

    if len(probability_columns) != 11:
        raise ValueError(
            f"{model_name} contains "
            f"{len(probability_columns)} probability columns; "
            "11 were expected."
        )

    class_names = [
        column.removeprefix("prob_")
        for column in probability_columns
    ]

    if len(data) != 1175:
        raise  ValueError(
            f"{model_name} contains {len(data)} rows; "
            "1,175 were expected."
        )
    data["true_label"] = data["true_label"].astype(str)
    data["predicted_label"] = data["predicted_label"].astype(str)


    unknown_true_label = sorted(
        set(data["true_label"]).difference(class_names)
    )

    if unknown_true_label:
        raise ValueError(
            f"{model_name} contains unknown true labels: "
            f"{unknown_true_label}"
        )

    probability_score = (
        data[probability_columns]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )

    probability_sums = probability_score.sum(axis=1)

    maximum_probability_error = float(
        np.abs(probability_sums - 1.0).max()
    )

    if maximum_probability_error > 1e-5:
        raise ValueError(
            f"{model_name} probability rows do not sum to 1."
        )

    binary_true_label = label_binarize(
        data["true_label"], 
        classes=class_names,
    )

    if binary_true_label.shape != probability_score.shape:
        raise ValueError(
            f"{model_name} label and probability shapes do not match: "
            f"{binary_true_label.shape} vs "
            f"{probability_score.shape}"
        )

    print(f"Rows: {len(data):, }")
    print(f"Classes: {len(class_names)}")
    print(
        "Maximum probability-sum error: "
        f"{maximum_probability_error:.10f}"
    )

    return(
        data,
        class_names,
        binary_true_label,
        probability_score,
    )

# --------------------------------------------------
# ROC AND AUC CALCULATION
# --------------------------------------------------

def calculate_roc_metrics(
        model_name: str,
        class_names: list[str],
        binary_true_label: np.ndarray,
        probability_scores: np.ndarray,
)-> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    pd.DataFrame,
]:

    """
    Calculate one-vs-rest ROC curves for every class, together
    with micro-average and macro-average ROC curves.
    """

    false_positive_rates: dict[str, np.ndarray] = {}
    true_positive_rates: dict[str, np.ndarray] = {}
    auc_values: dict[str, float] = {}

    class_rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(
            binary_true_label[:, class_index],
            probability_scores[:, class_index],
        )

        class_auc = float(auc(fpr, tpr))

        false_positive_rates[class_name] = fpr
        true_positive_rates[class_name] = tpr
        auc_values[class_name] = class_auc

        class_rows.append(

            {
                "model": model_name,
                "class_name": class_name,
                "auc": class_auc,
                
            }
        )

        micro_fpr, micro_tpr, _ = roc_curve(
            binary_true_label.ravel(),
            probability_scores.ravel(),
        )

        micro_auc = float (
            auc(micro_fpr, micro_tpr)
        )

        false_positive_rates["micro"] = micro_fpr
        true_positive_rates["micro"] = micro_tpr
        auc_values["micro"] = micro_auc

        interpolated_tprs = []

        for class_name in class_names:
            interpolated_tpr = np.interp(
                COMMON_FPR,
                false_positive_rates[class_name],
                true_positive_rates[class_name],
            )

            interpolated_tpr[0] = 0.0
            interpolated_tprs.append(interpolated_tpr)

        macro_tpr = np.mean(
            interpolated_tprs,
            axis=0,
        )

        macro_tpr[-1] = 1.0

        macro_auc = float(

            auc(COMMON_FPR, macro_tpr)
        )

        false_positive_rates["macro"] = COMMON_FPR
        true_positive_rates["macro"] = macro_tpr
        auc_values["macro"] = macro_auc

        class_auc_table = pd.DataFrame(class_rows)

        print(f"Micro-averahe AUC: {micro_auc:.4f}")
        print(f"Macro-average AUC: {macro_auc:.4f}")

        return(
            false_positive_rates,
            true_positive_rates,
            auc_values,
            class_auc_table,
        )



# ------------------------------------------------------
# INDIVIDUAL MODEL ROC CHART
# ------------------------------------------------------

def plot_model_roc(
        model_name: str,
        class_names: list[str],
        false_positive_rates: dict[str, np.ndarray],
        true_positive_rates: dict[str, np.ndarray],
        auc_values: dict[str, float],
        output_directory: Path,
)-> None:
    """
    Create one ROC chart showing the 11 one-vs-rest class curves, 
    together with te model's micro-average and macro-average curves.
    """

    plt.figure(figsize= (12, 9))

    for class_name in class_names:
        plt.plot(
            false_positive_rates[class_name],
            true_positive_rates[class_name],
            linewidth=1.2,
            label=(
                f"{class_name} "
                f"(AUC = {auc_values[class_name]:.3f})"
            ),
        )

    plt.plot(
        false_positive_rates["micro"],
        true_positive_rates["micro"],
        linestyle="--",
        linewidth = 2.5,
        label=(
            "Micro-average "
            f"(AUC = {auc_values['micro']:.3f})"
        ),
    )

    plt.plot(
        false_positive_rates["macro"],
        true_positive_rates["macro"],
        linestyle=":",
        lineewidth=2.5,
        label=(
            "Macro-average "
            f"(AUC = {auc_values['macro']:.3f})"
        ),
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name}: One-vs-Rest ROC Curves"
    )

    plt.legend(
        loc="lower right",
        fontsize=8,
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_model_name = (
        model_name
        .lower()
        .replaced(" ", "_")
    )

    output_path = (
        output_directory
        /f"{safe_model_name}_roc_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved ROC chart: "
        f"{output_directory.relative_to(PROJECT_ROOT)}"
    )

# ------------------------------------------------------
# COMBINED AVERAGE ROC COMPARISON
# ------------------------------------------------------

def plot_average_comparison(
        average_type: str,
        model_roc_results: dict[
            str,
            tuple[
                dict[str, np.ndarray],
                dict[str, np.ndarray],
                dict[str, float],
        ],
    ],
    output_directory: Path,
)-> None:
    """
    Compare either the micro-average or macro average ROC curve 
    across all three final primary models.
    """

    if average_type not in {"micro", "macro"}:
        raise ValueError(
            "average_type must be either 'micro' or 'macro'."
        )

    plt.figure(figsize=(10,7))

    for model_name, (
        false_positive_rates,
        true_positive_rates,
        auc_values,
    ) in model_roc_results.items():
        plt.plot(

            false_positive_rates[average_type],
            true_positive_rates[average_type],
            linewidth=2.5,
            labe=(
                f"{model_name} "
                f"(AUC = {auc_values[average_type]:.3f})"
            ),
            
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle= "--",
            linewidth=1,
            label="Random classifier",
        )

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Postive Rate")

        display_name = average_type.capitalize()

        plt.title(
            f"{display_name} - Average ROC Comparsion"

        )

        plt.legend(
            loc="lower right",
            fontsize=9,
        )

        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = (
            output_directory
            / f"{average_type}_average_roc_comparison.png"
        )


        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved comparison chart: "
            f"{output_path.relative_to(PROJECT_ROOT)}"
        )

# ----------------------------------------------------------
# AUC SUMMARY TABLES
# ----------------------------------------------------------

def save_auc_tables(
        class_auc_tables: list[pd.DataFrame],
        model_roc_results: dict[
            str,
            tuple[
                dict[str, np.ndarray],
                dict[str, np.ndarray],
                dict[str, float],
            ],
        ],
        output_directory: Path,
)-> None:
    """
    Save detailed per-class AUC results and a summary containing the micro-average
    macro-average, strongest and weakest class results for each model.
    """

    combined_class_auc = pd.concat(
        class_auc_tables,
        ignore_index=True,
    )

    long_table_path = (
        output_directory
        / "per_class_auc_long.csv"
    )

    combined_class_auc.to_csv(
        long_table_path,
        index=False,
    )

    comparison_table = combined_class_auc.pivot(
        index="class_name",
        columns="model",
        values="auc",
    ).reset_index()

    comparison_table_path = (
        output_directory
        /"per_class_auc_comparison.csv"
    )

    comparison_table.to_csv(
        comparison_table_path,
        index=False,
    )

    summary_rows: list[dict[str, object]] = []

    for model_name, (
        _,
        _,
        auc_values,
    ) in model_roc_results.items():
        class_auc_values = {
            class_name: auc_value
            for class_name, auc_value in auc_values.items()
            if class_name not in {"micro", "macro"}
        }

        strongest_class = max(
            class_auc_values,
            key = class_auc_values.get,
        )

        weakest_class = min(
            class_auc_values,
            key = class_auc_values.get,
        )

        summary_rows.append(
            {
                "model": model_name,
                "micro_average_auc": auc_values["micro"],
                "macro_average_auc": auc_values["macro"],
                "strongest_class": strongest_class,
                "strongest_class_auc": (
                    class_auc_values[strongest_class]
                ),
                "weakest_class": weakest_class,
                "weakest_class_auc":(
                    class_auc_values[weakest_class]
                ),
            }
        )

        summary_tables = pd.DataFrame(summary_rows)

        summary_table_path = (
            output_directory
            / "auc.summary.csv"
        )

        "summary_table".to_csv(
            summary_table_path,
            index = False,
        )

        print(
            "Saved AUC table: "
            f"{long_table_path.relative_to(PROJECT_ROOT)}"
        )

        print(
            "Saved AUC comparison: "
            f"{comparison_table_path.relative_to(PROJECT_ROOT)}"
        )

        print("Saved AUC summary: "
              f"{summary_table_path.relative_to(PROJECT_ROOT)}"
        )

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main() -> None: 
    print("Starting EMA ROC and AUC analysis.")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "Saving outputs to: "
        f"{OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT)}"
    )

    model_roc_results: dict[
        str, tuple[
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            dict[str, float],
        ],
    ] = {}

    class_auc_tables: list[pd.DataFrame] = []

    expected_class_names: list[str] | None = None

    for model_name, csv_path in MODEL_FILES.items():
        (
            _,
            class_names,
            binary_true_label,
            probability_scores,           
        ) = load_model_data(
            model_name=model_name,
            csv_path=csv_path,
        )

        if expected_class_names is None:
            expected_class_names = class_names
        elif class_names != expected_class_names:
            raise ValueError(
                f"{model_name} usea a different class order."
            )

        (
            false_positive_rates,
            true_positive_rates,
            auc_values,
            class_auc_table,
        ) = calculate_roc_metrics(

            model_name=model_name,
            class_names=class_names,
            binary_true_label=binary_true_label,
            probability_scores=probability_scores,
        )

        plot_model_roc(
            model_name=model_name,
            class_names=class_names,
            false_positive_rates=false_positive_rates,
            true_positive_rates=true_positive_rates,
            auc_values=auc_values,
            output_directory=OUTPUT_DIRECTORY,
        )

        model_roc_results[model_name] = (
            false_positive_rates,
            true_positive_rates,
            auc_values,
        )

        class_auc_tables.append(class_auc_table)

        plot_average_comparison(
            average_type="micro",
            model_roc_results= model_roc_results,
            output_directory=OUTPUT_DIRECTORY,
        )

        plot_average_comparison(
            average_type="macro",
            model_roc_results=model_roc_results,
            output_directory=OUTPUT_DIRECTORY,
        )

        save_auc_tables(
            class_auc_tables=class_auc_table,
            model_roc_results=model_roc_results,
            output_directory=OUTPUT_DIRECTORY,
        )

        print("\n" + "=" * 80)
        print("ROC and AUC analysis completed successfully.")
        print(
            "Results saved to: "
            f"{OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main()