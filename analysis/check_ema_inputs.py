from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRECTORY = PROJECT_ROOT / "results"

FIELD_ALIASES = {

    "image_path": {"image_path", "path", "file_path", "filename", "image"},
    "true_label": {"true_label", "actual_label", "actual", "target"},
    "predicted_label": {"predicted_label", "predicted", "prediction", "prediction_label"},

    "correct": {"correct", "correct_flag", "is_correct",},
    "confidence":{ "confidence", "confidence_score", "top_confidence", "prediction_confidence"},
    "inference_time": {"inference_time", "inference_time_sec", "inference_time_ms", "prediction_time",},
    "model_name": {"model_name", "model"},

    }

def field_present(columns: set[str], aliases: set[str]) -> bool:
    """
    Check if any of the aliases are present in the columns.
    """
    return bool(columns.intersection(aliases))

def inspect_csv(csv_path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {csv_path.relative_to(PROJECT_ROOT)}")

    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as error:
        print(f"ERROR: Could not read CSV file. {error}")
        return
    
    normalised_columns = {
        str(column).strip().lower() for column in dataframe.columns
    }

    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")
    print("\nColumn names:")
    for column in dataframe.columns:
        print(f"  - {column}")

    print("\nrequired field check:")
    for field_name, aliases in FIELD_ALIASES.items():
        status = "FOUND" if field_present(normalised_columns, aliases) else "MISSING"
        print(f"  - {field_name}: {status}")
    
    probability_columns = [
        column
        for column in dataframe.columns
        if str(column).strip().lower().startswith(
            ("prob_", "probability_", "class_probability_", "score_")
        )
    ]

    print(f"\nDetected full-class probability columns: {len(probability_columns)}")

    if probability_columns:
        print("Probability columns:")
        for column in probability_columns:
            print(f"  - {column}")

    else: 
        print(
            "No full-class probability columns detected. "
            "This CSV may not be sufficient for multiclass ROC Analysis.")
            
    duplicate_count = 0
    for image_column in FIELD_ALIASES["image_path"]:
        if image_column in normalised_columns:
            original_column = next(
                column
                for column in dataframe.columns
                if str(column).strip().lower() == image_column
            )
            duplicate_count = dataframe[original_column].duplicated().sum()
            break
    print(f"\nDuplicated image paths: {duplicate_count}")


    missing_values = dataframe.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("\nMissing values: None detected.")
    else:
        print("Missing Values:")
        for column, count in missing_values.items():
            print(f" - {column}: {count}")

def main() -> None:
    if not RESULT_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {RESULT_DIRECTORY}"
        )
    
    csv_files = sorted(path for path in RESULT_DIRECTORY.rglob("*.csv")
                            if "per_image" in path.name.lower()
    )

    if not csv_files:
        print("No per-image CSV files were found.")
        return
    
    print(f"Found {len(csv_files)} per-image CSV file(s).")

    for csv_path in csv_files:
        inspect_csv(csv_path)

if __name__ == "__main__":
    main()
