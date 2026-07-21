from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -------------------------------------------------------------
# PROJECT SETTINGS
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "test"
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "ema_probabilities"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
IMAGE_SIZE = (224, 224)

NORMALISATION_MEAN = [0.485, 0.456, 0.406]
NORMALISATION_STD = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# MODEL DEFINITIONS
# -------------------------------------------------------------

class BaselineCNN(nn.Module):
    """Three-layer Baseline CNN used in the original experiment."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)
    

def build_baseline(num_classes: int) -> nn.Module:
    return BaselineCNN(num_classes)

def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenetv2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
      model.classifier[1].in_features, 
      num_classes,
    )
    return model

MODEL_CONFIGURATIONS = {
    "BaselineCNN": {
        "builder": build_baseline,
        "checkpoint": (
            PROJECT_ROOT
            / "results"
            / "baseline"
            / "run_10062026_191445"
            / "baseline_model.pth"
        ),
        "original_csv": (
            PROJECT_ROOT
            / "results"
            / "baseline"
            / "run_10062026_191445"
            / "baseline_per_image_predictions.csv"
        ),
        "output_file": "baseline_ema_predictions.csv",
    },
    "ResNet18": {
        "builder": build_resnet18,
        "checkpoint": (
            PROJECT_ROOT
            / "results"
            / "resnet18"
            / "run_10062026_201112"
            / "resnet18_model.pth"
        ),
        "original_csv": (
            PROJECT_ROOT
            / "results"
            / "resnet18"
            / "run_10062026_201112"
            / "resnet18_per_image_predictions.csv"
        ),
        "output_file": "resnet18_ema_predictions.csv",
    },
    "MobileNetV2": {
        "builder" : build_mobilenetv2,
        "checkpoint": (
            PROJECT_ROOT
            / "results"
            / "mobilenetv2"
            / "run_12062026_193343"
            / "mobilenetv2_model.pth"
        ),
        "original_csv": (
            PROJECT_ROOT
            / "results"
            / "MobileNetV2"
            / "run_12062026_193343"
            / "MobileNetV2_per_image_predictions.csv"
        ),
        "output_file": "mobilenetv2_ema_predictions.csv",
    },
}


# -------------------------------------------------------------
# PATH AND VALIDATION HELPERS
# -------------------------------------------------------------

def create_relative_key(path_value: object) -> str:
    """
    Produce a consistent class/filename key from old and current paths. 
    
    This allows files created on the previoust computer to be matched against the dataset now stored in C:\\Projects\\TM470.
    """

    path_text = str(path_value).replace("\\", "/")
    path_parts = [part for part in path_text.split("/") if part]

    lower_parts = [part.lower() for part in path_parts]

    if "test" in lower_parts:
        test_index = max(
            index
            for index, part in enumerate(lower_parts)
            if part == "test"
        )
        relative_parts = path_parts[test_index + 1:]
    else:
        relative_parts = path_parts[-2:]

    return "/".join(relative_parts).lower()


def load_original_predictions(csv_path: Path) -> pd.DataFrame:
    required_columns = {
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "inference_time_sec",
        "model_name",
        "split",
    }

    original = pd.read_csv(csv_path)

    missing_columns = required_columns.difference(original.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )
    
    original["relative_image_key"] = original["image_path"].map(
        create_relative_key
    )

    duplicate_count = original["relative_image_key"].duplicated().sum()

    if duplicate_count:
        raise ValueError(
            f"{csv_path.name} contains " 
            f"{duplicate_count} duplicate paths."
        )
    
    return original


# --------------------------------------------------------------------------
# PROBABILITY EXPORT
# --------------------------------------------------------------------------

def export_model_probabilities (
        model_name: str,
        model_builder: Callable[[int], nn.Module],
        checkpoint_path: Path,
        original_csv_path: Path,
        output_path: Path,
        dataset: datasets.ImageFolder,
        loader: DataLoader,
        class_names: list[str]
) ->None:
    print("\n" + "=" * 80)
    print(f"Preocssing: {model_name}")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    
    if not original_csv_path.exists():
        raise FileNotFoundError(
            f"Original CSV does not exist: {original_csv_path}"
        )
    
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_path}"
        )
    
    model = model_builder(len(class_names)).to(DEVICE)

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    probability_columns = [
        f"prob_{class_name}"
        for class_name in class_names
    ]

    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for index, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            predicted_index = int(
                torch.argmax(probabilities, dim=1).item()
            )

            true_index = int(labels.item())

            probability_values = (
                probabilities.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )


            image_path = Path(dataset.samples[index][0])

            row: dict[str, object] = {
                "image_path": str(image_path),
                "relative_image_path": (
                    image_path.relative_to(TEST_DIRECTORY).as_posix()
                ),
                "true_label": class_names[true_index],
                "predicted_label": class_names[predicted_index],
                "correct": true_index == predicted_index,
                "confidence": float(
                    probability_values[predicted_index]
                ),
                "model_name": model_name,
                "split": "test",
            }

            for class_index, probability_column in enumerate(
                probability_columns
            ):
                row[probability_column] = float(
                    probability_values[class_index]
                )

            rows.append(row)

        generated = pd.DataFrame(rows)

        generated["relative_image_key"] = generated[
            "relative_image_path"
            ].map(create_relative_key)
        
        original = load_original_predictions(original_csv_path)


        comparison = generated.merge(
            original[
                [
                    "relative_image_key",
                    "true_label",
                    "predicted_label",
                    "confidence",
                    "inference_time_sec",
                ]
            ],
            on="relative_image_key",
            how="left",
            suffixes=("", "_original"),
            validate="one_to_one",
        )

        missing_original_rows = int(
            comparison["predicted_label_original"].isna().sum()
        )

        if missing_original_rows:
            raise ValueError(
                f"{model_name}: {missing_original_rows} generated rows "
                "could not be matched to the original CSV."
            )
        
        true_label_mismatches = int(
            (

                comparison["true_label"]
                != comparison["true_label_original"]
            ).sum()
        )
        prediction_mismatches = int(
            (
                comparison["predicted_label"]
                != comparison["predicted_label_original"]
            ).sum()
        )

        confidence_difference = (
            comparison["confidence"]
            - comparison["confidence_original"]
        ).abs()

        maximum_confidence_difference = float(
            confidence_difference.max()
            )
        
        probability_sums = generated[probability_columns].sum(axis=1)
        
        maximum_probability_sum_error = float(
            np.abs(probability_sums - 1.0).max()
        )


        print(f"Rows generated: {len(generated):,}")
        print(f"True-label mismatches: {true_label_mismatches}")
        print(f"Prediction mismatches: {prediction_mismatches}")
        print(
                "Maximum confidence difference: "
                f"{maximum_confidence_difference:.10f}"
            )
        print(
                "Maximum probability-sum error: "
                F"{maximum_probability_sum_error:.10f}"
            )

        if true_label_mismatches:
                raise ValueError(
                    f"{model_name}: true labels do not match the original CSV."
                )

        if prediction_mismatches:
                mismatch_examples = comparison.loc[
                    comparison["predicted_label"]
                    != comparison["predicted_label_original"],
                    [
                        "relative_image_path",
                        "predicted_label",
                        "predicted_label_original",
                    ],
                ].head()

                print("\nPrediction mismatch examples:")
                print(mismatch_examples.to_string(index=False))

                raise ValueError(
                    f"{model_name}: predictions do not reproduce the "
                    "original checkpoint results."
                )

        if maximum_probability_sum_error > 1e-5:
                raise ValueError(
                    f"{model_name}: class probabilities do not sum to 1."
                )

        inference_time_lookup = original.set_index(
                "relative_image_key"
            )["inference_time_sec"]

        generated["inference_time_sec"] = generated[
                "relative_image_key"
            ].map(inference_time_lookup)

        output_columns = [
                "image_path",
                "relative_image_path",
                "true_label",
                "predicted_label",
                "correct",
                "confidence",
                "inference_time_sec",
                "model_name",
                "split",
                *probability_columns,
            ]

        generated[output_columns].to_csv(
                output_path,
                index=False,
            )

        print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")

def main() -> None:
        print(f"Device: {DEVICE}")
        
        if DEVICE.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        if not TEST_DIRECTORY.exists():
            raise FileNotFoundError(
                f"Test dataset does not exist: {TEST_DIRECTORY}"
            )

        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=NORMALISATION_MEAN,
                    std=NORMALISATION_STD,
                ),
            ]
        )

        test_dataset = datasets.ImageFolder(
            TEST_DIRECTORY,
            transform=transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE.type =="cuda",
        )

        class_names = test_dataset.classes

        print(f"Test images: {len(test_dataset):,}")
        print(f"Classes ({len(class_names)}): {class_names}")

        if len(test_dataset) !=1175:
            raise ValueError(
                "The test dataset does not contain the expected "
                "1,175 images."
            )

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        for model_name, configuration in MODEL_CONFIGURATIONS.items():
            export_model_probabilities(
            model_name = model_name,
            model_builder = configuration["builder"],
            checkpoint_path = configuration["checkpoint"],
            original_csv_path = configuration["original_csv"],
            output_path=(
                OUTPUT_DIRECTORY
                / configuration["output_file"]
            ),
            dataset = test_dataset,
            loader = test_loader,
            class_names = class_names,
        )
                
        print("\n" + "=" *80)
        print ("EMA probability export completed successfully.")
        print("No model training was performed.")
if __name__ == "__main__":
    main()         