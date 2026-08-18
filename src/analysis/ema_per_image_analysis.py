from pathlib import Path
import pandas as pd


# Project root
project_root = Path(r"C:\Projects\TM470")


# Primary model per-image CSV files
baseline_path = (
    project_root
    / "results"
    / "baseline"
    / "run_10062026_191445"
    / "baseline_per_image_predictions.csv"
)

resnet_path = (
    project_root
    / "results"
    / "resnet18"
    / "run_10062026_201112"
    / "resnet18_per_image_predictions.csv"
)

mobilenet_path = (
    project_root
    / "results"
    / "MobileNetV2"
    / "run_12062026_193343"
    / "MobileNetV2_per_image_predictions.csv"
)


# Load results
baseline = pd.read_csv(baseline_path)
resnet = pd.read_csv(resnet_path)
mobilenet = pd.read_csv(mobilenet_path)


# Combine into one DataFrame
combined = pd.concat(
    [baseline, resnet, mobilenet],
    ignore_index=True
)


# Check the combined data
print("\nPredictions per model:")
print(combined["model_name"].value_counts())


print("\nAverage confidence by model and correctness:")
confidence_summary = (
    combined
    .groupby(["model_name", "correct"])["confidence"]
    .mean()
)

print(confidence_summary)


# Create EMA analysis output folder
output_dir = project_root / "results" / "ema_per_image_analysis"
output_dir.mkdir(parents=True, exist_ok=True)


# Save combined CSV
output_file = output_dir / "combined_per_image_results.csv"

combined.to_csv(output_file, index=False)

print(f"\nCombined results saved to:")
print(output_file)



import matplotlib.pyplot as plt


# Convert inference time from seconds to milliseconds
combined["inference_time_ms"] = combined["inference_time_sec"] * 1000


# Make model names more readable on the graph
combined["display_model"] = combined["model_name"].replace({
    "BaselineCNN": "Baseline CNN",
    "ResNet18": "ResNet18",
    "MobileNetV2": "MobileNetV2"
})


# Create scatterplot
fig, ax = plt.subplots(figsize=(11, 7))

markers = {
    True: "o",
    False: "x"
}

for model in combined["display_model"].unique():

    model_data = combined[
        combined["display_model"] == model
    ]

    for correct_value in [True, False]:

        subset = model_data[
            model_data["correct"] == correct_value
        ]

        ax.scatter(
            subset["inference_time_ms"],
            subset["confidence"],
            marker=markers[correct_value],
            alpha=0.45,
            label=(
                f"{model} - "
                f"{'Correct' if correct_value else 'Incorrect'}"
            )
        )


ax.set_title(
    "Prediction Confidence vs Inference Time by Model"
)

ax.set_xlabel("Inference Time (ms)")
ax.set_ylabel("Prediction Confidence")

ax.set_ylim(0, 1)

ax.grid(alpha=0.2)

ax.legend(
    fontsize=8,
    title="Model / Prediction"
)

plt.tight_layout()


# Save figure
scatter_file = (
    output_dir
    / "confidence_vs_inference_time_scatter.png"
)

plt.savefig(
    scatter_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\nScatterplot saved to:")
print(scatter_file)



print("\nConfidence vs inference-time correlation:")

for model, group in combined.groupby("model_name"):
    correlation = group["confidence"].corr(
        group["inference_time_ms"]
    )
    
    print(f"{model}: {correlation:.3f}")