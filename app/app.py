"""
Streamlit mushroom image-recognition demonstrator for the TM470 EMA.
"""

from pathlib import Path
import time

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "run_10062026_191445"
    / "baseline_model.pth"
)

RESNET_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "resnet18_corrected"
    / "run_24072026_215002"
    / "resnet18_model.pth"
)

MOBILENET_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "MobileNetV2"
    / "run_12062026_193343"
    / "MobileNetV2_model.pth"
)

CLASS_NAMES = [
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Exidia",
    "Hygrocybe",
    "Inocybe",
    "Lactarius",
    "Russula",
    "Suillus",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture used during model training.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 28 * 28,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(self, image_tensor):
        image_tensor = self.features(
            image_tensor
        )

        image_tensor = self.classifier(
            image_tensor
        )

        return image_tensor


@st.cache_resource
def load_baseline_model() -> BaselineCNN:
    """
    Load the final Baseline CNN checkpoint.
    """

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Baseline checkpoint not found at: "
            f"{BASELINE_CHECKPOINT}"
        )

    model = BaselineCNN(
        num_classes=len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(
            BASELINE_CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model

@st.cache_resource
def load_resnet_model() -> nn.Module:
    """
    Load the final corrected ResNet18 checkpoint.
    """

    if not RESNET_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Corrected ResNet18 checkpoint not found at: "
            f"{RESNET_CHECKPOINT}"
        )

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    model.load_state_dict(
        torch.load(
            RESNET_CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model

@st.cache_resource
def load_mobilenet_model() -> nn.Module:
    """
    Load the final from-scratch MobileNetV2 checkpoint.
    """

    if not MOBILENET_CHECKPOINT.exists():
        raise FileNotFoundError(
            "MobileNetV2 checkpoint not found at: "
            f"{MOBILENET_CHECKPOINT}"
        )

    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES),
    )

    model.load_state_dict(
        torch.load(
            MOBILENET_CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model

def predict_with_baseline(
    image: Image.Image,
) -> tuple[str, float, float]:
    """
    Run the uploaded image through the Baseline CNN.
    """

    model = load_baseline_model()

    image_tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return (
        predicted_class,
        confidence.item(),
        inference_time_ms,
    )

def predict_with_resnet(
    image: Image.Image,
) -> tuple[str, float, float]:
    """
    Run the uploaded image through the corrected ResNet18.
    """

    model = load_resnet_model()

    image_tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return (
        predicted_class,
        confidence.item(),
        inference_time_ms,
    )


def predict_with_mobilenet(
    image: Image.Image,
) -> tuple[str, float, float]:
    """
    Run the uploaded image through MobileNetV2.
    """

    model = load_mobilenet_model()

    image_tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return (
        predicted_class,
        confidence.item(),
        inference_time_ms,
    )

st.set_page_config(
    page_title="CNN Mushroom Recognition Comparison",
    page_icon="🍄",
    layout="wide",
)

st.title(
    "CNN Mushroom Recognition Comparison"
)

st.write(
    "Upload a mushroom image to compare predictions from the "
    "Baseline CNN, corrected ResNet18 and MobileNetV2 models. "
    "The models can only classify images into the 11 mushroom "
    "genera used during training."
)

with st.expander("Supported mushroom classes"):
    st.write(", ".join(CLASS_NAMES))

st.warning(
    "Academic demonstrator only. The models always select the closest "
    "matching class from the 11 trained mushroom genera and cannot "
    "reliably recognise unsupported mushrooms or non-mushroom images. "
    "Do not use these predictions to determine whether a mushroom is "
    "safe to touch or consume."
)

st.caption(
    f"Application device: {DEVICE}"
)

uploaded_file = st.file_uploader(
    "Upload an image for model comparison",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=400,
    )

    st.subheader("Model predictions")

    overall_start = time.perf_counter()

    try:
        (
            baseline_prediction,
            baseline_confidence,
            baseline_time_ms,
        ) = predict_with_baseline(image)

    except Exception as error:
        baseline_prediction = None
        baseline_confidence = None
        baseline_time_ms = None

        st.error(
            "The Baseline CNN could not be loaded or executed."
        )

        st.exception(error)

    try:
        (
            resnet_prediction,
            resnet_confidence,
            resnet_time_ms,
        ) = predict_with_resnet(image)

    except Exception as error:
        resnet_prediction = None
        resnet_confidence = None
        resnet_time_ms = None

        st.error(
            "The corrected ResNet18 could not be loaded or executed."
        )

        st.exception(error)

    baseline_column, resnet_column, mobilenet_column = (
        st.columns(3)
    )

    try:
        (
            mobilenet_prediction,
            mobilenet_confidence,
            mobilenet_time_ms,
        ) = predict_with_mobilenet(image)

    except Exception as error:
        mobilenet_prediction = None
        mobilenet_confidence = None
        mobilenet_time_ms = None

        st.error(
            "MobileNetV2 could not be loaded or executed."
        )

        st.exception(error)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

        total_processing_time_ms = (
            time.perf_counter() - overall_start
            ) * 1000

    with baseline_column:
        st.markdown("### Baseline CNN")

        if baseline_prediction is not None:
            st.success(
                f"Closest trained class: {baseline_prediction}"
            )

            st.metric(
                "Confidence",
                f"{baseline_confidence * 100:.2f}%",
            )

            st.metric(
                "Inference time",
                f"{baseline_time_ms:.2f} ms",
            )

        else:
            st.error("Prediction unavailable")

    with resnet_column:
        st.markdown("### ResNet18")

        if resnet_prediction is not None:
            st.success(
                f"Closest trained class: {resnet_prediction}"
            )

            st.metric(
                "Confidence",
                f"{resnet_confidence * 100:.2f}%",
            )

            st.metric(
                "Inference time",
                f"{resnet_time_ms:.2f} ms",
            )

        else:
            st.error("Prediction unavailable")

    with mobilenet_column:
        st.markdown("### MobileNetV2")

        if mobilenet_prediction is not None:
            st.success(
                f"Closest trained class: {mobilenet_prediction}"
            )

            st.metric(
                "Confidence",
                f"{mobilenet_confidence * 100:.2f}%",
            )

            st.metric(
                "Inference time",
                f"{mobilenet_time_ms:.2f} ms",
            )

        else:
            st.error("Prediction unavailable")

    st.markdown("---")

    st.metric(
        "Total Processing Time",
        f"{total_processing_time_ms:.2f} ms",
    )