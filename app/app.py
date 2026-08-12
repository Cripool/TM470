"""
Streamlit mushroom image-recognition demonstrator for the TM470 EMA.
"""

from pathlib import Path
import time
from datetime import datetime, timezone
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import uuid
from supabase import create_client, Client

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
def get_supabase_client() -> Client:
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )

def save_test_result(test_result: dict):
    supabase = get_supabase_client()

    response = (
        supabase
        .table("user_test_results")
        .insert(test_result,
                returning="minimal",)
        .execute()
    )

    return response
def save_survey_result(survey_result: dict):
    supabase = get_supabase_client()

    response = (
        supabase
        .table("user_testing_surveys")
        .insert(
            survey_result,
            returning="minimal",
        )
        .execute()
    )

    return response


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

if "participant_stage" not in st.session_state:
    st.session_state.participant_stage = "information"

if "user_id" not in st.session_state:
    existing_user_id = st.query_params.get("participant")

    if existing_user_id:
        st.session_state.user_id = existing_user_id
        st.session_state.participant_stage = "testing"
    else:
        st.session_state.user_id = None

if "test_count" not in st.session_state:
    st.session_state.test_count = 0

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "test_complete" not in st.session_state:
    st.session_state.test_complete = False

if st.session_state.participant_stage == "information":

    st.title("Participant Information")

    st.write(
        "Please read the participant information below before "
        "continuing to the consent form."
    )

    # Insert the final Participant Information Sheet content here.

    st.info(
        "The full Participant Information Sheet will be displayed here."
    )

    if st.button("Continue to Consent"):
        st.session_state.participant_stage = "consent"
        st.rerun()

    st.stop()

if st.session_state.participant_stage == "consent":

    st.title("Consent Form")

    read_information = st.checkbox(
        "I have read and understand the Participant Information."
    )

    age_confirmation = st.checkbox(
        "I confirm I am aged 18 or over."
    )

    voluntary_confirmation = st.checkbox(
        "I understand that my participation is voluntary."
    )

    data_confirmation = st.checkbox(
        "I consent to the collection of the testing data "
        "described in the Participant Information."
    )

    consent_given = all(
        [
            read_information,
            age_confirmation,
            voluntary_confirmation,
            data_confirmation,
        ]
    )

    if st.button(
        "Consent and Begin Testing",
        disabled=not consent_given,
    ):
        if st.session_state.user_id is None:
            st.session_state.user_id = (
                f"User-{uuid.uuid4().hex[:8].upper()}"
            )

        st.query_params["participant"] = st.session_state.user_id
        st.session_state.participant_stage = "testing"
        st.rerun()

    st.stop()

if st.session_state.participant_stage == "survey":

    st.title("Usability Survey")

    st.write(
        "Thankyou for completing the image testing. "
        "Please answer the following questions based on your "
        "experience using the application."
    )

    st.write(
        "**1 = Strongly disagree | 2 = Disagree | "
        "3 = Neither agree nor disagree | 4 = Agree | "
        "5 = Strongly agree**"
    )

    with st.form("usability_survey"):

        ease_of_use = st.radio(
            "1. The Application was easy to use.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        instructions_clear = st.radio(
            "2. The instructions within the application were clear.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        upload_process = st.radio(
            "3. Uploading and testing an image was straightforward.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        predictions_clear = st.radio(
            "4. The model predictions were presented clearly.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        confidence_clear = st.radio(
            "5. The confidence scores were easy to understand.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        processing_speed = st.radio(
            "6. The application responded quickly enough when testing an image.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        navigation_clear = st.radio(
            "7. It was clear how to test another image or finish testing.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        overall_usability = st.radio(
            "8. Overall, I was satisfied with the usability of the application.",
            [1, 2, 3, 4, 5],
            index= None,
            horizontal= True,
        )

        experienced_errors = st.radio(
            " Did you experience any errors or problems while using the application?",
            ["No", "Yes"],
            index=None,
            horizontal=True,
        )

        comments = st.text_area(
            "Additional comments or suggestions (optional)",
            max_chars=1000,
        )

        submit_survey = st.form_submit_button(
            "Submit Survey",
            type="primary",
        )


        if submit_survey:

            required_answers = [
                ease_of_use,
                instructions_clear,
                upload_process,
                predictions_clear,
                confidence_clear,
                processing_speed,
                navigation_clear,
                overall_usability,
                experienced_errors,
            ]

            if any(answer is None for answer in required_answers):

                st.warning(
                    "Please answer all required questions before "
                    "submitting the survey."
                )
            else:

                survey_result = {
                    "user_id": st.session_state.user_id,
                    "ease_of_use": ease_of_use,
                    "instructions_clear": instructions_clear,
                    "upload_process": upload_process,
                    "predictions_clear": predictions_clear,
                    "confidence_clear": confidence_clear,
                    "processing_speed": processing_speed,
                    "navigation_clear": navigation_clear,
                    "overall_usability": overall_usability,
                    "experienced_errors": experienced_errors == "Yes",
                    "comments": comments.strip() or None,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }

                try:
                    save_survey_result(survey_result)

                except Exception as error:
                    st.error(
                        "The survey could not be saved. "
                        "Please try again."
                    )

                    st.exception(error)
                    st.stop()

                st.session_state.participant_stage = "complete"

                st.rerun()

        st.stop()

if st.session_state.participant_stage == "complete":

    st.title("Testing Complete")

    st.success(
        "Thankyou for taking part in the application testing."
    )

    st.write(
        "Your image-testing results and usability survey "
        "have been successfully submitted."
    )

    st.write(
        f"Your Participation ID is: **{st.session_state.user_id}**"
    )

    st.info(
        "Please keep your Participant ID if you wish to "
        "request withdrawal of your data within the stated "
        "withdrawal period."
    )

    st.stop()


    st.info(
        "The usability survey will be added here next."
    )

    st.stop()

st.title(
    "CNN Mushroom Recognition Comparison"
)

st.subheader("Your Participant ID")

st.code(
    st.session_state.user_id,
    language=None,
)

st.caption(
    "Please keep a copy of this ID. It can be used if you "
    "wish to request withdrawal of your testing data."
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

if not st.session_state.test_complete:

    uploaded_file = st.file_uploader(
        "Upload an image for model comparison",
        type=["jpg", "jpeg", "png"],
        key=f"uploaded_file_{st.session_state.test_count}",
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

        image_type = st.radio(
            "What type of image are you testing?",
            [
                "Mushroom",
                "Non-mushroom",
                "Unsure",
            ],
            index= None,
            horizontal=True,
        )

        if st.button(
            "Run Test",
            type="primary",
            disabled=image_type is None,
        ):
            test_id = (
                f"TST-{uuid.uuid4().hex[:8].upper()}"
            )

            st.write(
                f"Test ID: {test_id}"
            )

            st.session_state.test_count += 1

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

            test_result = {
                "user_id": st.session_state.user_id,
                "test_id": test_id,
                "image_type": image_type,
                "baseline_prediction": baseline_prediction,
                "baseline_confidence": baseline_confidence,
                "baseline_inference_time_ms": baseline_time_ms,
                "resnet18_prediction": resnet_prediction,
                "resnet18_confidence": resnet_confidence,
                "resnet18_inference_time_ms": resnet_time_ms,
                "mobilenetv2_prediction": mobilenet_prediction,
                "mobilenetv2_confidence": mobilenet_confidence,
                "mobilenetv2_inference_time_ms": mobilenet_time_ms,
                "total_app_processing_time_ms": total_processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                
            }

            try:
                save_test_result(test_result)

            except Exception as error:
                st.error(
                    "The test result could not be saved. "
                    "Please try again."
                )

                st.exception(error)
                st.stop()

            st.session_state.test_count += 1
            st.session_state.latest_result = test_result
            st.session_state.test_complete = True
            st.session_state.latest_image = image
            st.rerun()


if(
    st.session_state.test_complete
    and st.session_state.latest_result is not None
):
            result = st.session_state.latest_result

            if st.session_state.latest_image is not None:
                st.image(
                    st.session_state.latest_image,
                    caption="Tested image",
                    width= 400,
                )

            st.success(
                f"Test completed successfully - {result['test_id']}"
            )

            st.write(
                f"Image type: **{result['image_type']}**"
            )

            st.subheader("Model predictions")

            baseline_column, resnet_column, mobilenet_column = st.columns(3)

            with baseline_column:
                st.markdown("### Baseline CNN",
                            help=( "The baseline Convolutional Neural Network (CNN) used as the "
                                   "reference model for this academic project. It uses a simple "
                                   "CNN architecture without the specialised features found in "
                                   "ResNet18 or MobileNetV2."),
                            )

                if result["baseline_prediction"] is not None:
                    st.markdown(
                        "**Prediction**",
                        help=(
                            "Represents the Model's best guess at what species of mushroom this image is."
                        ),
                    )
                    st.success(
                        f"Closest trained class: {result['baseline_prediction']}"
                    )

                    st.metric(
                        "Confidence",
                        f"{result['baseline_confidence'] *100:.2f}%",
                        help=(
                                "How strongly the model favours this prediction compared "
                                "with the other trained classes. A high confidence score "
                                "does not guarantee that the prediction is correct."
                            ),
                    )

                    st.metric(
                        "Inference time",
                        f"{result['baseline_inference_time_ms']:.2f} ms",
                        help=(
                                "The time taken by this model to process the image and "
                                "produce its prediction, measured in milliseconds."
                            ),
            )
                else:
                    st.error("Predicition Unavailable")

            with resnet_column:
                st.markdown("### ResNet18",
                            help=(
                                    "A deeper Convolutional Neural Network architecture that uses "
                                    "residual or skip connections. These connections help information "
                                    "flow through the network and support training of deeper models."
                                ),
                    )

                if result["resnet18_prediction"] is not None:
                    st.markdown(
                        "**Prediction**",
                        help=(
                            "Represents the Model's best guess at what species of mushroom this image is."
                        ),
                    )

                    st.success(
                        f"Closest trained class: {result['resnet18_prediction']}"
                    )

                    st.metric(
                        "Confidence",
                        f"{result['resnet18_confidence'] * 100:.2f}%",

                         help=(
                                "How strongly the model favours this prediction compared "
                                "with the other trained classes. A high confidence score "
                                "does not guarantee that the prediction is correct."
                            ),
                    )

                    st.metric(
                        "inference time",
                        f"{result['resnet18_inference_time_ms']:.2f} ms",
                        help=(
                                "The time taken by this model to process the image and "
                                "produce its prediction, measured in milliseconds."
                            ),
                    )

                else:
                    st.error("Prediction Unavailable")


            with mobilenet_column:
                st.markdown("### MobileNetV2",

                    help=(
                    "A Convolutional Neural Network architecture designed to be more "
                    "computationally efficient. It uses specialised convolution methods "
                    "to reduce processing requirements while still performing image recognition."
                    ),
                )

                if result["mobilenetv2_prediction"] is not None:

                    st.markdown(
                        "**Prediction**",
                        help=(
                            "Represents the Model's best guess at what species of mushroom this image is."
                        ),
                    )

                    st.success(
                        f"Closest trained class: {result['mobilenetv2_prediction']}"
                    )

                    st.metric(
                        "Confidence",
                        f"{result['mobilenetv2_confidence'] * 100:.2f}%",

                        help=(
                                "How strongly the model favours this prediction compared "
                                "with the other trained classes. A high confidence score "
                                "does not guarantee that the prediction is correct."
                            ),

                    )

                    st.metric(
                        "inference time",
                        f"{result['mobilenetv2_inference_time_ms']:.2f} ms",

                        help=(
                                "The time taken by this model to process the image and "
                                "produce its prediction, measured in milliseconds."
                            ),
                    )
                else: 
                    st.error("Prediction Unavailable")

            st.markdown("---")

            st.metric(
                "Total Processing Time",
                f"{result['total_app_processing_time_ms']:.2f} ms",

                help=(
                    "The total time taken by the application to process the uploaded "
                    "image through all three AI models and produce their predictions. "
                    "Measured in milliseconds."
                ),
            )

            another_column, finish_column = st.columns(2)

            with another_column:
                if st.button(
                    "Test Another Image",
                    type="primary"
                ):
                    st.session_state.test_complete = False
                    st.session_state.latest_result = None
                    st.session_state.latest_image = None

                    st.rerun()

            with finish_column:
                if st.button(
                    "Finish Testing",
                ):
                    
                    st.session_state.test_complete = False
                    st.session_state.latest_result = None
                    st.session_state.latest_image = None
                    st.session_state.participant_stage = "survey"

                    st.rerun()
                    
                


            with st.expander("Debug: Recorded Test Data"):
                st.json(result)