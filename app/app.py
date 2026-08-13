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
import logging
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
def create_participant_record():
    supabase = get_supabase_client()

    response = (
        supabase
        .rpc("create_participant")
        .execute()
    )

    if not response.data:
        raise RuntimeError("Participant couldnot be created.")

    return response.data[0]

def resume_participant_record(resume_token: str):
    supabase = get_supabase_client()

    response = (
        supabase.rpc(
            "resume_participant",
            {"p_resume_token": resume_token},
        )
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

    
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

def save_withdrawal_request(user_id: str):
    supabase = get_supabase_client()

    response = (
    supabase
    .table("withdrawal_requests")
    .insert(
        {"user_id": user_id},
        returning="minimal"
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

if "viewing_information" not in st.session_state:
    st.session_state.viewing_information = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "resume_token" not in st.session_state:
    st.session_state.resume_token = st.query_params.get("resume")

if "completed_tests" not in st.session_state:
    st.session_state.completed_tests = 0

if "survey_submitted" not in st.session_state:
    st.session_state.survey_submitted = False


if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "test_complete" not in st.session_state:
    st.session_state.test_complete = False

if "latest_image" not in st.session_state:
    st.session_state.latset_image = None

if(
    st.session_state.resume_token
    and st.session_state.user_id is None
    and not st.session_state.viewing_information
):

    try:
        participant = resume_participant_record(
            st.session_state.resume_token
        )

        if participant is not None:
            st.session_state.user_id = participant ["user_id"]
            st.session_state.completed_tests = int(
                participant["test_count"]
            )
            st.session_state.survey_submitted = bool(
                participant["survey_submitted"]
            )

            if st.session_state.survey_submitted:
                st.session_state.participant_stage = "complete"
            else:
                st.session_state.participant_stage = "testing"

        else:
            st.session_state.resume_token = None
            st.query_params.clear()

    except Exception as error:
        st.error(
            "The participant session could not be restored."
        )
        logging.exception("Failed to save test result")
        st.stop()

if st.session_state.participant_stage == "information":

    st.title("Participant Information")

    st.subheader("About this project")

    st.write(
        "This user testing forms part of a TM470 Computing and IT project "
        "investigating and comparing three Convolutional Neural Network (CNN) "
        "models for mushroom image recognition."
    )

    st.subheader("What will I be asked to do?")

    st.write(
        "You will be asked to upload different images to the application. "
        "The images may contain mushrooms or non-mushroom subjects. "
        "The application will process each image using three AI models and "
        "display their predictions, confidence scores and processing times."
    )

    st.write(
        "Please aim to test at least 5 different images if possible. "
        "You may continue testing up to a maximum of 10 images."
    )

    st.subheader("What information will be collected?")

    st.write(
        "The application will record a randomly generated Participant ID, "
        "a unique Test ID for each test, the type of image selected, model "
        "predictions, confidence scores, processing times, timestamps and "
        "your responses to the usability survey."
    )

    st.write(
        "Your uploaded images are processed temporarily by the application "
        "and are not stored as part of the research data."
    )

    st.subheader("Voluntary participation")

    st.write(
        "Taking part is voluntary. You may stop participating at any time. "
        "You do not have to provide a reason."
    )

    st.subheader("Withdrawal of your data")

    st.write(
        "You will be provided with a randomly generated Participant ID. "
        "Please keep this ID if you may wish to request withdrawal of your "
        "data. Withdrawal requests can be made within 7 days of completing "
        "the testing."
    )

    st.info(
        "Instructions explaining how to request withdrawal will be provided "
        "when you complete the testing."
    )

    st.subheader("Important safety information")

    st.warning(
        "This application is an academic prototype only. It always selects "
        "the closest match from the mushroom classes it has been trained to "
        "recognise and cannot reliably determine whether an image contains "
        "a mushroom. Do not use the application to determine whether a "
        "mushroom is safe to touch or consume."
    )

    st.subheader("Who can participate?")

    st.write(
        "Participants must be aged 18 or over."
    )

    st.subheader("Contact")

    st.write(
        "If you have any questions about the project or wish to request "
        "withdrawal of your data, please contact:"
    )

    st.write(
        "**Researcher:** Brendan Fitzpatrick  \n"
        "**Email:** zy923716@ou.ac.uk"
    )

    continue_column, withdrawal_column = st.columns(2)

    with continue_column:
        if st.session_state.user_id is None:
            if st.button(
                "Continue to Consent",
                type="primary",
        ):
                st.session_state.participant_stage = "consent"
                st.rerun()

    with withdrawal_column:
        if st.button(
            "Request Data Withdrawal",
        ):
            st.session_state.participant_stage = "withdrawal"
            st.rerun()

    st.stop()

if st.session_state.participant_stage == "consent":

    st.title("Consent Form")

    st.write(
        "Please confirm each of the statements below before beginning "
        "the application testing."
    )

    read_information = st.checkbox(
        "I confirm that I have read and understood the Participant "
        "Information provided for this study."
    )

    age_confirmation = st.checkbox(
        "I confirm that I am aged 18 or over."
    )

    voluntary_confirmation = st.checkbox(
        "I understand that my participation is voluntary and that I may "
        "stop participating at any time."
    )

    data_confirmation = st.checkbox(
        "I understand what testing data will be collected and that it "
        "will be used as part of this academic project."
    )

    image_confirmation = st.checkbox(
        "I understand that images I upload will be processed temporarily "
        "by the application and will not be stored as part of the research data."
    )

    withdrawal_confirmation = st.checkbox(
        "I understand that I will receive a Participant ID which can be "
        "used to request withdrawal of my data within 7 days of completing "
        "the testing."
    )

    consent_given = all([
        read_information,
        age_confirmation,
        voluntary_confirmation,
        data_confirmation,
        image_confirmation,
        withdrawal_confirmation,
    ])

    if st.button(
        "Consent and Begin Testing",
        disabled=not consent_given,
        type="primary",
    ):
        try:
            participant = create_participant_record()

            st.session_state.user_id = participant["user_id"]

            st.session_state.resume_token = str(
                participant["resume_token"]
            )

            st.session_state.completed_tests = 0
            st.session_state.survey_submitted = False

            st.query_params["resume"] = (
                st.session_state.resume_token
            )

            st.session_state.participant_stage = "testing"

            st.rerun()

        except Exception as error:
            st.error(
                "Your participant session could not be created. "
                "Please try again."
            )
            st.stop()

    st.stop()


def refresh_participant_progress():
    if not st.session_state.resume_token:
        return

    participant = resume_participant_record(
        st.session_state.resume_token
    )

    if participant is None:
        raise RuntimeError(
            "Participant record could not be found."
        )

    st.session_state.user_id = participant["user_id"]

    st.session_state.completed_tests = int(
        participant["test_count"]
    )

    st.session_state.survey_submitted = bool(
        participant["survey_submitted"]
    )


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

                    logging.exception("Failed to save test result")
                    st.stop()

                st.session_state.participant_stage = "complete"

                st.rerun()

        st.stop()

if st.session_state.participant_stage == "complete":

    st.title("Testing Complete")

    st.success(
        "Thank you for taking part in the application testing."
    )

    st.write(
        "Your image-testing results and usability survey "
        "have been successfully submitted."
    )

    st.write(
        f"Your Participant ID is: **{st.session_state.user_id}**"
    )

    st.info(
        "Please keep your Participant ID if you wish to "
        "request withdrawal of your data within the stated "
        "withdrawal period."
    )

    withdrawal, information = st.columns(2)

    with withdrawal:

        if st.button("Request Data Withdrawal"):
            st.session_state.viewing_information = False
            st.session_state.participant_stage = "withdrawal"
            st.rerun()

    with information:
        if st.button("Back to Participant Information"):
            st.session_state.viewing_information = True
            st.session_state.participant_stage = "information"
            st.rerun()     

    st.stop()


if st.session_state.participant_stage =="withdrawal":

    st.title("Request Data Withdrawal")

    st.write(
        "If you previously took part in the application testing and would "
        "like your research data to be removed, enter the Participant ID "
        "you were given when you completed the testing"
    )

    st.info(
        "Your Participant ID will look similar to: User-12AB34CD"
    )

    with st.form("withdrawal_request_form"):
        withdrawal_user_id = st.text_input(
            "Participant ID"
        )

        confirm_withdrawal = st.checkbox(
            "I confirm that I am requesting withdrawal of the research "
            "data associated with this Participant ID."
        )

        submit_withdrawal = st.form_submit_button(
            "Submit Withdrawal Request",
            type="primary",
        )

    if submit_withdrawal:

        cleaned_user_id = withdrawal_user_id.strip()

        if not cleaned_user_id:
            st.warning(
                "Please enter your Participant ID."
            )

        elif not confirm_withdrawal:
            st.warning(
                "Please confirm that you wish to request withdrawal."
            )

        else:
            try:
                save_withdrawal_request(
                    cleaned_user_id
                )

                st.success(
                    "Your withdrawal request has been submitted. "
                    "The research data associated with the supplied "
                    "Participant ID will be reviewed for removal."
                )

            except Exception as error:
                st.error(
                    "The withdrawal request could not be submitted. "
                    "Please check your Participant ID and try again."
                )

    if st.button("Back to Participant Information"):
        st.session_state.participant_stage = "information"
        st.rerun()

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


if st.session_state.participant_stage == "testing":

    try:
        refresh_participant_progress()

    except Exception:
        st.error(
            "Your testing progress could not be loaded. "
            "Please refresh the page and try again."
        )
        st.stop()

if not st.session_state.test_complete:

    uploaded_file = st.file_uploader(
        "Upload an image for model comparison",
        type=["jpg", "jpeg", "png"],
        key=f"uploaded_file_{st.session_state.completed_tests}",
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

                logging.exception("Failed to save test result")

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

                logging.exception("Failed to save test result")

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

                logging.exception("Failed to save test result")
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

                logging.exception("Failed to save test result")
                st.stop()

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

                if st.session_state.completed_tests < 10:
                    if st.button(
                        "Test Another Image",
                        type="primary"
                ):
                        st.session_state.test_complete = False
                        st.session_state.latest_result = None
                        st.session_state.latest_image = None
                        st.rerun()
                else:
                    st.button(
                        "Test Another Image",
                        type="primary",
                        disabled= True,
                    )

            with finish_column:
                    
                    if st.button(
                        "Finish Testing",
                        disabled=st.session_state.completed_tests < 5,
                ):
                        st.session_state.test_complete = False
                        st.session_state.latest_result = None
                        st.session_state.latest_image = None
                        st.session_state.participant_stage = "survey"
                        st.rerun()


                    if st.session_state.completed_tests < 5:
                        st.info(
                            f"Please complete at least 5 image tests before finishing. "
                            f"You have currently completed {st.session_state.completed_tests}."
                        )

                    elif st.session_state.completed_tests < 10:
                        st.success(
                            f"You have completed {st.session_state.completed_tests} out of 10 image tests. "
                            "Thank you for reaching the recommended minimum of 5. "
                            "You may continue testing more images, or finish testing and complete the usability survey."
                        )

                    else:
                        st.success(
                            "Thankyou for completing the 10 image tests. "
                            "You have reached the maximum recommended number of test images.  "
                            "Please finish testing and complete the usability survey."
                        )