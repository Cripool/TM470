"""
Streamlit mushroom image-recognition demonstator for the TM470 EMA.
"""

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="CNN Mushroom Recognition Comparison",
    page_icon="🍄",
    layout="wide",
)

st.title("CNN Mushroom Recognition Comparison")

st.write(
    "Upload a mushroom image to compare predictions from the "
    "Baseline CNN, corrected ResNet18 and MobileNetV2 models."
)

st.warning(
    "Academic demonstrator only. Do not use these predictions "
    "to determine whether the mushroom is safe to touch or consume."
)

upload_file = st.file_uploader(
    "Upload a mushroom image",
    type=["jpeg", "jpg", "png"],
)

if upload_file is not None:
    image= Image.open(upload_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded mushroom image",
        width= 400,
    )

    st.subheader("Model Predictions")

    baseline_column, resnet_column, mobilenet_column = st.columns(3)

    with baseline_column:
        st.markdown("### Baseline CNN")
        st.info("Model connection in progress")

    with resnet_column:
        st.markdown("### ResNet18")
        st.info("Model connection in progress")


    with mobilenet_column:
        st.markdown("### MobileNetV2")
        st.info("Model connection in progress")

