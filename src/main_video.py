import json
from io import BytesIO
from PIL import Image
import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config

import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import ResnetModel

import cv2
import numpy as np
from ultralytics import YOLO

import matplotlib.pyplot as plt
import tempfile
from datetime import datetime


@st.cache_data()
def load_model(path: str = "utils/runs/detect/train2/weights/best4.pt") -> ResnetModel:
    model = YOLO(path, "v8")
    return model


@st.cache_data()
def load_index_to_label_dict(path: str = "utils/class_label.json") -> dict:
    with open(path, "r") as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()
    }
    return index_to_class_label_dict


def load_files_from_s3(
    keys: list, bucket_name: str = "bird-classification-bucket"
) -> list:
    """Retrieves files from S3 bucket"""
    s3 = boto3.client("s3")
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw["Body"].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files


@st.cache_data()
def load_s3_file_structure(path: str = "src/all_image_files.json") -> dict:
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data()
def load_list_of_images_available(
    all_image_files: dict, image_files_dtype: str, bird_species: str
) -> list:
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files


@st.cache_data()
def predict(img, conf_rate) -> list:
    #     formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
    formatted_predictions = model.predict(source=[img], conf=conf_rate, save=False)
    return formatted_predictions


def image_annotation(detect_params, frame, class_list, detection_colors):
    total_detections = len(detect_params[0]) if len(detect_params[0]) != 0 else 1
    if total_detections != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            class_name = class_list[int(clsID)]
            # class_counts[class_name] += 1

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                font_scale,
                (255, 255, 255),
                2,
            )
    return frame


def cal_classes_counts(total_detections, detect_params, class_list):
    class_counts = {value: 0 for value in class_list.values()}
    if total_detections != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            class_name = class_list[int(clsID)]
            class_counts[class_name] += 1

    return class_counts


# def cal_classes_percentage(total_detections, class_counts):
#     class_percentages = {
#         class_name: count / total_detections * 100
#         for class_name, count in class_counts.items()
#     }
#     for class_name in class_counts.items():
#         print(f"Percentage of {class_name}: {class_percentages[class_name]:.2f}%")


def cal_classes_percentage(total_detections, class_counts):
    class_percentages = {
        class_name: count / total_detections * 100
        for class_name, count in class_counts.items()
    }

    return class_percentages


def save_uploaded_image(file, uploaded_path):
    os.makedirs(uploaded_path, exist_ok=True)
    file_path = os.path.join(uploaded_path, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())


def save_image(image, image_name, output_path):
    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)

    valid_extensions = [".jpg", ".jpeg", ".png"]  # Add more extensions if needed
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in valid_extensions:
        output_path = os.path.splitext(output_path)[0] + image_name

    cv2.imwrite(output_path, image)

    if os.path.exists(output_path):
        print(f"Image saved successfully to: {output_path}")
    else:
        print("Failed to save the image.")


def generate_pie_chart(class_counts):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)

    ax.patch.set_alpha(0.0)

    st.pyplot(fig)


def generate_enhanced_bar_chart(class_counts):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))  # Set width=10 inches, height=6 inches
    bars = ax.bar(labels, sizes, color="skyblue")

    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{size}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=10,
        )

    ax.set_xlabel("Categories")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)


def generate_enhanced_pie_chart(class_counts):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    explode = [0.1] * len(labels)
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        shadow=True,
    )

    ax.axis("equal")

    st.pyplot(fig)


def detection_image(file):
    img = Image.open(file)

    save_uploaded_image(file, uploaded_path)

    image_path = uploaded_path + file.name
    frame = cv2.imread(image_path)
    print("image_path: ", image_path)
    prediction = predict(frame, confidence_rate)
    print("confidence_rate", confidence_rate)

    predicted_image = image_annotation(prediction, frame, class_list, detection_colors)
    save_image(predicted_image, file.name, predicted_path)

    total_detections = len(prediction[0]) if len(prediction[0]) != 0 else 1

    class_counts = cal_classes_counts(total_detections, prediction, class_list)

    file_path = os.path.join(predicted_path, file.name)
    img = Image.open(file_path)

    new_height = 550

    width, height = img.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    resized_image = img.resize((new_width, new_height))

    # resized_image = img.resize((336, 336))
    st.title("Detected Output ")
    st.image(resized_image)
    df = pd.DataFrame(
        data=np.zeros((5, 2)),
        columns=["Species", "Confidence Level"],
        index=np.linspace(1, 5, 5, dtype=int),
    )

    st.title("Empety VS Parked Bar Chart")
    generate_enhanced_bar_chart(class_counts)
    st.title("Empety VS Parked Pie Chart")
    generate_enhanced_pie_chart(class_counts)


def detection_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    frame_step = 10
    frame_position = 0
    output_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_position < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_position += frame_step

        prediction = predict(frame, confidence_rate)

        predicted_image = image_annotation(
            prediction, frame, class_list, detection_colors
        )

        output_frames.append(predicted_image)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    output_directory = "predicted_video/"
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_name = f"processed_video_{current_datetime}.mp4"
    output_video_path = os.path.join(output_directory, output_video_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        30.0,
        (output_frames[0].shape[1], output_frames[0].shape[0]),
    )

    for frame in output_frames:
        out.write(frame)

    out.release()

    return output_video_path


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


if __name__ == "__main__":
    uploaded_path = "uploaded_images/"
    predicted_path = "predicted_images/"

    model = load_model()
    class_list = load_index_to_label_dict()
    all_image_files = load_s3_file_structure()
    types_of_birds = sorted(list(all_image_files["test"].keys()))
    types_of_birds = [bird.title() for bird in types_of_birds]

    detection_colors = [(10, 239, 8), (252, 10, 73)]

    confidence_rate = 0.45

    st.title("Smart Parking Management!")
    instructions = """
        SPM (Smart Parking Management) aims to use machine learning, specifically the YOLO v8 algorithm, to analyze parking images and count cars and available spaces.
        """
    st.write(instructions)

    #     file = st.file_uploader("Upload An Image")
    file = st.file_uploader("Upload A Video", type=["mp4", "avi"])
    dtype_file_structure_mapping = {"Video": "video", "Image": "image"}
    data_split_names = list(dtype_file_structure_mapping.keys())

    global data_type
    data_type = "video"

    if file:
        if data_type == "image":
            detection_image(file)
        elif data_type == "video":
            file_path = save_uploaded_file(file)
            processed_video_path = detection_video(file_path)
            print(processed_video_path)
            st.video(processed_video_path)
    else:
        data_type = st.sidebar.selectbox("Input Type", data_split_names)
        confidence_rate = (
            st.sidebar.slider("Confidence Rate:", min_value=0, max_value=100, value=50)
            / 100
        )
