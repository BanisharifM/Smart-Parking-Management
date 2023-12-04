import json
from io import BytesIO
from PIL import Image
import os

import boto3
from botocore import UNSIGNED  # contact public s3 buckets anonymously
from botocore.client import Config  # contact public s3 buckets anonymously

import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import ResnetModel

import cv2
import numpy as np
from ultralytics import YOLO


@st.cache_data()
def load_model(path: str = "utils/runs/detect/train2/weights/best.pt") -> ResnetModel:
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


# @st.cache_data()
# def load_list_of_images_available(
#     all_image_files: dict, image_files_dtype: str, bird_species: str
# ) -> list:
#     """Retrieves list of available images given the current selections"""
#     species_dict = all_image_files.get(image_files_dtype)
#     list_of_files = species_dict.get(bird_species)
#     return list_of_files


@st.cache_data()
def predict(img: Image.Image) -> list:
    #     formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
    formatted_predictions = model.predict(source=[img], conf=0.45, save=False)
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

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )
    return frame


# def cal_classes_counts(total_detections, detect_params, class_list):
#     class_counts = {class_name: 0 for class_name in class_list}
#     print("class_counts", class_counts)
#     if total_detections != 0:
#         for i in range(len(detect_params[0])):
#             boxes = detect_params[0].boxes
#             box = boxes[i]
#             clsID = box.cls.numpy()[0]
#             class_name = class_list[int(clsID)]
#             if class_name in class_counts:
#                 class_counts[class_name] += 1
#             else:
#                 # Optionally handle the case where class_name is not found
#                 # You can choose to ignore, log, or handle this situation accordingly
#                 pass
#         #     class_counts[class_name] += 1

#     return class_counts


def cal_classes_percentage(total_detections, class_counts):
    class_percentages = {
        class_name: count / total_detections * 100
        for class_name, count in class_counts.items()
    }
    for class_name in class_counts.items():
        print(f"Percentage of {class_name}: {class_percentages[class_name]:.2f}%")


if __name__ == "__main__":
    model = load_model()
    class_list = load_index_to_label_dict()
    all_image_files = load_s3_file_structure()
    #     types_of_birds = sorted(list(all_image_files["test"].keys()))
    #     types_of_birds = [bird.title() for bird in types_of_birds]

    detection_colors = [(252, 191, 73), (207, 233, 8)]

    st.title("Welcome To Project Eagle Vision!")
    instructions = """
        SPM(Smart Parking Management) aims to transform this using machine learning, specifically the YOLO v8 algorithm, to analyze parking images, count cars, and spot available spaces.
        """
    st.write(instructions)

    file = st.file_uploader("Upload An Image")
    dtype_file_structure_mapping = {
        "All Images": "consolidated",
        "Images Used To Train The Model": "train",
        "Images Used To Tune The Model": "valid",
        "Images The Model Has Never Seen": "test",
    }
    data_split_names = list(dtype_file_structure_mapping.keys())

    print(file)
    if file:  # if user uploaded file
        temp_directory = "./uploaded_images"
        os.makedirs(temp_directory, exist_ok=True)
        file_path = os.path.join(temp_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        st.write("File saved to:", file_path)

        img = Image.open(file)

        image_path = "uploaded_images/" + file.name  # Change this to your image path
        frame = cv2.imread(image_path)
        detect_params = predict(frame)

        print(detect_params)

        # total_detections = len(prediction[0]) if len(prediction[0]) != 0 else 1

        # class_counts = cal_classes_counts(total_detections, prediction, class_names)
        # print(class_counts)

        # class_sums = sum(class_counts.values())
        # print("Class Counts:", class_sums)

        # classes_percentage = cal_classes_percentage(total_detections, class_counts)

        DP = detect_params[0].numpy()
        print(DP)

        class_counts = {value: 0 for value in class_list.values()}

        total_detections = len(detect_params[0]) if len(detect_params[0]) != 0 else 1

        if total_detections != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                class_name = class_list[int(clsID)]
                print("class_counts", class_counts)
                print("class_list", class_list)
                print("clsID", clsID)
                print("class_name", class_name)
                class_counts[class_name] += 1

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

        # class_sums = [class_counts[class_name] for class_name in class_list]
        # print("Class Counts:", class_sums)

        class_sums = sum(class_counts.values())
        class_percentages = {
            class_name: count / total_detections * 100
            for class_name, count in class_counts.items()
        }

        for class_name, count in class_counts.items():
            print(f"Percentage of {class_name}: {class_percentages[class_name]:.2f}%")

    # top_prediction = prediction[0][0]
    # available_images = all_image_files.get("train").get(top_prediction.upper())
    # examples_of_species = np.random.choice(available_images, size=3)
    # files_to_get_from_s3 = []

    # for im_name in examples_of_species:
    #     path = os.path.join("train", top_prediction.upper(), im_name)
    #     files_to_get_from_s3.append(path)
    # images_from_s3 = load_files_from_s3(keys=files_to_get_from_s3)

#     else:
#         dataset_type = st.sidebar.selectbox("Data Portion Type", data_split_names)
#         image_files_subset = dtype_file_structure_mapping[dataset_type]

#         selected_species = st.sidebar.selectbox("Bird Type", types_of_birds)
#         available_images = load_list_of_images_available(
#             all_image_files, image_files_subset, selected_species.upper()
#         )
#         image_name = st.sidebar.selectbox("Image Name", available_images)
#         if image_files_subset == "consolidated":
#             s3_key_prefix = "consolidated/consolidated"
#         else:
#             s3_key_prefix = image_files_subset
#         key_path = os.path.join(s3_key_prefix, selected_species.upper(), image_name)
#         files_to_get_from_s3 = [key_path]
#         examples_of_species = np.random.choice(available_images, size=3)

#         for im in examples_of_species:
#             path = os.path.join(s3_key_prefix, selected_species.upper(), im)
#             files_to_get_from_s3.append(path)
#         images_from_s3 = load_files_from_s3(keys=files_to_get_from_s3)
#         img = images_from_s3.pop(0)
#         prediction = predict(img, index_to_class_label_dict, model, 5)

#     st.title("Here is the image you've selected")
#     resized_image = img.resize((336, 336))
#     st.image(resized_image)
#     st.title("Here are the five most likely bird species")
#     df = pd.DataFrame(
#         data=np.zeros((5, 2)),
#         columns=["Species", "Confidence Level"],
#         index=np.linspace(1, 5, 5, dtype=int),
#     )

#     for idx, p in enumerate(prediction):
#         link = "https://en.wikipedia.org/wiki/" + p[0].lower().replace(" ", "_")
#         df.iloc[idx, 0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
#         df.iloc[idx, 1] = p[1]
#     st.write(df.to_html(escape=False), unsafe_allow_html=True)
#     st.title(f"Here are three other images of the {prediction[0][0]}")

#     st.image(images_from_s3)
# st.title('How it works:')
