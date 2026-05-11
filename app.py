import os
import cv2
import torch
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_categories(data_path="data/volleyball"):
    if os.path.isdir(os.path.join(data_path, "volleyball")):
        data_dir = os.path.join(data_path, "volleyball")
    else:
        data_dir = data_path

    classes_path = os.path.join(data_dir, "classes.txt")

    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        class_names = ["Player", "Referee", "Ball"]

    return ["background"] + class_names


@st.cache_resource
def load_model(checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=num_classes
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict_image(model, device, image_bgr, categories, image_size=416, conf_threshold=0.1):
    ori_image = image_bgr.copy()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    image = cv2.resize(image_rgb, (image_size, image_size))
    image = image / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float().to(device)
    image = [image]

    with torch.no_grad():
        predictions = model(image)

    for box, score, label in zip(
        predictions[0]["boxes"],
        predictions[0]["scores"],
        predictions[0]["labels"]
    ):
        if score > conf_threshold:
            xmin, ymin, xmax, ymax = box

            xmin = int(xmin / image_size * width)
            ymin = int(ymin / image_size * height)
            xmax = int(xmax / image_size * width)
            ymax = int(ymax / image_size * height)

            label_id = int(label.item())
            class_name = categories[label_id] if label_id < len(categories) else f"class_{label_id}"
            text = f"{class_name} {score:.2f}"

            cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), (128, 0, 128), 2)
            cv2.putText(
                ori_image,
                text,
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    return ori_image


st.set_page_config(page_title="Volleyball Detection", layout="wide")

st.title("Phát hiện người chơi, trọng tài và bóng trong ảnh bóng chuyền")
st.write("Mô hình sử dụng Faster R-CNN MobileNet để phát hiện các đối tượng trong ảnh.")

checkpoint_path = "trained_models/best.pt"
data_path = "data/volleyball"

categories = load_categories(data_path)

st.sidebar.header("Cài đặt")
conf_threshold = st.sidebar.slider("Ngưỡng confidence", 0.05, 0.9, 0.1, 0.05)
image_size = st.sidebar.selectbox("Kích thước ảnh", [320, 416, 512], index=1)

st.sidebar.write("Classes:", categories)

model, device = load_model(checkpoint_path, len(categories))

uploaded_file = st.file_uploader("Tải ảnh bóng chuyền lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    result_bgr = predict_image(
        model=model,
        device=device,
        image_bgr=image_bgr,
        categories=categories,
        image_size=image_size,
        conf_threshold=conf_threshold
    )

    with col2:
        st.subheader("Kết quả dự đoán")
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Hãy tải một ảnh bóng chuyền lên để xem kết quả dự đoán.")