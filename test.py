from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import cv2
import argparse
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser(description="Test Faster R-CNN Volleyball Detection")

    parser.add_argument(
        "-i", "--image",
        type=str,
        default="test_images/1.jpg",
        help="path to test image"
    )

    parser.add_argument(
        "-s", "--size",
        default=416,
        type=int,
        help="image size"
    )

    parser.add_argument(
        "-t", "--conf_threshold",
        default=0.5,
        type=float,
        help="confidence threshold"
    )

    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="trained_models/best.pt",
        help="path to model checkpoint file"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/volleyball",
        help="path to volleyball dataset"
    )

    args = parser.parse_args()
    return args


def load_categories(data_path):
    """
    Đọc danh sách class từ classes.txt.
    Dataset của bạn đang nằm dạng:
    data/volleyball/volleyball/classes.txt
    """

    if os.path.isdir(os.path.join(data_path, "volleyball")):
        data_dir = os.path.join(data_path, "volleyball")
    else:
        data_dir = data_path

    classes_path = os.path.join(data_dir, "classes.txt")

    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]
    else:
        class_names = ["volleyball"]

    # Faster R-CNN dùng 0 làm background
    categories = ["background"] + class_names

    return categories


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories = load_categories(args.data_path)
    print("Categories:", categories)
    print("Number of classes:", len(categories))

    # Tạo model đúng số class lúc train
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=len(categories)
    )

    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Đọc ảnh test
    ori_image = cv2.imread(args.image)

    if ori_image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh test: {args.image}")

    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Resize giống lúc train
    image = cv2.resize(image, (args.size, args.size))

    # Normalize giống train.py
    image = image / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float().to(device)
    image = [image]

    # Dự đoán
    with torch.no_grad():
        predictions = model(image)

    # Vẽ bounding box
    for box, score, label in zip(
        predictions[0]["boxes"],
        predictions[0]["scores"],
        predictions[0]["labels"]
    ):
        if score > args.conf_threshold:
            xmin, ymin, xmax, ymax = box

            xmin = int(xmin / args.size * width)
            ymin = int(ymin / args.size * height)
            xmax = int(xmax / args.size * width)
            ymax = int(ymax / args.size * height)

            label_id = int(label.item())

            if label_id < len(categories):
                class_name = categories[label_id]
            else:
                class_name = f"class_{label_id}"

            text = f"{class_name} {score:.2f}"

            cv2.rectangle(
                ori_image,
                (xmin, ymin),
                (xmax, ymax),
                (128, 0, 128),
                2
            )

            cv2.putText(
                ori_image,
                text,
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (255, 255, 0),
                2
            )

    # Lưu ảnh kết quả
    output_path = "prediction.jpg"
    cv2.imwrite(output_path, ori_image)

    print(f"Đã lưu ảnh kết quả tại: {output_path}")


if __name__ == "__main__":
    args = get_args()
    test(args)