import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class VolleyballDataset(Dataset):
    def __init__(self, root, year=None, image_set="train", download=False, transform=None):
        """
        root: đường dẫn gốc dataset, ví dụ: data/volleyball
        image_set: train hoặc val/test
        transform: transform ảnh từ train.py
        """

        self.root = root
        self.image_set = image_set
        self.transform = transform

        # Dataset của bạn đang nằm trong: data/volleyball/volleyball
        if os.path.isdir(os.path.join(root, "volleyball")):
            self.data_dir = os.path.join(root, "volleyball")
        else:
            self.data_dir = root

        # Đọc classes.txt
        classes_path = os.path.join(self.data_dir, "classes.txt")

        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip()
                ]
        else:
            # Nếu không có classes.txt thì mặc định 1 class
            class_names = ["volleyball"]

        # Faster R-CNN dùng label 0 làm background
        # Vì vậy class thật bắt đầu từ 1
        self.categories = ["background"] + class_names

        # Lấy toàn bộ file ảnh trong thư mục
        all_images = [
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        all_images = sorted(all_images)

        # Tự chia train/val theo tỷ lệ 80/20
        split_index = int(len(all_images) * 0.8)

        if image_set == "train":
            self.image_files = all_images[:split_index]
        else:
            self.image_files = all_images[split_index:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        img_file = self.image_files[item]
        img_path = os.path.join(self.data_dir, img_file)

        # Đọc ảnh
        image = Image.open(img_path).convert("RGB")

        # Transform ảnh: Resize, ToTensor, Normalize
        if self.transform:
            image = self.transform(image)

        # Sau transform, image có dạng tensor [C, H, W]
        _, new_h, new_w = image.shape

        # File nhãn YOLO có cùng tên với ảnh, đuôi .txt
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.data_dir, label_file)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    # YOLO format cần ít nhất 5 giá trị:
                    # class_id x_center y_center width height
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(float(parts[0]))

                        x_center = float(parts[1]) * new_w
                        y_center = float(parts[2]) * new_h
                        width = float(parts[3]) * new_w
                        height = float(parts[4]) * new_h

                        xmin = x_center - width / 2
                        ymin = y_center - height / 2
                        xmax = x_center + width / 2
                        ymax = y_center + height / 2

                        # Giới hạn box nằm trong ảnh
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(new_w, xmax)
                        ymax = min(new_h, ymax)

                        # Chỉ lấy box hợp lệ
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])

                            # YOLO class_id bắt đầu từ 0
                            # Faster R-CNN dùng 0 cho background
                            # nên phải +1
                            labels.append(class_id + 1)

                    except ValueError:
                        continue

        # Nếu ảnh không có box hợp lệ,
        # vẫn phải trả về tensor đúng shape [0, 4]
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target