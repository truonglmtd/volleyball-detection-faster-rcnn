FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip install pandas scikit-learn matplotlib opencv-python torchmetrics tensorboard
RUN apt update
RUN apt install vim -y
RUN apt install libgl1-mesa-glx -y
RUN apt install libgtk2.0-dev -y

COPY train.py /workspace/train.py
COPY dataset.py /workspace/dataset.py

CMD ["python", "train.py"]





