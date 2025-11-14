FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# TORCH PALING RINGAN YANG SUPPORT PYTHON 3.8
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLOv5
RUN git clone https://github.com/ultralytics/yolov5
RUN pip install -r yolov5/requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-t", "300", "app:app"]
