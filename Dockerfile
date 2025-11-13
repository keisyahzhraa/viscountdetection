FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch wheels (CEPAT)
RUN pip install --upgrade pip && \
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install --no-cache-dir -r yolov5/requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app"]
