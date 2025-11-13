FROM pytorch/pytorch:2.8.0-cpu

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Clone YOLOv5 (menghindari torch.hub) ---
RUN git clone https://github.com/ultralytics/yolov5
RUN pip install --no-cache-dir -r yolov5/requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app"]
