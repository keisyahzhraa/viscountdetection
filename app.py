import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from tqdm import tqdm
from yolov5 import detect 

# ==== Konfigurasi dasar ====
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# ==== Load YOLOv5 Model ====
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_yolo(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    counts = []

    for _ in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]
        people = df[df['name'] == 'person']

        # Draw bounding boxes
        for _, row in people.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        count = len(people)
        counts.append(count)
        cv2.putText(frame, f'People: {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    stats = {
        'frames': len(counts),
        'avg': round(np.mean(counts), 2),
        'max': int(np.max(counts)),
        'max_frame': int(np.argmax(counts)),
        'data': counts
    }
    return stats


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "Tidak ada file video", 400
        file = request.files['video']
        if file.filename == '':
            return "File belum dipilih", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = f"{timestamp}_{filename}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_name)
            file.save(input_path)

# Pastikan nama output valid dan .mp4
            base_name = os.path.splitext(input_name)
            output_name = f"processed_{base_name}.mp4"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)

            # Proses video
            stats = process_video_yolo(input_path, output_path)

            return render_template('result.html',
                                   input_video=url_for('static', filename=f'uploads/{input_name}'),
                                   output_video=url_for('static', filename=f'processed/{output_name}'),
                                   stats=stats)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
