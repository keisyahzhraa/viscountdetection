import os
import cv2
import torch
from yolov5 import YOLO
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from tqdm import tqdm

# ==== Konfigurasi dasar ====
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Batasi 500MB

# ==== Load YOLOv5 Model ====
try:
    model = YOLO('yolov5s.pt')
    model.conf = 0.4  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_yolo(input_path, output_path):
    if model is None:
        raise Exception("Model not loaded")
        
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Gunakan codec yang lebih kompatibel
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        'avg': round(np.mean(counts), 2) if counts else 0,
        'max': int(np.max(counts)) if counts else 0,
        'max_frame': int(np.argmax(counts)) if counts else 0,
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

            # Perbaikan nama output
            base_name = os.path.splitext(input_name)[0]  # Perbaikan di sini
            output_name = f"processed_{base_name}.mp4"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)

            # Proses video
            try:
                stats = process_video_yolo(input_path, output_path)
                return render_template('result.html',
                                       input_video=url_for('static', filename=f'uploads/{input_name}'),
                                       output_video=url_for('static', filename=f'processed/{output_name}'),
                                       stats=stats)
            except Exception as e:
                return f"Error processing video: {str(e)}", 500
                
    return render_template('upload.html')

@app.errorhandler(413)
def too_large(e):
    return "File terlalu besar. Maksimal 500MB.", 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)