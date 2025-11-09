import os
import cv2
import torch
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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ==== Load YOLOv5 Model ====
def load_model():
    try:
        # Cara 1: Gunakan ultralytics package langsung
        from ultralytics import YOLO
        model = YOLO('yolov5s.pt')
        print("Model loaded successfully via ultralytics")
        return model
    except Exception as e:
        print(f"Error with ultralytics: {e}")
        try:
            # Cara 2: Gunakan torch hub dengan source alternatif
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
            print("Model loaded successfully via torch hub")
            return model
        except Exception as e2:
            print(f"Error with torch hub: {e2}")
            return None

model = load_model()

if model is None:
    print("Warning: Model could not be loaded. The app will not function properly.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_yolo(input_path, output_path):
    if model is None:
        raise Exception("Model not loaded. Please check the logs for loading errors.")
        
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise Exception("Tidak dapat membaca frame dari video")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    counts = []

    for _ in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek
        results = model(frame)
        
        # Handle different result formats based on how model was loaded
        if hasattr(results, 'pandas'):
            # Untuk model torch hub
            df = results.pandas().xyxy[0]
        else:
            # Untuk model ultralytics YOLO
            df = results[0].pandas().xyxy[0]
            
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

            base_name = os.path.splitext(input_name)[0]
            output_name = f"processed_{base_name}.mp4"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)

            try:
                stats = process_video_yolo(input_path, output_path)
                return render_template('result.html',
                                       input_video=url_for('static', filename=f'uploads/{input_name}'),
                                       output_video=url_for('static', filename=f'processed/{output_name}'),
                                       stats=stats)
            except Exception as e:
                # Hapus file input jika error
                if os.path.exists(input_path):
                    os.remove(input_path)
                return f"Error processing video: {str(e)}", 500
                
    return render_template('upload.html')

@app.errorhandler(413)
def too_large(e):
    return "File terlalu besar. Maksimal 500MB.", 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)