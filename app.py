import os
import cv2
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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB saja

# ==== Load Model ====
def load_model():
    try:
        # Gunakan ultralytics YOLO langsung
        from ultralytics import YOLO
        
        # Coba load model YOLOv8n yang lebih kecil
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying to install missing dependencies...")
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_yolo(input_path, output_path):
    if model is None:
        raise Exception("Model tidak berhasil di-load. Silakan cek log untuk detailnya.")
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Tidak dapat membuka file video")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        cap.release()
        raise Exception("Tidak dapat membaca frame dari video")
    
    # Buat video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise Exception("Tidak dapat membuat output video")
    
    counts = []
    processed_frames = 0

    print(f"Memproses video: {frame_count} frames")
    
    for i in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Deteksi objek dengan YOLO
            results = model(frame, verbose=False)
            
            # Process results
            count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        if model.names[cls] == 'person':  # Deteksi orang saja
                            count += 1
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, 'Person', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            counts.append(count)
            
            # Tambahkan counter di frame
            cv2.putText(frame, f'People: {count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Frame: {i}/{frame_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            processed_frames += 1
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    cap.release()
    out.release()
    
    if processed_frames == 0:
        raise Exception("Tidak ada frame yang berhasil diproses")

    # Hitung statistik
    stats = {
        'frames': len(counts),
        'avg': round(np.mean(counts), 2) if counts else 0,
        'max': int(np.max(counts)) if counts else 0,
        'min': int(np.min(counts)) if counts else 0,
        'max_frame': int(np.argmax(counts)) if counts else 0,
        'data': counts
    }
    
    print(f"✅ Video processing completed: {processed_frames} frames processed")
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
            
            try:
                file.save(input_path)
                print(f"✅ File saved: {input_path}")

                base_name = os.path.splitext(input_name)[0]
                output_name = f"processed_{base_name}.mp4"
                output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)

                stats = process_video_yolo(input_path, output_path)
                
                return render_template('result.html',
                                       input_video=url_for('static', filename=f'uploads/{input_name}'),
                                       output_video=url_for('static', filename=f'processed/{output_name}'),
                                       stats=stats)
                                       
            except Exception as e:
                # Cleanup jika error
                if os.path.exists(input_path):
                    os.remove(input_path)
                if 'output_path' in locals() and os.path.exists(output_path):
                    os.remove(output_path)
                    
                return f"❌ Error processing video: {str(e)}", 500
        else:
            return "Format file tidak didukung. Gunakan MP4, AVI, MOV, atau MKV.", 400
                
    return render_template('upload.html')

@app.errorhandler(413)
def too_large(e):
    return "File terlalu besar. Maksimal 100MB.", 413

@app.errorhandler(500)
def internal_error(e):
    return "Terjadi kesalahan internal server.", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)