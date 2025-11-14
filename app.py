import os
import cv2
import json
import torch
import numpy as np
import threading
import time
import secrets
from flask import Flask, request, render_template, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime

# Configuration
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed" 
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # Reduced to 100MB for Railway

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Global variables
processing_jobs = {}
model = None

def load_model():
    """Load YOLO model once at startup with memory optimization"""
    global model
    try:
        print("🔄 Loading YOLOv5 model...")
        
        # Clear cache to save memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load model with optimizations
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
        model.conf = 0.4
        
        # Use CPU only to save memory
        model.to('cpu')
        
        # Set model to evaluation mode
        model.eval()
        
        print("✅ YOLOv5 model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_fast(input_path, output_path, json_output):
    """Fast video processing with optimizations for Railway"""
    global model
    
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Reduce resolution for faster processing
        if width > 640:  # Reduced from 1280 to 640
            scale = 640 / width
            width = 640
            height = int(height * scale)
        
        # Skip frames for very long videos
        frame_skip = 3 if fps > 15 else 1
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps/frame_skip, (width, height))
        counts = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
                
            # Resize frame
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            # Process with YOLO
            if model is not None:
                results = model(frame)
                boxes = results.xyxy[0].cpu().numpy()
                people = [b for b in boxes if int(b[5]) == 0]
                count = len(people)
                
                # Draw bounding boxes (simplified)
                for p in people:
                    x1, y1, x2, y2 = map(int, p[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thinner lines
            else:
                count = 0
            
            counts.append(count)
            
            # Simplified text overlay
            cv2.putText(frame, f"People: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Smaller text
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Calculate statistics
        stats = {
            "frames": len(counts),
            "avg": float(np.mean(counts)) if counts else 0,
            "max": int(np.max(counts)) if counts else 0,
            "max_frame": int(np.argmax(counts)) if counts else 0,
        }
        
        with open(json_output, "w") as f:
            json.dump(stats, f)
            
        return True
        
    except Exception as e:
        print(f"❌ Error in video processing: {e}")
        return False

def run_processing(job_id, input_path, output_path, json_path):
    """Run video processing in background thread"""
    try:
        processing_jobs[job_id] = {"status": "processing", "progress": 0}
        success = process_video_fast(input_path, output_path, json_path)
        processing_jobs[job_id] = {"status": "done" if success else "error", "progress": 100}
        
        # Clean up input file after processing
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass
            
    except Exception as e:
        print(f"❌ Background job error: {e}")
        processing_jobs[job_id] = {"status": "error", "progress": 0}

# Load model when app starts
@app.before_first_request
def startup():
    load_model()

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'video' not in request.files:
            return render_template("upload.html", error="No file selected")
        
        file = request.files['video']
        
        if file.filename == '':
            return render_template("upload.html", error="No file selected")
        
        if not allowed_file(file.filename):
            return render_template("upload.html", error="File type not allowed. Please upload MP4, AVI, MOV, or MKV")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = secure_filename(file.filename)
            filename = f"{timestamp}_{original_name}"
            
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)
            
            file_size = os.path.getsize(input_path)
            if file_size > MAX_FILE_SIZE:
                os.remove(input_path)
                return render_template("upload.html", error=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
            
            base = os.path.splitext(filename)[0]
            output_name = f"processed_{base}.mp4"
            output_path = os.path.join(PROCESSED_FOLDER, output_name)
            json_path = os.path.join(RESULT_FOLDER, f"{base}.json")
            
            job_id = base
            processing_jobs[job_id] = {"status": "queued", "progress": 0}
            
            thread = threading.Thread(
                target=run_processing,
                args=(job_id, input_path, output_path, json_path),
                daemon=True
            )
            thread.start()
            
            return render_template("upload.html", job_id=job_id)
            
        except Exception as e:
            return render_template("upload.html", error=f"Upload error: {str(e)}")
    
    return render_template("upload.html")

# ... (rest of the routes remain the same as your previous app.py)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)