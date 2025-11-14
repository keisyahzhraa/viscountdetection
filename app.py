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
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))  # 🔒 IMPORTANT

# Global variables
processing_jobs = {}
model = None

def load_model():
    """Load YOLO model once at startup"""
    global model
    try:
        print("🔄 Loading YOLOv5 model...")
        # Use local model to avoid download issues
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
        model.conf = 0.4
        print("✅ YOLOv5 model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def allowed_file(filename):
    """Validate file extension and filename"""
    if not "." in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Additional security: check for dangerous filenames
    dangerous_patterns = ['.php', '.py', '.exe', '.sh', '.bat', '.cmd']
    if any(pattern in filename.lower() for pattern in dangerous_patterns):
        return False
        
    return True

def validate_video_file(file_path):
    """Basic video file validation using OpenCV"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Check if it can read at least one frame
        ret, frame = cap.read()
        cap.release()
        return ret
    except:
        return False

def get_video_duration(input_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 0

def process_video_optimized(input_path, output_path, json_output):
    """Optimized video processing with security checks"""
    global model
    
    try:
        # Security: Validate the input file is actually a video
        if not validate_video_file(input_path):
            raise ValueError("Invalid video file")
        
        # Get video info
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Security: Limit maximum processing time (30 minutes max)
        max_processing_time = 30 * 60  # 30 minutes
        estimated_time = total_frames / (fps if fps > 0 else 25)
        if estimated_time > max_processing_time:
            cap.release()
            raise ValueError(f"Video too long. Maximum processing time: {max_processing_time//60} minutes")
        
        # Reduce resolution for large videos
        if width > 1280:
            scale = 1280 / width
            width = 1280
            height = int(height * scale)
        
        # Adjust FPS for very long videos
        if total_frames > 1800:
            target_fps = min(fps, 10)
            frame_skip = max(1, int(fps / target_fps))
        else:
            frame_skip = 1
        
        print(f"🎥 Processing: {total_frames} frames, {fps} fps")
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps/frame_skip, (width, height))
        counts = []
        frame_idx = 0
        processed_frames = 0
        
        start_time = time.time()
        
        while True:
            # Security: Check processing time limit
            if time.time() - start_time > max_processing_time:
                print("⏰ Processing time limit exceeded")
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
                
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            if model is not None:
                try:
                    results = model(frame)
                    boxes = results.xyxy[0].cpu().numpy()
                    people = [b for b in boxes if int(b[5]) == 0]
                    count = len(people)
                    
                    for p in people:
                        x1, y1, x2, y2 = map(int, p[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Model inference error: {e}")
                    count = 0
            else:
                count = 0  # Don't use random numbers for security
            
            counts.append(count)
            
            cv2.putText(frame, f"People: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {processed_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            out.write(frame)
            frame_idx += 1
            processed_frames += 1
            
            if processed_frames % 50 == 0:
                progress = min(90, int((processed_frames / total_frames) * 100))
                print(f"📊 Progress: {progress}%")
        
        cap.release()
        out.release()
        
        stats = {
            "frames": len(counts),
            "avg": float(np.mean(counts)) if counts else 0,
            "max": int(np.max(counts)) if counts else 0,
            "max_frame": int(np.argmax(counts)) if counts else 0,
            "duration": get_video_duration(input_path),
            "original_frames": total_frames,
            "processed_frames": processed_frames
        }
        
        with open(json_output, "w") as f:
            json.dump(stats, f)
            
        print(f"✅ Processing complete: {stats}")
        return True
        
    except Exception as e:
        print(f"❌ Error in video processing: {e}")
        # Clean up failed output files
        for file_path in [output_path, json_output]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        return False

def cleanup_old_files(hours=6):  # Reduced from 24 to 6 hours for security
    """Clean up files older than specified hours"""
    try:
        current_time = time.time()
        cutoff = current_time - (hours * 3600)
        
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, RESULT_FOLDER]:
            if not os.path.exists(folder):
                continue
                
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                try:
                    if os.path.isfile(filepath) and os.path.getctime(filepath) < cutoff:
                        os.remove(filepath)
                        print(f"🧹 Cleaned up: {filename}")
                except Exception as e:
                    print(f"⚠️ Cleanup error for {filename}: {e}")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

def run_processing(job_id, input_path, output_path, json_path):
    """Run video processing in background thread with timeout"""
    try:
        processing_jobs[job_id] = {"status": "processing", "progress": 0}
        success = process_video_optimized(input_path, output_path, json_path)
        processing_jobs[job_id] = {"status": "done" if success else "error", "progress": 100}
        
        # Clean up input file after processing to save space
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
    threading.Thread(target=cleanup_old_files, daemon=True).start()

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'video' not in request.files:
            return render_template("upload.html", error="No file selected")
        
        file = request.files['video']
        
        if file.filename == '':
            return render_template("upload.html", error="No file selected")
        
        # Enhanced security validation
        if not allowed_file(file.filename):
            return render_template("upload.html", 
                                 error="File type not allowed. Please upload MP4, AVI, MOV, or MKV")
        
        try:
            # More secure filename generation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = secrets.token_hex(8)
            original_name = secure_filename(file.filename)
            filename = f"{timestamp}_{random_id}_{original_name}"
            
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)
            
            # Security: Check file size again after save
            file_size = os.path.getsize(input_path)
            if file_size > MAX_FILE_SIZE:
                os.remove(input_path)
                return render_template("upload.html", 
                                     error=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
            
            # Security: Validate it's actually a video file
            if not validate_video_file(input_path):
                os.remove(input_path)
                return render_template("upload.html", error="Invalid video file")
            
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
            # Security: Don't expose internal error details
            print(f"Upload error: {e}")
            return render_template("upload.html", 
                                 error="An error occurred during upload. Please try again.")
    
    return render_template("upload.html")

@app.route("/progress/<job_id>")
def progress(job_id):
    # Security: Basic validation of job_id
    if not job_id or len(job_id) > 100:
        return jsonify({"status": "not_found", "progress": 0})
    
    job = processing_jobs.get(job_id, {"status": "not_found", "progress": 0})
    return jsonify(job)

@app.route("/result/<job_id>")
def result(job_id):
    # Security: Validate job_id format
    if not job_id or len(job_id) > 100 or '..' in job_id:
        return render_template("upload.html", error="Invalid job ID")
    
    json_file = os.path.join(RESULT_FOLDER, f"{job_id}.json")
    video_file = os.path.join(PROCESSED_FOLDER, f"processed_{job_id}.mp4")
    
    if not os.path.exists(json_file) or not os.path.exists(video_file):
        return render_template("upload.html", error="Analysis not complete or job not found")
    
    try:
        with open(json_file) as f:
            stats = json.load(f)
        
        output_video = url_for("static", filename=f"processed/processed_{job_id}.mp4")
        
        return render_template(
            "result.html",
            output_video=output_video,
            stats=stats,
            job_id=job_id
        )
    except Exception as e:
        print(f"Result error: {e}")
        return render_template("upload.html", error="Error loading results")

@app.route("/download/<job_id>")
def download_video(job_id):
    # Security: Validate job_id
    if not job_id or len(job_id) > 100 or '..' in job_id:
        return "File not found", 404
    
    video_file = os.path.join(PROCESSED_FOLDER, f"processed_{job_id}.mp4")
    if os.path.exists(video_file):
        return send_file(video_file, as_attachment=True, 
                        download_name=f"viscount_result_{job_id}.mp4")
    return "File not found", 404

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "processing_jobs": len(processing_jobs)
    })

# Security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Never run with debug=True in production
    app.run(host="0.0.0.0", port=port, debug=False)