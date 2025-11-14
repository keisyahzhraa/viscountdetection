import os
import cv2
import json
import torch
import numpy as np
import threading
from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_yolo(input_path, output_path, json_output):

    # ==== LOAD YOLO DI DALAM (BUKAN GLOBAL!!) ====
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    model.conf = 0.4

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = []

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results.xyxy[0].cpu().numpy()

        people = [b for b in boxes if int(b[5]) == 0]

        for p in people:
            x1, y1, x2, y2 = map(int, p[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        count = len(people)
        counts.append(count)

        cv2.putText(frame, f"People: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        out.write(frame)

    cap.release()
    out.release()

    stats = {
        "frames": len(counts),
        "avg": float(np.mean(counts)) if counts else 0,
        "max": int(np.max(counts)) if counts else 0,
        "max_frame": int(np.argmax(counts)) if counts else 0,
    }

    with open(json_output, "w") as f:
        json.dump(stats, f)


processing_jobs = {}

def run_background(job_id, input_path, output_path, json_path):
    process_video_yolo(input_path, output_path, json_path)
    processing_jobs[job_id] = "done"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("video")

        if not file or file.filename == "":
            return "File belum dipilih", 400

        if not allowed_file(file.filename):
            return "Format tidak valid", 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")

        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        base = os.path.splitext(filename)[0]
        output_name = f"processed_{base}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        json_path = os.path.join(RESULT_FOLDER, f"{base}.json")

        job_id = base
        processing_jobs[job_id] = "processing"

        threading.Thread(
            target=run_background,
            args=(job_id, input_path, output_path, json_path),
            daemon=True
        ).start()

        return render_template("upload.html", job_id=job_id)

    return render_template("upload.html")


@app.route("/progress/<job_id>")
def progress(job_id):
    return jsonify({"status": processing_jobs.get(job_id, "not_found")})


@app.route("/result/<job_id>")
def result(job_id):
    json_file = os.path.join(RESULT_FOLDER, f"{job_id}.json")

    if not os.path.exists(json_file):
        return "Belum selesai", 404

    with open(json_file) as f:
        stats = json.load(f)

    return render_template(
        "result.html",
        output_video=url_for("static", filename=f"processed/processed_{job_id}.mp4"),
        stats=stats
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))