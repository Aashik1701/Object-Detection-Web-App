import argparse
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
import os
from pathlib import Path
import urllib.parse
from ultralytics import YOLO
from utils import downloads
from torch.cuda import is_available

MODEL = YOLO('yolov8n.pt')
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
COLORS = np.random.uniform(0, 255, size=(80, 3)) # random colors for different classes
imgpath = None
device =  "cuda:0" if is_available() else "cpu"
video_path = None
stream = False
its_image = False
PROJECT_PATH = "runs/detect"
os.makedirs(f"{PROJECT_PATH}", exist_ok=True)
DOWNLOADS_FOLDER = "uploads"
os.makedirs(f"{DOWNLOADS_FOLDER}", exist_ok=True)
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth

def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name

def check_file(file, download=True):
    """Search/download file (if necessary) and return path."""
    file = str(file).strip()  # convert to string and strip spaces
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    if download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(f"{DOWNLOADS_FOLDER}/{file}").exists():
            print(f"Found {clean_url(url)} locally at {file}")  # file already exists
        else:
            filename = downloads.safe_download(url=url, file=file, dir=DOWNLOADS_FOLDER, unzip=False)
            print(f"Downloaded {clean_url(url)} to {filename}")  # download
        return filename 

def check_source(source):
    if isinstance(source, (str, int, Path)):
        source = str(source)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        # is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        if is_url:
            print(f"Warning: Ambiguous source '{source}', assuming URL.")
            source = check_file(source)  # download
    else:
        return

    return source, is_url


@app.route("/", methods=["GET", "POST"])
def predict_img():
    global imgpath, its_image, video_path
    if request.method == "POST":
        if 'text' in request.form:
            video_link = request.form['text']
            filepath, is_url = check_source(video_link) # https://drive.google.com/file/d/1OhWnaYeOtMRidwbRpawgoUGhCs2g046h/view?usp=sharing
            if is_url:
                filepath = str(check_file(filepath))
                filename = filepath.split('/')[-1]
            predict_img.imgpath = filepath
            print("uploaded file path: ", filepath)
            imgpath, file_extension = (filename.rsplit('.', 1)[0].lower(), filename.rsplit('.', 1)[1].lower())
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,DOWNLOADS_FOLDER, f.filename)
            f.save(filepath)
            predict_img.imgpath = f.filename
                                               
            imgpath, file_extension = (f.filename.rsplit('.', 1)[0].lower(), f.filename.rsplit('.', 1)[1].lower()) 
            
        if os.path.exists(f"{PROJECT_PATH}"):
            os.makedirs(f"{PROJECT_PATH}/{imgpath}", exist_ok=True)
        else:
            os.makedirs(f"runs", exist_ok=True)
            os.makedirs(f"{PROJECT_PATH}")
            os.makedirs(f"{PROJECT_PATH}/{imgpath}")

        if file_extension in IMG_FORMATS:
            its_image = True
            frame = cv2.imread(filepath)
            # Inference
            results = MODEL.predict(
                frame, show=False, verbose=False, save=False, device="cuda:0", conf=0.5
            )

            # Check if robot is detected
            if results[0].boxes.cpu().numpy().xyxy.shape[0] != 0:
                # Show results on image
                boxes = results[0].boxes.cpu().numpy().xyxy.astype(int)
                labels = results[0].boxes.cpu().numpy().cls
                conf = results[0].boxes.cpu().numpy().conf
                for box, label, conf in zip(boxes, labels, conf):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(label)], 4)
                    cv2.putText(
                        frame,
                        MODEL.names[int(label)] + ": " + str(round(conf, 2)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        COLORS[int(label)],
                        4,
                    )
                cv2.imwrite(f"{PROJECT_PATH}/{imgpath}/output.jpg", frame)
                image_feed()
                return render_template('index.html')  
        
        elif file_extension in VID_FORMATS:
            its_image = False
            video_path = filepath
            video_feed()
            return render_template('index.html')

    return render_template('index.html')


def get_video_frame():
    global imgpath, video_path
    if video_path==None:
        mp4_files = f"{PROJECT_PATH}/{imgpath}/output.mp4"
        video = cv2.VideoCapture(mp4_files)  # detected video path
        while True:
            success, image = video.read()
            if not success:
                break
            _, jpeg = cv2.imencode('.jpg', image) 
        
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        video.release()
    else:
        cap = cv2.VideoCapture(video_path)
        # Initialize variables
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            f"{PROJECT_PATH}/{imgpath}/output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = MODEL.predict(
                frame, show=False, verbose=False, save=False, device="cuda:0", conf=0.5
            )

            # Check if robot is detected
            if results[0].boxes.cpu().numpy().xyxy.shape[0] != 0:
                # Show results on image
                boxes = results[0].boxes.cpu().numpy().xyxy.astype(int)
                labels = results[0].boxes.cpu().numpy().cls
                conf = results[0].boxes.cpu().numpy().conf
                for box, label, conf in zip(boxes, labels, conf):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(label)], 2)
                    cv2.putText(
                        frame,
                        MODEL.names[int(label)] + ": " + str(round(conf, 2)),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        COLORS[int(label)],
                        2,
                    )
            out.write(frame)
            cv2.imwrite(f"{PROJECT_PATH}/{imgpath}/output.jpg", frame)
        
            _,jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        cap.release()
        out.release() 


def get_image_frame():
    global imgpath, its_image
    if its_image:
        img_files = f"{PROJECT_PATH}/{imgpath}/output.jpg"
        image = cv2.imread(img_files)
        _, jpeg = cv2.imencode('.jpg', image) 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    else: # clear the image
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

@app.route("/image_feed")
def image_feed():
    return Response(get_image_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_video_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    # Start the Flask app in a separate thread
    app.run(host='0.0.0.0', port= args.port, debug=True, threaded=True)