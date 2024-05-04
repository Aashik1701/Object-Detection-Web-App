import argparse
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, flash
import os
from ultralytics import YOLO
from torch.cuda import is_available

MODEL = YOLO('yolov8n.pt')
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
COLORS = np.random.uniform(0, 255, size=(80, 3)) # random colors for different classes
imgpath = None
device =  "cuda:0" if is_available() else "cpu"
video_path = None
source_link = None
its_image = False
PROJECT_PATH = "runs/detect"
os.makedirs(f"{PROJECT_PATH}", exist_ok=True)
DOWNLOADS_FOLDER = "uploads"
os.makedirs(f"{DOWNLOADS_FOLDER}", exist_ok=True)
app = Flask(__name__)
app.secret_key = 'ee_267_sjsu$2024'

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    global imgpath, its_image, video_path, device, source_link 
    video_path = None
    imgpath = None
    source_link = None
    its_image = False
    if request.method == "POST":
        if request.files:
            if len(request.files["file"].filename)==0:
                if 'text' in request.form:
                    source_link = request.form['text']
                    video_feed()
                    return render_template('index.html')
                else:
                    flash("Please upload a file first to proceed!")
                    return render_template('index.html')
            else:
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
                    if frame is None:
                        flash("Could not read Image, Please upload a valid image file!")
                        return render_template('index.html')
                    # Inference
                    results = MODEL.predict(
                        frame, show=False, verbose=False, save=False, device=device, conf=0.5
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

        if request.form:
            if len(request.form["text"])==0:
                flash("Please provide a link first to proceed!")
                return render_template('index.html')
            else:
                if 'text' in request.form:
                    source_link = request.form['text']
                    video_feed()
                    return render_template('index.html')

    return render_template('index.html')


def get_video_frame():
    global imgpath, video_path, device, source_link

    if video_path!=None and source_link==None:
        pth = video_path
        video_path = None
        imgpath = None
        cap = cv2.VideoCapture(pth)
        if not cap.isOpened():
            flash("Error opening video file, please upload a valid video!")
            return render_template('index.html')
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
                frame, show=False, verbose=False, save=False, device=device, conf=0.5
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
    
    elif source_link!=None:
        print("source_link: ", source_link)
        # Inference
        pth = source_link
        source_link = None
        results = MODEL.predict(
            pth, show=False, verbose=False, save=False, device=device, conf=0.5, stream=True
        )
        while True:
            for result in results:
                frame = result.orig_img
                # Show results on image
                boxes = result.boxes.cpu().numpy().xyxy.astype(int)
                labels = result.boxes.cpu().numpy().cls
                conf = result.boxes.cpu().numpy().conf
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
                _,jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    else: # clear the image
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

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
    imgpath = None
    its_image = False

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