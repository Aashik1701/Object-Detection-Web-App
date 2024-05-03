# Object Detection Web Application

Welcome to the Object Detection Web Application repository! This project implements a Flask web application for object detection using YOLOv8 model trained on the COCO dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Object Detection Web Application allows users to upload images or provide downloadable links to images or videos. The application performs object detection using the YOLOv8 model and displays the results with bounding boxes and labels for detected objects.

## Features

- Upload images or provide downloadable links for object detection.
- Real-time object detection for images and videos.
- Frame-by-frame analysis for video files.
- Detection of objects from 81 classes provided by the COCO dataset.
- Easy deployment on Google Cloud Platform or any other hosting service.

## Installation

To run the Object Detection Web Application locally, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/object-detection-web-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd object-detection-web-app
   ```

3. Install the required Python packages:
   
   ```bash
   pip3 install -r requirements.txt
   ```
   For Linux (optional): 
   ```bash
   export PATH="$PATH:~/.local/bin"
   ```

## Usage

1. Start the Flask app:

   ```bash
   python3 real-time.py
   ```

2. Open a web browser and navigate to `http://localhost:8000` to access the application.

3. Upload an image/video file or provide a accessible link (example given below) to an image or video for object detection-
   ```bash
   https://www.youtube.com/watch?v=ddTV12hErTc&t
   ```

### Note: Most of the Youtube videos require special permissions to be accesssible via python API. Please make sure the video is accessible before testing.

## Demo

To see a live demo of the Object Detection Web Application, visit [Demo](http://34.125.36.167:8000/).

## Contributing

Contributions to the Object Detection Web Application are welcome! If you find any bugs, have feature requests, or want to contribute improvements, please submit an issue or pull request.

## License

All copyrights are reserved by [Kamlesh364](https://github.com/kamlesh364).
