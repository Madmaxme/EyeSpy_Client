# EyeSpy Client

Face detection client component of the EyeSpy system. This client captures faces from your screen or video files and sends them to the EyeSpy backend server for processing.

## Features

- **Dual Monitoring Modes**: 
  - **Screen Monitoring** (`FotoRec_client.py`): Captures faces from your screen in real-time
  - **Video Analysis** (`VideoRec_client.py`): Processes faces from video files
- **AWS Rekognition Integration**: Powerful cloud-based face detection
- **Duplicate Avoidance**: Uses AWS face collection to avoid processing the same face twice
- **Backend Connectivity**: Automatic upload of detected faces to the server
- **Face Tracking**: Intelligent tracking of faces across video frames

## Prerequisites

1. Python 3.9+
2. AWS Account with Rekognition access
3. AWS credentials with Rekognition permissions
4. EyeSpy backend server (running locally or remotely)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your AWS credentials:
```
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
AWS_REGION=eu-west-1
EYESPY_BACKEND_URL=http://localhost:8080
```

## Usage

### Screen Monitoring

Run the screen monitoring client:
```bash
python FotoRec_client.py --server http://localhost:8080
```

Options:
- `--server`: Backend server URL (default: http://localhost:8080)

### Video Analysis

Run the video analysis client:
```bash
python VideoRec_client.py --server http://localhost:8080 --video path/to/video.mp4
```

Options:
- `--server`: Backend server URL (default: http://localhost:8080)
- `--video`: Path to video file to analyze
- `--clear`: Clear the AWS face collection before processing

### Interactive Controls

The clients provide interactive controls during operation:

- Press 'q' to quit
- Press 'p' to pause/resume processing
- Press 's' to take a screenshot (VideoRec only)
- Press '+'/'-' to increase/decrease video playback speed (VideoRec only)
- Press 'd'/'f' to decrease/increase face indicator display time (VideoRec only)

## How It Works

### FotoRec_client.py (Screen Monitoring)

1. Captures the screen content at regular intervals
2. Uses AWS Rekognition to detect faces in the captured images
3. For each detected face:
   - Crops and extracts the face image
   - Checks if the face has been seen before using AWS Rekognition's face collection
   - Saves new faces to the `detected_faces` directory
   - Uploads new faces to the backend server

### VideoRec_client.py (Video Analysis)

1. Loads a video file or connects to a webcam
2. Processes video frames at a configurable playback speed
3. Uses AWS Rekognition to detect faces in each processed frame
4. Applies face tracking to maintain face identity across frames
5. For stable face detections:
   - Crops and extracts the face image
   - Checks if the face has been seen before using AWS Rekognition's face collection
   - Saves new faces to the `detected_faces` directory
   - Uploads new faces to the backend server

## Directory Structure

- `FotoRec_client.py`: Screen monitoring client
- `VideoRec_client.py`: Video analysis client
- `requirements.txt`: Client dependencies
- `detected_faces/`: Directory for storing detected face images (created on first run)

## AWS Rekognition Notes

- The client creates and manages an AWS Rekognition face collection called `eyespy-faces`
- The collection is used for deduplication and tracking of detected faces
- AWS Rekognition charges apply for API usage (approximately $1 per 1,000 face operations)
- Face processing is throttled to reduce API costs

## Troubleshooting

- **AWS Authentication Errors**: Verify AWS credentials and permissions
- **Backend Connection Issues**: Ensure the backend server is running and accessible
- **Face Detection Problems**: Ensure adequate lighting and face visibility
- **Slow Performance**: Adjust the processing interval or playback speed# EyeSpy_Client
