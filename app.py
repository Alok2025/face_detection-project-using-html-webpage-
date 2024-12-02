import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Face Detection solution
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the face detection model with high accuracy
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

# Initialize webcam feed
cap = cv2.VideoCapture(0)
face_tracker = {}
face_id = 0

# Flag to control webcam feed
webcam_active = False


@app.route('/')
def index():
    return render_template('index.html')


# Video streaming route
def gen_frames():
    global face_tracker, face_id
    while True:
        if webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB (MediaPipe works with RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect faces
            results = face_detection.process(rgb_frame)

            # List to track new face detections
            new_faces = []

            if results.detections:
                for detection in results.detections:
                    # Get the bounding box of each face
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Check if the face is already in the tracker
                    assigned_id = None
                    for f_id, (fx, fy, fw, fh) in face_tracker.items():
                        if abs(fx - x) < 50 and abs(fy - y) < 50:
                            assigned_id = f_id
                            break

                    if assigned_id is None:
                        face_id += 1
                        assigned_id = face_id
                        face_tracker[assigned_id] = (x, y, w, h)
                    else:
                        face_tracker[assigned_id] = (x, y, w, h)

                    # Draw bounding box and the face ID
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode the frame to JPEG format for web streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active
    webcam_active = True
    return "Webcam Started"


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return "Webcam Stopped"


@app.route('/exit_webcam', methods=['POST'])
def exit_webcam():
    global webcam_active
    webcam_active = False
    cap.release()
    return "Exiting Webcam"


# Start the Flask app
if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=5000,debug=True)
