from flask import Flask, render_template, url_for, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.static_folder = 'static'  # Set static folder path

# Load your model and labels (adjust paths as needed)
model = load_model("model.h5")
labels = np.load("labels.npy")

# Initialize MediaPipe holistic model
holistic = mp.solutions.holistic.Holistic()

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame with MediaPipe
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Gesture detection logic
        if results.face_landmarks:
            lst = []
            for i in results.face_landmarks.landmark:
                lst.append(i.x - results.face_landmarks.landmark[1].x)
                lst.append(i.y - results.face_landmarks.landmark[1].y)

            for hand_results in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_results:
                    for i in hand_results.landmark:
                        lst.append(i.x - hand_results.landmark[8].x)
                        lst.append(i.y - hand_results.landmark[8].y)
                else:
                    lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = labels[np.argmax(model.predict(lst))]
            cv2.putText(frame, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.face_landmarks, mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Flask app code

@app.route('/')
def index():
    return render_template('index.html')  # Replace with your HTML template

@app.route('/video_feed')
def video_feed():
    return render_template('video_feed.html')

#@app.route('/set_feed')
#def video_feed():
#    return render_template('set_feed.html')

@app.route('/video_frame_stream')
def video_frame_stream():
    """Generate video frames as a response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for easier development
