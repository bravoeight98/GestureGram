from flask import Flask, render_template, url_for, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import openai
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.static_folder = 'static'  # Set static folder path

# Load your model and labels (adjust paths as needed)
model = load_model("model.h5")
labels = np.load("labels.npy")

# Initialize MediaPipe holistic model
holistic = mp.solutions.holistic.Holistic()

# Global variable to store the prediction
pred = "Initial Prediction"

# Initialize OpenAI API client
openai.api_key = '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'

# Initialize an empty list to store prediction values
pred_history = []

def generate_frames():
    cap = cv2.VideoCapture(0)
    global pred  # Access the global variable
    global pred_history

    
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
            pred_history.append(pred)
            cv2.putText(frame, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            print(pred)

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

@app.route('/set_feed')
def set_feed():
    return render_template('set_feed.html')

@app.route('/video_frame_stream')
def video_frame_stream():
    """Generate video frames as a response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pred')
def get_pred():
    global pred  # Access the global variable
    return pred

@app.route('/predhistory')
def get_predhistory():
    global pred_history
    return jsonify(pred_history)

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    text_to_speak = request.json['text']
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=text_to_speak,
        max_tokens=30,
        temperature=0.8,
        n=1,
        stop=None
    )
    audio_url = response['choices'][0]['audio']
    return jsonify({'audio_url': audio_url})

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for easier development
