import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from test import detect_hand_sign

app = Flask(__name__)

camera_on = True
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


last_prediction = ""
last_confidence = 0
history = []

CONF_THRESHOLD = 80


def generate_frames():
    global last_prediction, last_confidence

    while True:
        if camera_on:
            success, frame = cap.read()
            if not success:
                continue

            frame, char, conf = detect_hand_sign(frame)

            if char:
                last_prediction = char
                last_confidence = conf
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA OFF",
                        (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera', methods=['POST'])
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return jsonify({"camera": camera_on})


@app.route('/prediction')
def prediction():
    return jsonify({
        "char": last_prediction,
        "confidence": last_confidence,
        "history": "".join(history)
    })


@app.route('/save', methods=['POST'])
def save_char():
    if last_prediction and last_confidence >= CONF_THRESHOLD:
        history.append(last_prediction)
    return jsonify({"status": "saved"})


@app.route('/space', methods=['POST'])
def add_space():
    history.append(" ")
    return jsonify({"status": "space"})


@app.route('/backspace', methods=['POST'])
def backspace():
    if history:
        history.pop()
    return jsonify({"status": "backspace"})


@app.route('/clear', methods=['POST'])
def clear_all():
    history.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)
