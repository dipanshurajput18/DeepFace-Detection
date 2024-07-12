from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained model
facetracker = tf.keras.models.load_model('facetracker.h5')

# Define the video capture object
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = frame[50:500, 50:500, :]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120,120))
            yhat = facetracker.predict(np.expand_dims(resized/255,0))
            sample_coords = yhat[1][0]
            
            if yhat[0] > 0.5: 
                cv2.rectangle(frame, 
                              tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                              tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                              (255,0,0), 2)
                cv2.rectangle(frame, 
                              tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                            [0,-30])),
                              tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                            [80,0])), 
                              (255,0,0), -1)
                cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                       [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
