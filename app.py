# pip install flask opencv-python-headless torch torchvision numpy ultralytics


from flask import Flask, render_template, Response, request
import cv2
from yolo import YOLO

app = Flask(__name__)
yolo = YOLO()

def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 for webcam, or use an IP camera URL
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = yolo.detect(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', yolo=yolo)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
