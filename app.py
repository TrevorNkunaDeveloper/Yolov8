from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

# YOLO model
model = YOLO('yolov8n.pt')

# Upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

recording = False
out = None
incoming_car_count = 0
outgoing_car_count = 0
total_car_count = 0
other_objects_count = 0

# Initialize counts for lanes and directions
lane_counts = [0, 0, 0, 0, 0, 0, 0, 0]  # Assuming 8 lanes for simplicity

# Define colors for each lane
lane_colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (0, 128, 128)   # Teal
]

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen():
    global recording, out, incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)
        annotated_frame = results[0].plot()  # Correctly access and plot the first result

        # Reset counts for other objects
        other_objects_count = 0
        lane_counts = [0, 0, 0, 0, 0, 0, 0, 0]

        height, width, _ = frame.shape
        lane_width = width / len(lane_counts)

        # Draw lane rectangles
        for i in range(len(lane_counts)):
            x1 = int(i * lane_width)
            x2 = int((i + 1) * lane_width)
            cv2.rectangle(annotated_frame, (x1, 0), (x2, height), lane_colors[i], 2)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Determine the lane based on the center x-coordinate
            lane_index = int(cx // lane_width)

            # Assuming class '2' is 'car'; adjust based on your model's class labels
            if box.cls == 2:
                lane_counts[lane_index] += 1
                total_car_count += 1
                if lane_index < 4:
                    incoming_car_count += 1
                else:
                    outgoing_car_count += 1
            else:
                other_objects_count += 1

        print(f"Incoming cars: {incoming_car_count}, Outgoing cars: {outgoing_car_count}")
        print(f"Lane counts: {lane_counts}, Total cars: {total_car_count}")

        if recording and out is not None:
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def get_counts():
    return jsonify(
        incoming_car_count=incoming_car_count,
        outgoing_car_count=outgoing_car_count,
        total_car_count=total_car_count,
        other_objects_count=other_objects_count,
        lane_counts=lane_counts
    )

@app.route('/reset_counts')
def reset_counts():
    global incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts
    incoming_car_count = 0
    outgoing_car_count = 0
    total_car_count = 0
    other_objects_count = 0
    lane_counts = [0, 0, 0, 0, 0, 0, 0, 0]
    return '', 204

@app.route('/start_recording')
def start_recording():
    global recording, out
    if not recording:
        recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'output_{int(time.time())}.avi', fourcc, 20.0, (640, 480))
        print("Recording started")
    return '', 204

@app.route('/stop_recording')
def stop_recording():
    global recording, out
    if recording:
        recording = False
        if out is not None:
            out.release()
            out = None
        print("Recording stopped")
    return '', 204

@app.route('/take_picture')
def take_picture():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'frame_{int(time.time())}.jpg', frame)
    return '', 204

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('process_video', filename=filename))
    return 'File not allowed'

@app.route('/process_video/<filename>')
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    global incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts
    return render_template('process_video.html', filename=filename, video_path=filepath, incoming_car_count=incoming_car_count, outgoing_car_count=outgoing_car_count, total_car_count=total_car_count, other_objects_count=other_objects_count, lane_counts=lane_counts)

def gen_video(video_path):
    global incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()  # Correctly access and plot the first result

        # Reset counts
        incoming_car_count = 0
        outgoing_car_count = 0
        total_car_count = 0
        other_objects_count = 0
        lane_counts = [0, 0, 0, 0, 0, 0, 0, 0]

        height, width, _ = frame.shape
        lane_width = width / len(lane_counts)

        # Draw lane rectangles and count objects
        for i in range(len(lane_counts)):
            x1 = int(i * lane_width)
            x2 = int((i + 1) * lane_width)
            cv2.rectangle(annotated_frame, (x1, 0), (x2, height), lane_colors[i], 2)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2

            # Determine the lane based on the center x-coordinate
            lane_index = int(cx // lane_width)

            # Assuming class '2' is 'car'; adjust based on your model's class labels
            if box.cls == 2:
                lane_counts[lane_index] += 1
                total_car_count += 1
                if lane_index < 4:
                    incoming_car_count += 1
                else:
                    outgoing_car_count += 1
            else:
                other_objects_count += 1

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(gen_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('process_image', filename=filename))
    return 'File not allowed'

@app.route('/process_image/<filename>')
def process_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_image_path = process_uploaded_image(filepath)
    global incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts
    return render_template('process_image.html', filename=filename, processed_image_path=processed_image_path, incoming_car_count=incoming_car_count, outgoing_car_count=outgoing_car_count, total_car_count=total_car_count, other_objects_count=other_objects_count, lane_counts=lane_counts)

def process_uploaded_image(image_path):
    global incoming_car_count, outgoing_car_count, total_car_count, other_objects_count, lane_counts

    frame = cv2.imread(image_path)
    results = model(frame)
    annotated_frame = results[0].plot()  # Correctly access and plot the first result

    # Reset counts
    incoming_car_count = 0
    outgoing_car_count = 0
    total_car_count = 0
    other_objects_count = 0
    lane_counts = [0, 0, 0, 0, 0, 0, 0, 0]

    height, width, _ = frame.shape
    lane_width = width / len(lane_counts)

    # Draw lane rectangles and count objects
    for i in range(len(lane_counts)):
        x1 = int(i * lane_width)
        x2 = int((i + 1) * lane_width)
        cv2.rectangle(annotated_frame, (x1, 0), (x2, height), lane_colors[i], 2)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx = (x1 + x2) / 2

        # Determine the lane based on the center x-coordinate
        lane_index = int(cx // lane_width)

        # Assuming class '2' is 'car'; adjust based on your model's class labels
        if box.cls == 2:
            lane_counts[lane_index] += 1
            total_car_count += 1
            if lane_index < 4:
                incoming_car_count += 1
            else:
                outgoing_car_count += 1
        else:
            other_objects_count += 1

    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, annotated_frame)
    return processed_image_path

@app.route('/processed_image/<filename>')
def processed_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
