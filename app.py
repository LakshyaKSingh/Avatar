from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import torch
from AnimeGANAvatar import AnimeGANAvatar
from pathlib import Path
import mediapipe as mp
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load overlay images
robot_face_image = cv2.imread("robot_face.png", cv2.IMREAD_UNCHANGED)
sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
thug_life_image = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
spiderman_image = cv2.imread("spiderman.png", cv2.IMREAD_UNCHANGED)
alien_image = cv2.imread("alien.png", cv2.IMREAD_UNCHANGED)

# Initialize the AnimeGANAvatar class
avatar_generator = AnimeGANAvatar()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

output_dir = Path("captured_images")
output_dir.mkdir(exist_ok=True)

def apply_bw_filter(image):
    """Apply enhanced black and white filter to the image."""
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.convertScaleAbs(bw_image, alpha=1.5, beta=0)
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

def overlay_with_autofit(image, overlay_image, face_coordinates, is_lens_filter=False):
    """Overlay an image on the detected face region with autofit."""
    for (x, y, w, h) in face_coordinates:
        if is_lens_filter:
            overlay_resized = cv2.resize(overlay_image, (w, int(h / 2)), interpolation=cv2.INTER_AREA)
            y_offset = y + int(h / 4)
            for i in range(overlay_resized.shape[0]):
                for j in range(overlay_resized.shape[1]):
                    if overlay_resized[i, j][3] != 0:
                        image[y_offset + i, x + j] = overlay_resized[i, j][:3]
        else:
            overlay_resized = cv2.resize(overlay_image, (w, h + int(h / 4)), interpolation=cv2.INTER_AREA)
            for i in range(h + int(h / 4)):
                for j in range(w):
                    if overlay_resized[i, j][3] != 0:
                        image[y + i - int(h / 4), x + j] = overlay_resized[i, j][:3]
    return image

def process_image(img, filter_option):
    """Process the image with the selected filter."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    img_with_overlay = img.copy()

    if results.detections:
        face_coordinates = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Adjust for better fit
            if filter_option in ["Sunglasses", "Thug Life"]:
                y -= int(0.05 * height)
                height = int(0.6 * height)
            else:
                x -= int(0.05 * width)
                y -= int(0.1 * height)
                width = int(1.1 * width)
                height = int(1.2 * height)

            face_coordinates.append((x, y, width, height))

        if filter_option == "Robot":
            img_with_overlay = overlay_with_autofit(img.copy(), robot_face_image, face_coordinates)
        elif filter_option == "Sunglasses":
            img_with_overlay = overlay_with_autofit(img.copy(), sunglasses_image, face_coordinates, is_lens_filter=True)
        elif filter_option == "Thug Life":
            img_with_overlay = overlay_with_autofit(img.copy(), thug_life_image, face_coordinates, is_lens_filter=True)
        elif filter_option == "SpiderMan":
            img_with_overlay = overlay_with_autofit(img.copy(), spiderman_image, face_coordinates)
        elif filter_option == "Alien":
            img_with_overlay = overlay_with_autofit(img.copy(), alien_image, face_coordinates)
        elif filter_option == "B/W":
            img_with_overlay = apply_bw_filter(img.copy())

    anime_image = avatar_generator.process_frame(img_with_overlay)
    return anime_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filter_option = request.form.get('filter_option', 'Avatar')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    processed_image = process_image(img, filter_option)

    _, buffer = cv2.imencode('.png', processed_image)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    _, buffer = cv2.imencode('.png', img)
    original_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'original_image': original_image_base64,
        'processed_image': processed_image_base64
    })

@app.route('/process_webcam', methods=['POST'])
def process_webcam_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    filter_option = data.get('filter_option', 'Avatar')
    
    processed_image = process_image(img, filter_option)

    _, buffer = cv2.imencode('.png', processed_image)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'processed_image': processed_image_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
