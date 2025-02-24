from flask import Flask, render_template, request, send_file
import os
import cv2
import dlib
import librosa
import numpy as np
import moviepy.editor as mp
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set upload and processed folders
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------------------------
# 1. Face detection and extraction
# -------------------------------------------------
def extract_faces(image_path):
    """
    Detect faces using OpenCV Haar cascades and dlib,
    and crop detected faces to 100x100 images.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First, try detecting with OpenCV
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))
    # If no faces detected with OpenCV, use dlib
    if len(faces) == 0:
        detector = dlib.get_frontal_face_detector()
        faces_dlib = detector(gray)
        faces = [(rect.left(), rect.top(), rect.width(), rect.height()) for rect in faces_dlib]

    face_images = []
    for i, (x, y, w, h) in enumerate(faces):
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))  # Resize face to 100x100
        face_path = os.path.join(PROCESSED_FOLDER, f"face_{i}.png")
        cv2.imwrite(face_path, face_resized)
        face_images.append(face_path)
    return face_images

# -------------------------------------------------
# 2. Extract audio rhythm (BPM)
# -------------------------------------------------
def extract_audio_rhythm(audio_path):
    """
    Extract the audio rhythm (BPM) using librosa.
    """
    y, sr = librosa.load(audio_path, sr=22050)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    print(f"Calculated BPM: {tempo}")
    if tempo is None or tempo <= 0:
        tempo = 120  # Default BPM
    return float(tempo)

# -------------------------------------------------
# 3. Create video with movement using optional background image
# -------------------------------------------------
def create_video_with_movement(face_image_path, audio_path, background_image_path=None):
    """
    Generate a video where the face image swings according to the audio rhythm,
    and place it in the bottom-right corner. If a background image is provided,
    it will be used as the background; otherwise, a black background is used.
    """
    bpm = extract_audio_rhythm(audio_path)  # Get BPM
    duration = librosa.get_duration(filename=audio_path)  # Audio duration
    print(f"Audio duration: {duration}")

    # Read and resize face image to 100x100
    face_img = cv2.imread(face_image_path)
    face_img = cv2.resize(face_img, (100, 100))

    # Define video dimensions
    width, height = 400, 400
    num_frames = int(duration * 30)  # Assume 30 FPS
    print(f"Number of frames: {num_frames}")

    # Load background image if provided, otherwise create a black background
    if background_image_path:
        bg_img = cv2.imread(background_image_path)
        bg_img = cv2.resize(bg_img, (width, height))
    else:
        bg_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate a time sequence covering the entire audio duration
    time_seq = np.linspace(0, duration, num_frames)
    # Calculate rotation angle for each frame using sine function for periodic swinging
    # Formula: angle = sin(2π*(BPM/60)*time) * 15, range ±15°
    angles = np.sin(2 * np.pi * (bpm / 60) * time_seq) * 15
    angles = angles.astype(float)
    print("First 5 angles:", angles[:5])
    print("Last 5 angles:", angles[-5:])

    frames = []
    for angle in angles:
        # Compute rotation matrix (center of face image is (50, 50))
        matrix = cv2.getRotationMatrix2D((50, 50), float(angle), 1)
        rotated = cv2.warpAffine(face_img, matrix, (100, 100))
        # Use a copy of the background image as the frame background
        frame = bg_img.copy()
        # Place the rotated face at the bottom-right corner (with 10-pixel margin)
        frame[height - 110:height - 10, width - 110:width - 10] = rotated
        frames.append(frame)

    # Create video using MoviePy's ImageSequenceClip (convert BGR to RGB)
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    clip = mp.ImageSequenceClip(rgb_frames, fps=30)
    audio_clip = mp.AudioFileClip(audio_path)
    clip = clip.set_audio(audio_clip)

    video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")
    clip.write_videofile(video_path, codec="libx264")
    return video_path

# -------------------------------------------------
# 4. Flask main route (upload photo, audio, and optional background)
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check for required files (photo and audio)
        if "photo" not in request.files or "audio" not in request.files:
            return "❌ Please upload both a photo and an audio file!"
        photo = request.files["photo"]
        audio = request.files["audio"]

        if photo.filename == "" or audio.filename == "":
            return "❌ Required file not selected!"

        # Save photo and audio
        photo_path = os.path.join(UPLOAD_FOLDER, secure_filename(photo.filename))
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio.filename))
        photo.save(photo_path)
        audio.save(audio_path)

        # Check for optional background image
        background = request.files.get("background", None)
        background_path = None
        if background and background.filename != "":
            background_path = os.path.join(UPLOAD_FOLDER, secure_filename(background.filename))
            background.save(background_path)

        # Extract faces from photo
        face_images = extract_faces(photo_path)
        if not face_images:
            return "❌ No face detected, please upload an image with a face!"

        # Create video using the first detected face
        video_path = create_video_with_movement(face_images[0], audio_path, background_image_path=background_path)
        return send_file(video_path, as_attachment=True)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
