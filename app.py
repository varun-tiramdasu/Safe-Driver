import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance
import face_recognition
from ultralytics import YOLO
import pygame
from flask import Flask, render_template, Response, request, redirect, url_for,session, flash
from flask_session import Session
from pymongo import MongoClient
import threading
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
import bcrypt
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from scipy.signal import medfilt



# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
# Initialize session
app.config['SECRET_KEY'] = 'your_secret_key'  # Set your own secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  # Initialize session

# Initialize MongoDB
# Corrected MongoDB connection
client = MongoClient("mongodb+srv://nikhathmahammad12:AN2bMxZTXGndN4or@cluster0.asogv.mongodb.net/?retryWrites=true&w=majority")
db = client["Drowsi"]  # Replace with your database name
users_collection = db["Drowsi"]  # Replace with your collection name

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for sound
pygame.mixer.init()

# Load the alert sound (make sure you have a sound file named "alert.wav" in the same directory)
alert_sound = pygame.mixer.Sound("music2.mp3")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

model = YOLO("mobile_detection.pt")

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):

    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (2.0 * D)
    return mar

# Real-time webcam processing with combined head pose and drowsiness detection
def combined_detection(email):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.42
    score = 0
    count = 0
    ear_history = []
    EAR_WINDOW = 10
    blinks = 0
    blink_start = None
    BLINK_THRESHOLD = 3 
    BLINK_DURATION = 3
    YAWN_THRESHOLD = 6
    yawn_count = 0
    last_yawn_alert = 0  
    YAWN_ALERT_DELAY = 5

    if len(ear_history) > EAR_WINDOW:
        ear_history = ear_history[-EAR_WINDOW:]

    head_pose_timer = None
    ALERT_DELAY = 5 #seconds
    ALERT_DELAY1 = 8 #seconds

    while cap.isOpened():
        try:
            success, image = cap.read()
            if not success:
                break

            start = time.time()

            # Flip the image horizontally and convert to RGB
            image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image_rgb.flags.writeable = False
            
            # Get face mesh results
            results = face_mesh.process(image_rgb)

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image_bgr.shape
            face_3d = []
            face_2d = []

            # Drowsiness Detection
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            count += 1

            if face_locations:
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

                for face_landmarks in face_landmarks_list:
                    left_eye = np.array(face_landmarks['left_eye'])
                    right_eye = np.array(face_landmarks['right_eye'])
                    mouth = np.array(face_landmarks['bottom_lip'])
                    print(f"Bottom Lip Landmarks: {face_landmarks['bottom_lip']}")

                    # Calculate EAR and MAR
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    ear_history.append(ear)
                    if len(ear_history) > EAR_WINDOW:
                         ear_history.pop(0)
                    
                    adaptive_threshold = max(0.2, np.mean(ear_history) * 0.8)
                    eye_flag = ear < adaptive_threshold

                    #mar = mouth_aspect_ratio(mouth)

                    # Check if eyes are closed
                    #eye_flag = ear < EYE_AR_THRESH
                    # Check if mouth is open
                    #mouth_flag = mar > MOUTH_AR_THRESH



                    # Update score more responsively
                    if eye_flag :
                        score += 1
                    #elif mouth_flag:
                        #score +=1  # Increase score faster
                    else:
                        score -= 1  # Decrease score more slowly
                    if score < 0:
                        score = 0

                    if eye_flag:
                        if blink_start is None:
                            blink_start = time.time()
                    else:
                        if blink_start and time.time() - blink_start > BLINK_DURATION / 30:
                            blinks += 1
                        blink_start = None
                    cv2.putText(image_bgr, f"Blinks/min: {blinks}", (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    mar = mouth_aspect_ratio(mouth)
                    print(f"MAR: {mar}")
                    mouth_flag = mar > MOUTH_AR_THRESH

                '''if -10 < y < 10:
                        if mouth_flag:
                            yawn_count += 1
                        #print(f"Yawning detected! Count: {yawn_count}")
                        else:
                            yawn_count -= 1
                    if yawn_count < 0:
                        yawn_count = 0
                    current_time = time.time() '''
                    
            

               
            y = 0  
            x = 0
            z = 0
            # Check Head Pose
            head_pose_alert = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    # See where the user's head is tilting
                    if y < -10:
                        text = "Looking Left"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY1:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "Distracted", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif y > 10:
                        text = "Looking Right"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY1:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "Distracted", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif x < -10:
                        text = "Looking Down"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "focus on the steering", (image_bgr.shape[1] - 400, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif x > 10:
                        text = "Looking Up"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "focus on the steering", (image_bgr.shape[1] - 400, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        text = "Forward"
                        head_pose_timer = None

                    if -10 < y < 10:
                        if mouth_flag:
                             yawn_count += 1
                        else:
                             yawn_count -= 1
                    if yawn_count < 0:
                         yawn_count = 0
                    

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                    cv2.line(image_bgr, p1, p2, (255, 0, 0), 3)
                    cv2.putText(image_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Display drowsiness score and warning
            cv2.putText(image_bgr, f"Score: {score}", (10, image_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

            if score >= 6:
                cv2.putText(image_bgr, "DROWSY!", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                head_pose_alert = True

            (text_width, text_height), _ = cv2.getTextSize(f"Score: {score}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.putText(image_bgr, f"Yawn: {yawn_count}", (10, image_bgr.shape[0] - 40),  # Move up
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if yawn_count >= 6 :
                cv2.putText(image_bgr, "Continuous Yawning", (image_bgr.shape[1] - 300, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                head_pose_alert = True



            # Check for cell phone detection
            results = model(image)
            names = model.names
            for r in results:
                for c in r.boxes.cls:
                    if(names[int(c)] == "cell phone"):
                        print("Cell phone detected!")
                        head_pose_alert = True
                        cv2.putText(image_bgr, "Mobile Phone  detected", (image_bgr.shape[1] - 400, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
           # Inside your combined_detection function

            if head_pose_alert:
                today_date = time.strftime("%Y-%m-%d")  # Get today's date in YYYY-MM-DD format

                # Check if user exists
                user = users_collection.find_one({"email": email})
                print(f"Today Date: {today_date}, Email: {email}, User: {user}")
                
                if user:  # Ensure user exists
                    # If today_date is already present in the user's data
                    if today_date in user["day"]:
                        index = user["day"].index(today_date)
                        # Update the count at the specific index of both arrays
                        users_collection.update_one(
                            {"email": email},
                            {"$inc": {f"count.{index}": 1}}  # Increment count at the specific index
                        )
                    else:
                        users_collection.update_one(
                            {"email": email},
                            {"$push": {"day": today_date}, "$addToSet": {"count": 0}}
                        )


                   

            if head_pose_alert:
                socketio.emit('alert', {'message': 'Drowsiness detected!'}, namespace='/alert')
                alert_sound.play()
                
                

            ret, buffer = cv2.imencode('.jpg', image_bgr)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            break


@app.route('/')
def home():
    return render_template('home.html')  # Home page with login and signup buttons
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone_number = request.form['phone_number']
        license_number = request.form['license_number']
        vehicle_number = request.form['vehicle_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        today_date = time.strftime("%Y-%m-%d")
        
        if not re.fullmatch(r'\d{10}', phone_number):
           flash("Phone number must be exactly 10 digits.", "danger")
           return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return redirect(url_for('signup'))
        
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            flash("Email already registered. Please login.", "danger")
            return redirect(url_for('login'))
        
        hashed_password = hash_password(password)
        user = {
            "name": name,
            "email": email,
            "phone_number": phone_number,
            "license_number": license_number,
            "vehicle_number": vehicle_number,
            "password": hashed_password,
            "day": [today_date],
            "count": [0]
        }
        users_collection.insert_one(user)
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))
    
    return render_template('signup.html')
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8') if isinstance(hashed, str) else hashed)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'email' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = users_collection.find_one({"email": email})
        
        if user and check_password(password, user['password']):
            session['email'] = email  # Store email in session
            
            # Generate and send OTP
            otp = generate_otp()
            session["otp"] = otp  # Store OTP in session
            if send_otp_email(email, otp):
                flash("OTP sent to your email. Please check.", "success")
                return redirect(url_for('otp_verification'))
            else:
                flash("Failed to send OTP. Try again.", "danger")
                return redirect(url_for('login'))
        else:
            flash("Invalid email or password", "danger")
    
    return render_template('login.html')


def generate_otp():
    otp = random.randint(1000, 9999)
    return otp
def send_otp_email(email, otp):
    sender_email = "nikhathmahammad12@gmail.com"
    sender_password = "yhos ywhn elzt rucx"
    receiver_email = email
    subject = "Your OTP Code"
    body = f"Your OTP code is: {otp}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    return True

@app.route('/otp_verification', methods=['POST', 'GET'])
def otp_verification():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        actual_otp = session.get("otp")  # Get stored OTP from session

        if actual_otp and entered_otp == str(actual_otp):  # Compare as string
            session.pop("otp")  # Remove OTP from session after verification
            return redirect(url_for('index'))
        else:
            flash("Invalid OTP, try again.", "danger")

    return render_template('otp_verification.html')

@app.route('/index')  # Index page route
def index():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    return render_template('index.html')  # This will render the index page
 # This will render the dashboard page

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

def send_alert():
    socketio.emit('alert', {'message': '🚨 Continuous Yawning Detected!'}, namespace='/')



@app.route('/profile')
def profile():
    if 'email' not in session:  
        return redirect(url_for('login'))
    email = session.get('email')  # Get email from session
    user = users_collection.find_one({"email": email})
    if user:
        return render_template('profile.html', user=user)  # Pass user data to the template
      # Redirect to login if email is not provided

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    email = session['email']  # Get email from session
    user = users_collection.find_one({"email": email})  # Fetch user data from the database

    if request.method == 'POST':
        # Update user details
        updated_name = request.form['name']
        updated_phone = request.form['phone_number']
        updated_license = request.form['license_number']
        updated_vehicle = request.form['vehicle_number']

        users_collection.update_one(
            {"email": email},
            {"$set": {
                "name": updated_name,
                "phone_number": updated_phone,
                "license_number": updated_license,
                "vehicle_number": updated_vehicle
            }}
        )

        # Redirect back to profile with updated details
        return redirect(url_for('profile'))

    # Render the edit profile page
    return render_template('edit_profile.html', user=user)

@app.route('/video')
def video():
    email = session.get('email')  # Get email from session
    return Response(combined_detection(email), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')  
def logout():
    session.pop('email', None)  
    return redirect(url_for('index'))

if __name__ == "__main__":
    socketio.run(app, debug=True)