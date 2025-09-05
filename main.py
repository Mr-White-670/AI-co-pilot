import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from transformers import pipeline
from flask import Flask, render_template, Response, jsonify
import threading
import time
import pyttsx3  # For text-to-speech alerts

app = Flask(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ----- Computer Vision Modules -----
class LaneDetector:
    def __init__(self):
        self.prev_lines = None
        
    def detect(self, frame):
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, 5, 150)
        
        # Region of interest (trapezoid shape)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        roi = np.array([[(0, height), (width//2 - 50, height//2 + 50), 
                         (width//2 + 50, height//2 + 50), (width, height)]], np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, minLineLength=40, maxLineGap=150)
        
        # Lane visualization and departure warning
        line_img = np.zeros_like(frame)
        warning = False
        
        if lines is not None:
            # Filter and average lines
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-5)
                if abs(slope) > 0.5:  # Filter horizontal lines
                    if slope < 0: left_lines.append(line[0])
                    else: right_lines.append(line[0])
            
            # Calculate lane center
            if left_lines and right_lines:
                left_avg = np.mean(left_lines, axis=0)
                right_avg = np.mean(right_lines, axis=0)
                lane_center = (left_avg[0] + right_avg[0]) / 2
                vehicle_center = width / 2
                
                # Lane departure check
                if abs(vehicle_center - lane_center) > 100:
                    warning = True
                    cv2.putText(line_img, "LANE DEPARTURE WARNING!", (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                
                # Draw lanes
                cv2.line(line_img, (int(left_avg[0]), int(left_avg[1])), 
                         (int(left_avg[2]), int(left_avg[3])), (0,255,0), 8)
                cv2.line(line_img, (int(right_avg[0]), int(right_avg[1])), 
                         (int(right_avg[2]), int(right_avg[3])), (0,255,0), 8)
        
        return cv2.addWeighted(frame, 0.8, line_img, 1, 0), warning

class ObjectDetector:
    def __init__(self):
        # Load COCO-trained SSD model
        self.model = tf.saved_model.load('ssd_mobilenet_v2_coco/saved_model')
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                        5: 'bus', 7: 'truck', 9: 'traffic light'}
        self.risk_objects = ['car', 'truck', 'bus', 'motorcycle', 'person']
        
    def detect(self, frame):
        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        input_tensor = tf.convert_to_tensor([rgb_frame])

        # Run detection
        detections = self.model(input_tensor)
        
        # Process results
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        
        # Visualization and collision warning
        warning = False
        height, width, _ = frame.shape
        
        for i in range(len(scores)):
            if scores[i] > 0.5:
                class_id = classes[i]
                if class_id in self.classes:
                    label = self.classes[class_id]
                    
                    # Collision risk check
                    if label in self.risk_objects:
                        ymin, xmin, ymax, xmax = boxes[i]
                        box_area = (xmax - xmin) * (ymax - ymin)
                        
                        if box_area > 0.3:  # Object is close
                            warning = True
                            cv2.putText(frame, "COLLISION RISK!", (50, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, 
                                (int(xmin*width), int(ymin*height)),
                                (int(xmax*width), int(ymax*height)),
                                (255,0,0), 2)
                    cv2.putText(frame, f"{label}: {scores[i]:.2f}", 
                                (int(xmin*width), int(ymin*height)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        return frame, warning

class DriverMonitor:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.eye_closed_frames = 0
        
    def detect(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Fatigue detection logic
        warning = False
        if len(eyes) < 2:  # Eyes not detected
            self.eye_closed_frames += 1
            if self.eye_closed_frames > 15:  # ~1 second at 15 FPS
                warning = True
                cv2.putText(frame, "FATIGUE WARNING!", (50, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        else:
            self.eye_closed_frames = 0
            
        # Draw eye markers
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        return frame, warning

# ----- Voice Assistant Module -----
class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.nlp = pipeline("text-classification", model="distilbert-base-uncased")
        self.commands = {
            "navigation": ["navigate to", "directions to", "take me to"],
            "vehicle": ["check battery", "fuel level", "tire pressure"],
            "media": ["play music", "next song", "volume up"],
            "climate": ["set temperature", "turn on ac", "fan speed"],
            "emergency": ["help", "emergency", "call assistance"]
        }
        
    def listen(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening for command...")
            try:
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print(f"Command: {text}")
                return text
            except:
                return ""
    
    def process_command(self, text):
        if not text:
            return "No command detected"
        
        # Intent classification
        result = self.nlp(text)[0]
        intent = result['label']
        confidence = result['score']
        
        # Command-specific responses
        if confidence > 0.85:
            if intent == "navigation":
                return "Calculating route to destination"
            elif intent == "vehicle":
                return "Battery: 85%, Range: 320 km, Tire Pressure: 35 PSI"
            elif intent == "media":
                return "Playing your favorite driving playlist"
            elif intent == "climate":
                return "Setting cabin temperature to 22°C"
            elif intent == "emergency":
                return "EMERGENCY ALERT ACTIVATED! Contacting roadside assistance"
        
        return "Sorry, I didn't understand that command"

# ----- Initialize Modules -----
lane_detector = LaneDetector()
object_detector = ObjectDetector()
driver_monitor = DriverMonitor()
voice_assistant = VoiceAssistant()

# ----- Video Processing Thread -----
def video_processing():
    cap = cv2.VideoCapture('test_drive.mp4')  # Use 0 for webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frames through all modules
        frame, lane_warning = lane_detector.detect(frame)
        frame, object_warning = object_detector.detect(frame)
        frame, fatigue_warning = driver_monitor.detect(frame)
        
        # Combine the warnings
        warnings = []
        if lane_warning: warnings.append("LANE")
        if object_warning: warnings.append("COLLISION")
        if fatigue_warning: warnings.append("FATIGUE")
        
        # Generate alert if any warnings
        if warnings:
            alert = "ALERT: " + ", ".join(warnings)
            cv2.putText(frame, alert, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            
            # Voice alert (run in separate thread)
            threading.Thread(target=engine.say, args=(alert,)).start()
            engine.runAndWait()
        
        # Store processed frame for dashboard
        global processed_frame
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame = buffer.tobytes()

# ----- Flask Dashboard -----
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if processed_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/voice_command', methods=['POST'])
def handle_voice_command():
    command = voice_assistant.listen()
    response = voice_assistant.process_command(command)
    return jsonify({"command": command, "response": response})

@app.route('/system_status')
def system_status():
    return jsonify({
        "modules": ["Lane Detection: Active", "Object Detection: Active", 
                    "Driver Monitor: Active", "Voice Assistant: Ready"],
        "alerts": ["No critical alerts"]
    })

if __name__ == '__main__':
    # For starting video processing thread
    threading.Thread(target=video_processing, daemon=True).start()

    # For starting Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)


processed_frame = None
