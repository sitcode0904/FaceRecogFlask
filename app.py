import os
import sqlite3
import cv2
import numpy as np
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from PIL import Image

basedir = os.path.abspath(os.path.dirname(__file__))

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
faceDetect = cv2.CascadeClassifier(face_cascade_path)

if faceDetect.empty():
    print("Error loading Haarcascade XML. Check file path!")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class database_model(db.Model):
    StudentId = db.Column(db.Integer, primary_key=True)
    StudentName = db.Column(db.String(200), nullable=False)
    StudentAge = db.Column(db.String(200), nullable=False)

    def __init__(self, StudentId, StudentName, StudentAge):
        self.StudentId = StudentId
        self.StudentName = StudentName
        self.StudentAge = StudentAge

@app.route("/")
def hello_world():
    return render_template("register.html")  

@app.route("/register", methods=['POST', 'GET'])
def submit():
    if request.method == "POST":
        StudentId = request.form['studentid']
        StudentName = request.form['studentname']
        StudentAge = request.form['studentage']

        insertorupdate(StudentId, StudentName, StudentAge)

        sampleNum = 0
        cam = cv2.VideoCapture(0)  # Initialize camera inside function
        if not cam.isOpened():
            return "Error: Camera not detected!"

        while sampleNum < 20:
            ret, img = cam.read()
            if not ret:
                return "Error capturing image from camera!"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite(f"dataset/user.{StudentId}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.waitKey(100)

            cv2.imshow("Face", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        return redirect("/")
    
    return render_template("register.html")

def insertorupdate(Id, Name, age):
    existing_student = database_model.query.filter_by(StudentId=Id).first()
    if existing_student:
        existing_student.StudentName = Name
        existing_student.StudentAge = age
    else:
        new_student = database_model(StudentId=Id, StudentName=Name, StudentAge=age)
        db.session.add(new_student)
    
    db.session.commit()

@app.route("/detect", methods=['POST', 'GET'])
def detection():
    if request.method == "POST":
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = os.path.join(basedir, 'dataset')

        ids, faces = getimgwithid(path)
        if len(faces) == 0 or len(ids) == 0:
            return render_template("detect.html", error="No training data found! Register a student first.")

        recognizer.train(faces, ids)
        recognizer.save('recognizer/trainingdata.yml')

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return render_template("detect.html", error="Camera not detected!")

        while True:
            ret, img = cam.read()
            if not ret:
                break  

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 50:
                    profile = database_model.query.filter_by(StudentId=id).first()  # SQLAlchemy query
                    if profile:
                        cv2.putText(img, f"Name: {profile.StudentName}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                        cv2.putText(img, f"Age: {profile.StudentAge}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                    else:
                        cv2.putText(img, "No Data Exists", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Face Recognition', img)

            if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release() 
        cv2.destroyAllWindows() 

        return redirect("/")

    return render_template("detect.html")

def getimgwithid(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    if not images_path:
        return [], []

    faces = []
    ids = []
    for img_path in images_path:
        faceImg = Image.open(img_path).convert('L')
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(img_path)[-1].split(".")[1])
        faces.append(faceNp)
        ids.append(id)
        
    return np.array(ids), faces

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 10000))  # Default Render port
    app.run(host="0.0.0.0", port=port)





