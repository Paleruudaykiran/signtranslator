from flask import Flask,render_template,request,flash,redirect,url_for,Response
import urllib.request
import os
from werkzeug.utils import secure_filename
from werkzeug.wrappers import response
import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def videoPrediction(path) : 
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    cap = cv2.VideoCapture(path) 
    sentence = []
    idx = 0
    while cap.isOpened() : 
        ret,image = cap.read() 
        if ret == False : 
            break
        with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if  results.multi_hand_landmarks : 
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                try:
                    ch = 'not found'
                    lis = hand_landmarks.landmark
                    #print('list ',lis)
                    row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in lis]).flatten())
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class,round(body_language_prob[np.argmax(body_language_prob)],2)*100)
                    #if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70 :
                    ch = body_language_class.split(' ')[0]
                    #print(ch)
                    if idx == 0 : 
                        sentence.append(ch) 
                        idx += 1
                    else : 
                        if sentence[-1] != ch : 
                            sentence.append(ch) 
                    print(sentence)
                except : 
                    pass
    print('lis : ',sentence)
    return ''.join(sentence)

def imagePrediction(path) : 
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print(path)
    image = cv2.imread(path)
    #print('image' , image) 
    #cv2.imshow('uploaded image',image)
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        try:
            ch = 'not found'
            lis = hand_landmarks.landmark
            #print('list ',lis)
            row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in lis]).flatten())
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class,round(body_language_prob[np.argmax(body_language_prob)],2)*100)
            #if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70 :
            ch = body_language_class.split(' ')[0]
            #print(ch)
            return ch
        except : 
            pass
    

app = Flask(__name__) #creating the Flask class object   
UPLOAD_FOLDER = 'static/images/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 

@app.route('/') #decorator drfines the   
def home():  
    mypath = UPLOAD_FOLDER
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    return render_template('home.html')  
  
@app.route('/stot') 
def SignToText() :
    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #         #print('upload_image filename: ' + filename)
    #         flash('Image successfully uploaded and displayed below')
    #         return render_template('index.html', filename=filename
    #     else:
    #         flash('Allowed image types are - png, jpg, jpeg, gif')return redirect(request.url)
    return render_template('stot.html')

@app.route('/tPredict',methods=['POST']) 
def textPredict() : 
    #print(request.files)
    if request.method == 'POST':
        if 'imgfile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['imgfile']
        #print(file)
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed below')
            path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            text = imagePrediction(path) 

            return render_template('tpredict.html', filename=filename,text=text)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    return 'hello'

@app.route('/vPredict',methods=['POST']) 
def videoPredict() : 
    if 'videofile' not in request.files : 
        return render_template('stot.html') 
    file = request.files['videofile'] 
    if file.filename == '' : 
        return render_template('stott.html') 
    else : 
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        #flash('Image successfully uploaded and displayed below')
        path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        text = videoPrediction(path)
        print(text)
        return render_template('vpredict.html',path=path,text=text)





@app.route('/ttos') 
def TextToSign() : 
    return render_template('ttos.html')

@app.route('/display_image/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='images/uploads/' + filename), code=301)

@app.route('/vRecord') 
def videoRecord() : 
    return render_template('vrecord.html')

sentence = [] 

def gen_frames() : 
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    global cap 
    cap = cv2.VideoCapture(0)
    idx = 0
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('image'+str(idx)+'.jpg',image)
            if results.multi_hand_landmarks:
                   for hand_landmarks in results.multi_hand_landmarks:
                       mp_drawing.draw_landmarks(
                             image,
                             hand_landmarks,
                             mp_hands.HAND_CONNECTIONS,
                             mp_drawing_styles.get_default_hand_landmarks_style(),
                             mp_drawing_styles.get_default_hand_connections_style())
            image = cv2.flip(image, 1)
            try:
                lis = hand_landmarks.landmark
                #print(lis)
                row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in lis]).flatten())
                #print(row)
                X = pd.DataFrame([row])
                #print('X ' ,X)
                body_language_class = model.predict(X)[0]
                #print('class : ',body_language_class)
                body_language_prob = model.predict_proba(X)[0]
                #print(body_language_class, body_language_prob)
                #print(round(body_language_prob[np.argmax(body_language_prob)],2))
                if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70 :
                    ch = body_language_class.split(' ')[0]
                    if idx == 0 : 
                        sentence.append(ch)
                        idx += 1
                    else : 
                        if sentence[-1] != ch : 
                            sentence.append(ch)
                        idx += 1
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (hand_landmarks.landmark[0].x, 
                                 hand_landmarks.landmark[0].y))
                        , [640,480]).astype(int))
            

                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
                # Display Class
                cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA,False)
                cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA,False)
            
                 # Display Probability
                cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,False)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,False)
                
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            except : 
                pass

@app.route('/video')
def video() : 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stopAndPredict') 
def stopAndPredict() : 
    # cap.release()
    # cv2.destroyAllWindows() 
    text = ''.join(sentence)
    return render_template('recresult.html',text=text)

if __name__ =='__main__': 
    #port = os.environ.get("PORT",5000) 
    app.run(debug =True)  