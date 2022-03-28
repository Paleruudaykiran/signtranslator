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
                    ch = '_'
                    lis = hand_landmarks.landmark
                    #print('list ',lis)
                    row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in lis]).flatten())
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class,round(body_language_prob[np.argmax(body_language_prob)],2)*100)
                    if round(body_language_prob[np.argmax(body_language_prob)],2)*100 >= 70 :
                        ch = body_language_class.split(' ')[0]
                        #print(ch)
                        if idx == 0 : 
                            sentence.append(ch) 
                            idx += 1
                        else : 
                            if sentence[-1] != ch : 
                                sentence.append(ch) 
                        #print(sentence)
                except : 
                    pass
    #print('lis : ',sentence)
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
        if not results.multi_hand_landmarks : 
            return '_'
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
    

    
