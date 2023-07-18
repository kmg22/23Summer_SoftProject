import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image, ImageTk

# 영어
actions = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','space','backspace','clear']
seq_length = 30
model = load_model('models/model2_1.0.h5')

cap = cv2.VideoCapture(0)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

seq = []
action_seq = []
word = ""

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:

            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            check = joint[5][0] - joint[17][0] # 🌟
            point = joint[5][0] - joint[0][0] # 🌟
            
            if(check >=0): # 🌟
                check = 1 # 🌟
            else: # 🌟
                check = 0 # 🌟

            comp_tip_1 = joint[[4,8,12,16], 0:2] # 🌟                
            comp_tip_2 = joint[[8,12,16,20], 0:2] # 🌟
                                       
            tip_to_tip = comp_tip_1 - comp_tip_2 # 🌟
            tip_to_tip = tip_to_tip.flatten() # 🌟
            tip_to_tip = tip_to_tip / np.linalg.norm(tip_to_tip) # 🌟
                                        
            zero = joint[[0,0,0,0,0], 1:2 ] # 🌟                
            tip = joint[[4,8,12,16,20], 1:2] # 🌟
                    
            zero_to_tip = zero - tip # 🌟
            zero_to_tip = zero_to_tip / np.linalg.norm(zero_to_tip) # 🌟

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint # ⭐
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint # ⭐
            v = v2 - v1

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            point_y = joint[5][1] - joint[0][1] # 🌟
            
            if(point_y >=0): # 🌟
                point_y = 1
            else: # 🌟
                point_y = 0
            
            thumb4 = joint[4][0] - joint[5][0] # 🌟
            thumb4_y = joint[4][1] - joint[5][1] # 🌟
            
            if(thumb4 >=0): # 🌟
                thumb4= 1
            else: # 🌟
                thumb4= 0
            
            if(thumb4_y >=0): # 🌟
                thumb4_y= 1
            else: # 🌟
                thumb4_y= 0
                
            second8 = joint[8][:2] - joint[17][:2] # 🌟
            second8 = second8.flatten() # 🌟
            second8 = second8 / np.linalg.norm(second8) # 🌟

            d = np.append(v.flatten(), check) # 🌟
            d = np.append(d, point) # 🌟                
            d = np.append(d,zero_to_tip) # 🌟
            d = np.append(d,tip_to_tip) # 🌟
            
            d = np.append(d, point_y ) # 🌟
            d = np.append(d, thumb4) # 🌟
            d = np.append(d, thumb4_y) # 🌟
            d = np.append(d, second8) # 🌟

            d = np.concatenate([d, angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.8: # ⭐
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]: # 정확한 제스처라면
                this_action = action
            
            if cv2.waitKey(1) == ord('a'): # ⭐
                if this_action == 'space':
                    word = word + " "

                elif this_action == 'backspace':
                    word = word[:-1]

                elif this_action == 'clear':
                    word = ""

                else:
                    word = word + this_action
            
            font = ImageFont.truetype('D:/python/sign_language_project_vscode/font/MaruBuri-Bold.ttf',20) # ⭐        
            img = Image.fromarray(img) # 한글을 출력하기 위해 image를 변환(PIL라이브러리 사용)
            draw = ImageDraw.Draw(img)

            draw.text((10, 30), this_action, font=font, fill=(0,0,0)) # ⭐
            draw.text((10, 70), word, font=font, fill=(255,255,255)) # ⭐
            print(this_action)
            print(word)

    img = np.array(img)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break