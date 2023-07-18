import cv2
import mediapipe as mp
import numpy as np
import time, os
from PIL import ImageFont, ImageDraw, Image
import sys

# ⭐ : 변동성 있는 코드
# 🌟 : 데이터셋 추가 정보 저장을 위해 추가한 코드

# 한글
actions = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅚ','ㅟ','ㅢ','space','backspace','clear','sum']

# 영어
# actions = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','space','backspace','clear' ]


seq_length = 30
secs_for_action = 15 # 데이터 저장 시간 # ⭐
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('D:/python/sign_language_project_vscode/dataset', exist_ok=True) # dataset 폴더 준비 # ⭐

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        img = cv2.flip(img, 1) # 좌우 반전
        img = Image.fromarray(img) #img배열을 PIL이 처리가능하게 변환

        draw = ImageDraw.Draw(img)
        myfont = ImageFont.truetype('D:/python/sign_language_project_vscode/font/MaruBuri-Bold.ttf',20) # 폰트 경로 # ⭐
        org=(10,30)# 글자 표시할 위치

        draw.text(org, f'Waiting for collecting {action.upper()} action...', font=myfont, fill=(0,0,0)) # 추측으론 fill은 글자색 0,0,0은 검은색
        img = np.array(img) # numpy가 처리할 수 있도록 다시 변환
        cv2.imshow('img', img)
        cv2.waitKey(2000) # 준비시간 # ⭐

        start_time = time.time()

        if img is None:
            print('Image load failed')
            sys.exit()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            
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
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
                    v = v2 - v1

                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    point_y = joint[5][1] - joint[0][1] # 🌟
            
                    if(point_y >=0): # 🌟
                        point_y = 1 # 🌟
                    else: # 🌟
                        point_y = 0 # 🌟
            
                    thumb4 = joint[4][0] - joint[5][0] # 🌟
                    thumb4_y = joint[4][1] - joint[5][1] # 🌟
            
                    if(thumb4 >=0): # 🌟
                        thumb4= 1 # 🌟
                    else: # 🌟
                        thumb4= 0 # 🌟
                        
                    if(thumb4_y >=0): # 🌟
                        thumb4_y= 1 # 🌟
                    else: # 🌟
                        thumb4_y= 0 # 🌟
            
                    second8 = joint[8][:2] - joint[17][:2] # 🌟
                    second8 = second8.flatten() # 🌟
                    second8 = second8 / np.linalg.norm(second8) # 🌟

                    d = np.append(v.flatten(), check) # 🌟
                    d = np.append(d, point) # 🌟
                    d = np.append(d,zero_to_tip) # 🌟
                    d = np.append(d,tip_to_tip) # 🌟

                    d = np.append(d, point_y) # 🌟
                    d = np.append(d,thumb4) # 🌟
                    d = np.append(d,thumb4_y) # 🌟
                    d = np.append(d,second8) # 🌟

                    d = np.concatenate([d, angle_label]) # 🌟

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('D:/python/sign_language_project_vscode/dataset', f'raw_{action}_{created_time}'), data) # ⭐

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('D:/python/sign_language_project_vscode/dataset', f'seq_{action}_{created_time}'), full_seq_data) # ⭐
    break