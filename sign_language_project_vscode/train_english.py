import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 영어
actions = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','space','backspace','clear','sum','1','2','3']

data = np.concatenate([ # ⭐
    np.load('D:/python/sign_language_project_vscode/dataset/seq_a_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_b_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_c_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_d_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_e_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_f_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_g_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_h_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_i_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_j_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_k_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_l_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_m_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_n_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_o_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_p_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_q_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_r_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_s_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_t_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_u_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_v_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_w_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_x_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_y_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_z_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_space_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_backspace_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_clear_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_sum_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_1_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_2_1689308441.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_3_1689308441.npy')
], axis=0)

data.shape

################################################

x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)

################################################

from tensorflow.keras.utils import to_categorical

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

################################################

from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

################################################

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model2_1.0.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)