import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 한글
actions = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅚ','ㅟ','ㅢ','space','backspace','clear','sum']

data = np.concatenate([ # ⭐
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㄱ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㄴ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㄷ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㄹ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅁ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅂ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅅ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅇ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅈ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅊ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅋ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅌ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅍ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅎ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅏ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅑ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅓ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅕ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅗ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅛ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅜ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅠ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅡ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅣ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅐ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅒ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅔ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅖ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅚ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_ㅟ_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_space_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_backspace_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_clear_1689653743.npy'),
    np.load('D:/python/sign_language_project_vscode/dataset/seq_sum_1689653743.npy')
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
    LSTM(16, activation='relu', input_shape=x_train.shape[1:3]), # ⭐
    Dense(1024, activation='relu'), # ⭐
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
        ModelCheckpoint('models/model2_1.0_hanguel.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)