
import Main
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


label_map = {label: num for num, label in enumerate(Main.actions)}

sequences, labels = [], []
for action in Main.actions:
    for sequence in range(Main.no_sequences):
        window = []
        for frame_num in range(Main.sequence_length):
            res = np.load(os.path.join(Main.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)

x = np.array(sequences)
y = to_categorical(labels).astype(int)
# for one hot encoded data => print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
print(Main.actions.shape[0])
res = [.7, .2, .1]
print(Main.actions[np.argmax(res)])


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
# 3 layers of Long short memory we have added to our model
# 64 units , 128 units and 64
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(Main.actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])




