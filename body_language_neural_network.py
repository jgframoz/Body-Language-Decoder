import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
import tensorflow as tf
import numpy as np

model_save_path = 'keypoint_classifier.hdf5'
tflite_save_path = 'keypoint_classifier.tflite'

df = pd.read_csv('body_language.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

NUM_CLASSES = len(y.unique())
print(NUM_CLASSES)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((501 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

print(model.summary())


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)

# Transform model (quantization)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)