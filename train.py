import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

dataset = pd.read_csv('resultado_atlo.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

if y.dtype == 'O' or not np.issubdtype(y.dtype, np.integer):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("Formato dos dados:")
print("Treino:", x_train.shape, y_train.shape)
print("Validação:", x_val.shape, y_val.shape)

print("Classes em treino:", np.unique(y_train))
print("Classes em validação:", np.unique(y_val))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((x_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])



classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))  # mapeia classe → peso
print("Pesos de classe:", class_weight_dict)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=40,
    verbose=1,
    mode='min',
    min_delta=0.001,
    cooldown=0,
    min_lr=0
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    min_delta=0.001,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=128,
    epochs=200,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_on_plateau]
)


model.save('dnn_model_atlo.h5')
model.evaluate(x_val, y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
