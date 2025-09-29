import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '.')
        if value.count('.') > 1: 
            value = value.replace('.', '')
        try:
            return float(value)
        except ValueError:
            return value 
    return value

ds = pd.read_csv('base_maior.csv')
for col in ds.columns[:-1]:
    ds[col] = ds[col].apply(convert_to_numeric)

colunas_num = ds.select_dtypes(include=[np.number]).columns
ds[colunas_num] = ds[colunas_num].pct_change()* 100 
ds = ds.dropna().reset_index(drop=True)

x = ds.iloc[:, :-1]. values
print(x.shape)
y = ds.iloc[:, -1].values

if y.dtype == 'O' or not np.issuebdtype(y.dtype, np.integer):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

## treino e validação ##
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("Formato dos dados:")
print("Treino:", x_train.shape, y_train.shape)
print("Validação:", x_val.shape, y_val.shape)

print("Classes em treino:", np.unique(y_train))
print("Classes em validação:", np.unique(y_val))

#dando erro do tipo argumento invalido (objeto) > transformar p float
x_train = x_train.astype(np.float64)
x_val = x_val.astype(np.float64)
y_train = y_train.astype(np.float64)
y_val = y_val.astype(np.float64)

## modelo 1 (dnn) >
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((x_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, activation='softmax')
])

classes = np.unique(y_train)
class_weight = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weight))
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
    batch_size=8,
    epochs=200,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_on_plateau]
)

model.save('dnn_model_1.h5')
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
