import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# converter tipo de dado
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

ds = pd.read_csv('base_maior.csv')
for col in ds.columns[:-1]:
    ds[col] = ds[col].apply(convert_to_numeric)

colunas_num = ds.select_dtypes(include=[np.number]).columns
ds[colunas_num] = ds[colunas_num].pct_change() * 100 
ds = ds.dropna().reset_index(drop=True)

#remover outliers extremos?
ds[colunas_num] = ds[colunas_num].clip(-100, 100)

x = ds.iloc[:, :-1]. values
#print(x.shape)
y = ds.iloc[:, -1].values

if y.dtype == 'O' or not np.issubdtype(y.dtype, np.integer):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# normalização z-score
scaler = RobustScaler()
x = scaler.fit_transform(x)

## treino e validação
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("Formato dos dados:")
print("Treino:", x_train.shape, y_train.shape)
print("Validação:", x_val.shape, y_val.shape)

print("Classes em treino:", np.unique(y_train))
print("Classes em validação:", np.unique(y_val))

#dando erro do tipo argumento invalido (objeto) > transformar p float
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

#pesos de classes
classes = np.unique(y_train)
class_weight = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weight))
print("Pesos de classe:", class_weight_dict)

## modelo 1 (dnn) >
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((x_train.shape[1],)),
    # 1 camada (+ neuronios)
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    # 2 camada
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    # 3 camada
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # taxa de aprendizado menor
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=20,
    verbose=1,
    min_delta=0.001,
    cooldown=5,
    min_lr=1e-6
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=40,
    min_delta=0.001,
    restore_best_weights=True
)

#treinar
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=16,
    epochs=200,
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_on_plateau]
)

#model.save('dnn_model_1.h5')
#model.evaluate(x_val, y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Treino')
plt.plot(epochs_range, val_acc, label='Validação')
plt.legend(loc='lower right')
plt.title('Acurácia - Treino vs validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Treino')
plt.plot(epochs_range, val_loss, label='Validação')
plt.legend(loc='upper right')
plt.title('Loss - Treino vs validação')
plt.show()