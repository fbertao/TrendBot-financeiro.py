import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)


# converter tipo de dado
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

# ler base de dados
ds = pd.read_csv('base_maior.csv')
for col in ds.columns[:-1]:
    ds[col] = ds[col].apply(convert_to_numeric)

colunas_num = ds.select_dtypes(include=[np.number]).columns
ds[colunas_num] = ds[colunas_num].pct_change() * 100 
ds = ds.dropna().reset_index(drop=True)

#remover outliers extremos?
ds[colunas_num] = ds[colunas_num].clip(-100, 100)

x = ds.iloc[:, :-1]. values
y = ds.iloc[:, -1].values

if y.dtype == 'O' or not np.issubdtype(y.dtype, np.integer):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    classes = list(encoder.classes_)

    if "Sim" in classes and classes.index("Sim") !=1:
        y = np.where(y == 1,0,1)
        print("'Sim' = 1 e 'Não' = 0")
    else:
        print("Subiu ja é classe positiva")


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

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

#epochs_range = range(len(acc))

#plt.figure(figsize=(12, 5))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Treino')
#plt.plot(epochs_range, val_acc, label='Validação')
#plt.legend(loc='lower right')
#plt.title('Acurácia - Treino vs validação')

#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Treino')
#plt.plot(epochs_range, val_loss, label='Validação')
#plt.legend(loc='upper right')
#plt.title('Loss - Treino vs validação')
#plt.show()

# --------------- métricas de avaliação do modelo --------------
y_pred_prob = model.predict(x_val)
y_pred = np.argmax(y_pred_prob, axis=1)

# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred)
# accuracy = np.mean(y_val == y_pred)
# auc = roc_auc_score(y_val, y_pred_prob[:, 1])

# metrics_df = pd.DataFrame({
#     'Métrica': ['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC'],
#     'Valor': [accuracy, precision, recall, f1, auc]
# })
# print("\nmétricas")
# print(metrics_df.to_string(index=False, float_format='{:,.4f}'.format))


# print("\nRelatório:")
# print(classification_report(y_val, y_pred, target_names=["Não subiu", "Subiu"]))

#matriz
# cm = confusion_matrix(y_val, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não subiu", "Subiu"])
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Matriz de Confusão - Validação")
# plt.show()

#curva roc
# fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob[:, 1])
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
# plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
# plt.xlabel('Falsos Positivos')
# plt.ylabel('Verdadeiros Positivos')
# plt.title('Curva ROC - Validação')
# plt.legend(loc='lower right')
# plt.show()

# ------------ simular treino ---------
prob_sim = y_pred_prob[:,1]

#testar thresholds
limite = [0.5, 0.6, 0.7, 0.8, 0.9]
resultados = []

for i in limite:
    pred_i = (prob_sim >= i).astype(int)
    acertos = np.sum(pred_i == y_val)
    total = len(y_val)
    acc = acertos / total

    operacoes = np.sum(pred_i)
    ganho = np.sum((pred_i == 1) & (y_val == 1))
    perder = np.sum((pred_i == 1) & (y_val == 0))
    if operacoes > 0:
        taxa_acerto = ganho / operacoes
    else: 
        taxa_acerto = np.nan

    resultados.append({
        "limite": i,
        "Acurácia:": acc,
        "Operações:": operacoes,
        "Taxa de acerto:": taxa_acerto
    })

    df_resultados = pd.DataFrame(resultados)
    print("\n SIMULAÇÃO")
    print(df_resultados.to_string(index=False, float_format='{:,.2%}'.format))