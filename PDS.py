import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


############################################################
#######Performance Test############

import wandb
import random

wandb.login(key="0cb9217480aec563c71c6df7e90c177b64b48447")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="GlobeCom",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "ANN",
    "dataset": "Batadal",
    "epochs": 50,
    }
)


############################################################

# Load and preprocess the dataset
df_pds = pd.read_csv("Power System Intrusion Dataset/archive/Train.csv")
X_pds= df_pds.drop('class', axis=1)
y_pds=df_pds['class']

# Standardize the features
scaler = StandardScaler()
X_pds = scaler.fit_transform(X_pds)

le_pds = LabelEncoder()
y_pds = le_pds.fit_transform(y_pds)

# One-hot encode the target variable for multi-class classification
y_pds = to_categorical(y_pds)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_pds, y_pds, train_size=0.7, test_size=0.3, random_state=100)# Build an ANN model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))  # Optional: helps to reduce overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Optional: helps to reduce overfitting
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
# opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))


# log metrics to wandb
epochs=50
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = history.history['accuracy'][epoch]
    loss = history.history['loss'][epoch]

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()