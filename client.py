import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

# from tfexample.task import load_data, load_model

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
df_batadal = pd.read_csv("Batadal Dataset\BATADAL_dataset04.csv")

df_batadal.drop(columns=['DATETIME'], inplace=True)
df_batadal.drop(columns=[' ATT_FLAG'], inplace=True)
df_batadal['ATT_Label'] = df_batadal['ATT_Label'].fillna(9999)

mapping = {
    'Attack 1': 1,
    'Attack 2': 2,
    'Attack 3': 3,
    'Attack 4': 4,
    'Attack 5': 5,
    'Attack 6': 6,
    'Attack 7': 7
}

df_batadal['ATT_Label'] = df_batadal['ATT_Label'].map(mapping)
df_batadal['ATT_Label'] = df_batadal['ATT_Label'].fillna(9999)
# Shuffle the data for better accuracy
df_batadal = df_batadal.sample(frac=1)

X = df_batadal.drop(columns=['ATT_Label'])
y = df_batadal['ATT_Label']

# Encode the target variable if it's categorical
le = LabelEncoder()
y = le.fit_transform(y)

# One-hot encode the target variable for multi-class classification
y = to_categorical(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build an ANN model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))  # Optional: helps to reduce overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Optional: helps to reduce overfitting
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Define Flower Client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
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
        return model.get_weights(), len(X_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy":accuracy}
    
fl.client.start_client(server_address="217.160.150.12:8080", client=FlowerClient().to_client(),)
#Close wandb client
wandb.finish()



# # Make predictions on the test set
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_test_classes = np.argmax(y_test, axis=1)

# # Evaluate the model
# print("Classification Report:")
# print(classification_report(y_test_classes, y_pred_classes))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test_classes, y_pred_classes))
