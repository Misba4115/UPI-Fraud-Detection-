import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_name = "UPI_Fraud_Dataset_Expanded.csv"
df = pd.read_csv(file_name)

x = df.iloc[:,:8].values
y = df.iloc[:,8].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

x_test.shape
x_train.shape


y_test.shape
y_train.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DecisionTreeClassifier instance
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the decision tree model
decision_tree.fit(x_train, y_train)

# Predict on the test set
y_preddt = decision_tree.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_preddt)


print("Accuracy:", accuracy)


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

CNN_model = tf.keras.models.Sequential()
CNN_model = tf.keras.models.Sequential()
CNN_model.add(tf.keras.layers.Dense(64,input_dim=8,activation ='relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(Dropout(0.2))
CNN_model.add(tf.keras.layers.Dense(128,activation = 'relu'))
CNN_model.add(BatchNormalization())
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(1,activation = 'sigmoid'))
CNN_model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])
CNN_model.fit(x_train, y_train, batch_size=32, epochs = 200, validation_data=(x_test,y_test))

loss,cnn_accuracy = CNN_model.evaluate(x_test,y_test,verbose =0)

print(cnn_accuracy)

y_predict=CNN_model.predict(x_test)
y_predict = (y_predict>0.5).astype(int)


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Your model accuracy data
Predictive_Models = ['Convolutional Neural Networks', 'Decision Tree']
Accuracy = [accuracy_score(y_test, y_predict) * 100, accuracy_score(y_test, y_preddt) * 100]

# Create a bar plot
plt.bar(Predictive_Models, Accuracy, color=['skyblue', 'coral'])

# Add labels and title
plt.xlabel('Predictive Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')

# Show plot
plt.show()

# Print individual accuracy scores
cnn_accuracy = accuracy_score(y_test, y_predict) * 100
dt_accuracy = accuracy_score(y_test, y_preddt) * 100

print('Convolutional Neural Networks Accuracy score:', cnn_accuracy)
print('Decision Tree Accuracy score:', dt_accuracy)

# Determine the model with the highest accuracy
if cnn_accuracy > dt_accuracy:
    best_model = 'Convolutional Neural Networks'
    highest_value = cnn_accuracy
else:
    best_model = 'Decision Tree'
    highest_value = dt_accuracy

print(f"Model predicting with the highest accuracy is: {best_model} with {highest_value:.2f}% accuracy.")