#--------------------------------Importing Libraries-------------------------------------------#
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
#----------------------------------------------------------------------------------------------#
#Reading CSV File
data = pd.read_csv("Alphabets-26.csv")
X=data.drop('label',axis=1)
y=data['label'].str.upper()

#Dividing the Dataset into Training and Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Creating the Model
model = RandomForestClassifier(n_estimators=200, max_depth=20)

#Training the model
model.fit(X_train,y_train)

#Testing the Model
prediction = model.predict(X_test)
print(f"Model Accuracy : {accuracy_score(y_test,prediction) * 100:.2f}")
# Check how many samples you have for each label
print(data['label'].value_counts())

#Saving the Model
with open("Alphabets_Model-26.pkl", 'wb') as f:
    pickle.dump(model,f)
