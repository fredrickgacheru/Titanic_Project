#importing libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Importing data sets 
dataset = pd.read_csv('train.csv')
useful_data = dataset[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]]

#Drop rows with missing values
useful_data = useful_data.dropna()

# Splitting features (X) and target variable (y)
X = useful_data.iloc[:, 1:]  # Select all columns except the first one (Survived)
y = useful_data.iloc[:, 0]  # Select the first column (Survived)

# Dealing with Categorical Data
X = pd.get_dummies(X, columns=['Sex'], drop_first=True).astype(int)

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###Modelling
# SVM
# #Fitting logistic regression to the training set 
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state=0, C = 1, gamma = 0.4, degree = 2)
# classifier.fit(X_train, y_train)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

##Predicting the Results 
y_pred = classifier.predict(X_test)

# Part 2 - Building the ANN
# import keras
# from keras.models import Sequential
# from keras.layers import Dense 

# #create the classifier
# classifier = Sequential()

# # Adding the input layer and the first hidden layer
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=5))

# # Adding the second hidden layer
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# # Adding the output layer
# classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# # Part 3 - Training the ANN

# # Compiling the ANN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# # Training the ANN on the Training set
# classifier.fit(X_train, y_train, batch_size = 32, epochs = 350)                   

# ##Predicting the Results 
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5).astype(int)

# #3 Random Forest
# # Training the Random Forest Classification model on the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# ## Applying k-fold cross validation 
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10, n_jobs=-1)
# print("accuracies mean", accuracies.mean())
# print("accuracies std", accuracies.std())


#using the model to work with the text data
test_dataset = pd.read_csv('test.csv')
test_id= test_dataset[['PassengerId']]
test_data = test_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch"]]

#fill missing values with mean
test_data['Age'].fillna(test_data['Age'].mean().round(1), inplace=True)

#handle categorical data in test data
test_data = pd.get_dummies(test_data, columns=['Sex'], drop_first=True).astype(int)

#feature scaling and prediction
test_data = sc.transform(test_data)
test_pred = classifier.predict(test_data)


# Convert test_pred to a DataFrame with the column name 'Survived'
test_pred_df = pd.DataFrame({'Survived': test_pred})

# Combine test_id and test_pred DataFrames
result = pd.concat([test_id, test_pred_df], axis=1)

# Save the combined DataFrame to a CSV file
result.to_csv('combined_results_dtr.csv', index=False)

