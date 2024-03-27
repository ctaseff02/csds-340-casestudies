import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')

X_train, y_train = train_df.iloc[:, :-1].values, train_df['Quality'].values
X_test, y_test = test_df.iloc[:, :-1].values, test_df['Quality'].values

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = SVC(C=100, kernel='rbf', gamma=0.1, random_state=1)
model.fit(X_train_std, y_train)
accuracy = model.score(X_test_std, y_test)

percentage = round((accuracy * 100), 2)
print("Test Accuracy: " + str(percentage) + "%")
