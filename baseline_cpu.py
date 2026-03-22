import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print('Preparing data...')

df = pd.read_csv('creditcard.csv')
x = df.drop('Class', axis=1)
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

cpu_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

print('Training...\n')

start_time = time.time()
cpu_model.fit(x_train, y_train)
end_time = time.time()
training_time = end_time - start_time

y_pred = cpu_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Training time: {training_time:.4f} seconds')
print(f'Accuracy: {accuracy * 100:.2f}%\n')
