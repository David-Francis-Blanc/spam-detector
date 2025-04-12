import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("spam_data.csv")
X = df['Message']
y = df['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_vect, y_train)

from sklearn.metrics import accuracy_score


# Predict
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

# User input predictions
while True:
    msg = input("Enter a message (or 'q' to quit): ")
    if msg.lower() == 'q':
        break
    vect_msg = vectorizer.transform([msg])
    pred = model.predict(vect_msg)[0]
    print(f"'{msg}' --> {pred}")

