import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("emails.csv")

# Input and Output
X = df["text"]
y = df["label"]

# --------------------------
# Convert Text to Features
# --------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# --------------------------
# Split Dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# Train Model
# --------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Make Predictions
# --------------------------
y_pred = model.predict(X_test)

# --------------------------
# Accuracy
# --------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# --------------------------
# Classification Report
# --------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# --------------------------
# Plot Confusion Matrix
# --------------------------
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing', 'Safe'],
    yticklabels=['Phishing', 'Safe']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# --------------------------
# Test Custom Emails
# --------------------------
while True:

    email = input("\nEnter an email message (or type 'exit'): ")

    if email.lower() == "exit":
        break

    email_vector = vectorizer.transform([email])

    prediction = model.predict(email_vector)[0]

    if prediction == "phishing":
        print("⚠️ Warning: This email is PHISHING")
    else:
        print("✅ This email is SAFE")