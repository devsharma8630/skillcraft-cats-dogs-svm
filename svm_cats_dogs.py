import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Dataset path
DATA_PATH = "dataset/train"
IMG_SIZE = 64

X = []
Y = []

print("Loading Images...")

for category in ["cat", "dog"]:
    folder = os.path.join(DATA_PATH, category)
    label = 0 if category == "cat" else 1

    for img in tqdm(os.listdir(folder)[:2000]):
        try:
            img_path = os.path.join(folder, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # HOG Feature Extraction
            features = hog(image, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
            X.append(features)
            Y.append(label)
        except:
            pass

X = np.array(X)
Y = np.array(Y)

print("Total Images:", len(X))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Model (Powerful RBF Kernel)
print("Training SVM Model...")
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save Model + Scaler
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model & Scaler Saved Successfully")
