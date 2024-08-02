import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Prepare data
input_dir = r'C:\Users\debad\OneDrive\Desktop\image_data'
categories = ['empty', 'non_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    if not os.path.exists(category_path):
        print(f"Directory {category_path} does not exist.")
        continue
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file)
        try:
            img = imread(img_path)
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_idx)
        except Exception as e:
            print(f"Error reading file {img_path}: {e}")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Check the number of samples and adjust parameters
n_samples = len(labels)
print("Class distribution:", dict(zip(*np.unique(labels, return_counts=True))))
print(f"Total samples: {n_samples}")

# Use Leave-One-Out Cross-Validation (LOOCV) if dataset is very small
loo = LeaveOneOut()

# Create and fit the model
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001], 'C': [1, 10]}]  # Simplified parameter grid
grid_search = GridSearchCV(classifier, parameters, cv=loo, error_score='raise')

# Fit the grid search
try:
    grid_search.fit(data, labels)
except ValueError as e:
    print(f"Error during GridSearchCV: {e}")

# Test performance (manual loop for LOOCV)
best_estimator = grid_search.best_estimator_
y_pred = []
y_true = []

for train_index, test_index in loo.split(data):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    best_estimator.fit(x_train, y_train)
    y_pred.extend(best_estimator.predict(x_test))
    y_true.extend(y_test)

score = accuracy_score(y_true, y_pred)
print(f'{score * 100}% of samples were correctly classified')

# Save the model
pickle.dump(best_estimator, open('./model.p', 'wb'))
