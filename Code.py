import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Load dataset
heart_df = pd.read_csv('heart.csv')

# Prepare features and target
X = heart_df.drop(columns=['target'])
y = heart_df['target']

# Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Train Decision Tree Classifier with max_depth to control overfitting
dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_clf.fit(X_train, y_train)

# Visualize Decision Tree using matplotlib (no Graphviz)
plt.figure(figsize=(20,10))
plot_tree(dt_clf, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'], rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# 2. Analyze overfitting by varying tree depth
train_acc, test_acc = [], []
for depth in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 11), train_acc, label='Train Accuracy')
plt.plot(range(1, 11), test_acc, label='Test Accuracy')
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting Analysis: Tree Depth vs Accuracy')
plt.legend()
plt.show()

# 3. Train Random Forest and compare accuracy
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

dt_accuracy = dt_clf.score(X_test, y_test)
rf_accuracy = rf_clf.score(X_test, y_test)

print(f'Decision Tree Accuracy: {dt_accuracy:.4f}')
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

# 4. Interpret and plot feature importances from Random Forest
importances = rf_clf.feature_importances_
plt.figure(figsize=(12,6))
plt.barh(X.columns, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# 5. Evaluate both models using cross-validation
dt_cv_scores = cross_val_score(dt_clf, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=5)

print(f'Decision Tree CV Accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})')
print(f'Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})')

