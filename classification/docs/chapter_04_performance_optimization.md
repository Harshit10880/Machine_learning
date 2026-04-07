# Chapter 4: Performance Analysis and Optimization

## 4.1 Model Performance Comparison

### Accuracy Results (Typical Ranges)
```
Iris Dataset:
- KNN (k=3): 95-100%
- KNN (k=5): 93-98%
- SVM (linear): 96-100%

Wine Dataset:
- KNN (k=3): 70-80%
- KNN (k=5): 68-78%
- SVM (linear): 95-98%
- SVM (rbf): 98-100%

Breast Cancer:
- KNN (k=3): 94-97%
- SVM (linear): 95-98%
- SVM (rbf): 96-99%

Digits:
- KNN (k=3): 95-98%
- SVM (linear): 94-97%
- SVM (rbf): 98-99%
```

## 4.2 Hyperparameter Tuning

### KNN Hyperparameters
```python
# Grid search for optimal k
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best k: {grid_search.best_params_['n_neighbors']}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### SVM Hyperparameters
```python
# SVM parameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## 4.3 Cross-Validation Techniques

### K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Stratified K-Fold (for imbalanced datasets)
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

---

**[← Back to Main Guide](../classification_guide.md)** | **[Next: Chapter 5 →](chapter_05_troubleshooting.md)**