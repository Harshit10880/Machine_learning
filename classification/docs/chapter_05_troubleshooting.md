# Chapter 5: Common Mistakes and Troubleshooting

## 5.1 Data-Related Issues

### Problem: Poor Model Performance
**Symptoms**: Low accuracy, inconsistent results
**Possible Causes**:
- Insufficient data preprocessing
- Imbalanced classes
- Feature scaling not applied
- Overfitting to training data

**Solutions**:
```python
# Check class distribution
print(df['target'].value_counts())

# Apply feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use stratified split for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Problem: Inconsistent Results
**Symptoms**: Different accuracy each run
**Cause**: No random_state in train_test_split
**Solution**: Always use `random_state=42` for reproducibility

## 5.2 Algorithm-Specific Issues

### KNN Issues
- **Slow prediction**: Large dataset - consider dimensionality reduction
- **Poor performance**: Features not scaled - apply StandardScaler
- **Overfitting**: k too small - increase k value

### SVM Issues
- **Slow training**: Large dataset - use linear kernel or reduce features
- **Poor performance**: Wrong kernel - try different kernels
- **Overfitting**: C too high - reduce C value

## 5.3 Code Quality Issues

### Problem: Memory Errors
**Cause**: Loading large datasets
**Solution**: Use `partial_fit` for incremental learning or reduce batch size

### Problem: Import Errors
**Common Issues**:
- Wrong sklearn version
- Missing dependencies
- Incorrect import paths

**Solution**:
```bash
# Check versions
pip list | grep scikit-learn

# Update packages
pip install --upgrade scikit-learn pandas numpy matplotlib seaborn
```

---

**[← Back to Main Guide](../classification_guide.md)** | **[Next: Chapter 6 →](chapter_06_advanced_topics.md)**