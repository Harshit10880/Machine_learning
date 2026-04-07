# Chapter 3: Dataset Analysis and Statistics

## 3.1 Dataset Comparison Table

| Dataset | Samples | Features | Classes | Type | Best Algorithm |
|---------|---------|----------|---------|------|----------------|
| Iris | 150 | 4 | 3 | Multiclass | KNN (96-100% acc) |
| Wine | 178 | 13 | 3 | Multiclass | SVM (95-98% acc) |
| Breast Cancer | 569 | 30 | 2 | Binary | KNN/SVM (94-97% acc) |
| Digits | 1797 | 64 | 10 | Multiclass | SVM (95-98% acc) |

## 3.2 Dataset Characteristics

### Iris Dataset
- **Features**: Sepal length/width, petal length/width
- **Class Distribution**: 50 samples per class (balanced)
- **Scale**: Features in cm, similar ranges
- **Complexity**: Low-dimensional, linearly separable

### Wine Dataset
- **Features**: 13 chemical properties
- **Class Distribution**: Class 1: 59, Class 2: 71, Class 3: 48 (slightly imbalanced)
- **Scale**: Different units (alcohol %, magnesium ppm, etc.)
- **Complexity**: Higher dimensional, requires scaling

### Breast Cancer Dataset
- **Features**: 30 computed features from images
- **Class Distribution**: Malignant: 212, Benign: 357 (imbalanced)
- **Scale**: Computed features, various ranges
- **Complexity**: High-dimensional, medical application

### Digits Dataset
- **Features**: 64 pixel values (8x8 image)
- **Class Distribution**: ~180 samples per digit (balanced)
- **Scale**: 0-16 grayscale values
- **Complexity**: Image data, high-dimensional

---

**[← Back to Main Guide](../classification_guide.md)** | **[Next: Chapter 4 →](chapter_04_performance_optimization.md)**