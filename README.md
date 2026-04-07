# Machine Learning Projects

This repository contains a collection of beginner-friendly machine learning projects implemented in Python. These projects demonstrate fundamental concepts in supervised learning, focusing on classification tasks using scikit-learn and other popular libraries.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Projects](#projects)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository serves as a learning resource for machine learning enthusiasts starting their journey. Each project builds upon the previous one, introducing new datasets and occasionally different algorithms while maintaining simplicity and clarity.

## Requirements

- Python 3.7+
- Required libraries:
  - scikit-learn
  - pandas
  - matplotlib
  - seaborn
  - numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshit10880/Machine_learning.git
   cd Machine_learning
   ```

2. Install the required packages:
   ```bash
   pip install scikit-learn pandas matplotlib seaborn numpy
   ```

## Projects

### Classification Folder

1. **classification_1.py** - Iris Dataset Classification
   - Dataset: Iris flowers
   - Algorithm: K-Nearest Neighbors (KNN)
   - Classes: 3 (Setosa, Versicolor, Virginica)
   - Features: 4 (sepal length, sepal width, petal length, petal width)

2. **classification_2.py** - Wine Dataset Classification
   - Dataset: Wine samples
   - Algorithm: K-Nearest Neighbors (KNN)
   - Classes: 3 (wine types based on chemical properties)
   - Features: 13 (alcohol, malic acid, ash, etc.)

3. **classification_3.py** - Breast Cancer Dataset Classification
   - Dataset: Breast cancer diagnosis
   - Algorithm: K-Nearest Neighbors (KNN)
   - Classes: 2 (malignant, benign)
   - Features: 30 (computed from digitized images)

4. **classification_4.py** - Digits Dataset Classification
   - Dataset: Handwritten digits
   - Algorithm: K-Nearest Neighbors (KNN)
   - Classes: 10 (digits 0-9)
   - Features: 64 (8x8 pixel images)

5. **classification_5.py** - Wine Dataset with SVM
   - Dataset: Wine samples
   - Algorithm: Support Vector Machine (SVM)
   - Classes: 3 (wine types)
   - Features: 13

6. **visulize_prob_1.py** - Data Visualization
   - Demonstrates data visualization techniques
   - Uses matplotlib and seaborn for plotting
   - Works with classification datasets

## Usage

Each Python file can be run independently:

```bash
python classification/classification_1.py
```

The scripts will:
- Load the dataset
- Preprocess the data
- Split into training and testing sets
- Train the model
- Make predictions
- Display accuracy and sample predictions

## Learning Outcomes

By working through these projects, you'll learn:
- How to load and explore datasets
- Data preprocessing techniques
- Model training and evaluation
- Different classification algorithms
- Basic data visualization

## Contributing

Contributions are welcome! If you'd like to:
- Add new projects
- Improve existing code
- Fix bugs
- Add documentation

Please fork the repository and create a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This repository is maintained for educational purposes. The code is designed to be simple and easy to understand for beginners in machine learning.