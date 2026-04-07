# Chapter 7: Practical Applications and Case Studies

## 7.1 Real-World Applications

### Healthcare
- **Breast Cancer Detection**: Binary classification for malignancy prediction
- **Disease Diagnosis**: Multi-class classification based on symptoms
- **Medical Image Analysis**: Classifying X-rays, MRIs

### Finance
- **Credit Scoring**: Binary classification for loan approval
- **Fraud Detection**: Anomaly detection in transactions
- **Stock Market Prediction**: Trend classification

### Marketing
- **Customer Segmentation**: Clustering similar customers
- **Churn Prediction**: Binary classification for customer retention
- **Recommendation Systems**: Product categorization

## 7.2 Industry Case Study: Email Spam Classification

```python
# Example spam classification pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create text classification pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Sample data
emails = ["Win a free iPhone!", "Meeting at 3 PM", "Buy cheap watches"]
labels = [1, 0, 1]  # 1=spam, 0=ham

text_pipeline.fit(emails, labels)
predictions = text_pipeline.predict(["Congratulations! You won!", "Project update"])
```

---

**[← Back to Main Guide](../classification_guide.md)** | **[Next: Chapter 8 →](chapter_08_resources.md)**