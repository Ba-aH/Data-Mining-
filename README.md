# 🍄 Mushroom Classification Analysis

**Comparing Random Forest vs KNN with/without preprocessing**  
*A Python data mining project demonstrating the impact of preprocessing on classifier performance*

## 🔍 Key Features
- Implements **Random Forest** and **KNN** classifiers
- Tests **raw data** vs **preprocessed data** (cleaning, encoding, scaling)
- Evaluates using **precision, recall, F1-score** metrics
- 80/20 train-test split with **5 repeated experiments**

## 📊 Key Findings: Preprocessing Impact

### Random Forest Performance
| Metric       | Raw Data | Preprocessed Data | Δ (Change) | Interpretation                |
|--------------|----------|-------------------|------------|-------------------------------|
| Error Rate   | 0.49     | 0.43              | -12.2%     | Significant improvement       |
| Precision    | 0.49     | 0.54              | +10.2%     | Better false positive control |
| Recall       | 0.67     | 0.71              | +6.0%      | Improved poison detection     |
| F1-Score     | 0.56     | 0.61              | +8.9%      | Best balanced performance     |

### KNN Performance (k=3)
| Metric       | Raw Data | Preprocessed Data | Δ (Change) | Interpretation                |
|--------------|----------|-------------------|------------|-------------------------------|
| Error Rate   | 0.57     | 0.58              | +1.8%      | Slight degradation            |
| Precision    | 0.44     | 0.42              | -4.5%      | More false positives          |
| Recall       | 0.63     | 0.55              | -12.7%     | Worse poison detection        |
| F1-Score     | 0.51     | 0.47              | -7.8%      | Overall performance drop      |

## 🛠️ Tech Stack
- Python 3
- Scikit-learn
- Pandas/Numpy
- Category Encoders
- Jupyter Notebook


