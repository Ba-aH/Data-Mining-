# 🍄 Mushroom Classification Analysis

**Comparing Random Forest vs KNN with/without preprocessing**  
*A Python data mining project demonstrating the impact of preprocessing on classifier performance*

## 🔍 Key Features
- Implements **Random Forest** and **KNN** classifiers
- Tests **raw data** vs **preprocessed data** (cleaning, encoding, scaling)
- Evaluates using **precision, recall, F1-score** metrics
- 80/20 train-test split with **5 repeated experiments**

## 📊 Key Findings
| Algorithm       | Best F1-Score | Error Rate |
|-----------------|---------------|------------|
| Random Forest   | 0.61          | 0.43       |
| KNN (k=3)       | 0.51          | 0.57       |

*Preprocessing improved Random Forest performance by 8.9%*

## 🛠️ Tech Stack
- Python 3
- Scikit-learn
- Pandas/Numpy
- Category Encoders
- Jupyter Notebook


