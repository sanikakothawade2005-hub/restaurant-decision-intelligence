# Decision Intelligence Platform

A restaurant analytics project that turns customer reviews into sentiment insights, ranked complaints, and recommended actions, with the results visualized in a Tableau dashboard.

## Overview

Restaurants receive a lot of written feedback, but few owners have time to read all of it. This project automates the first step: it classifies each review as positive or negative, finds what customers complain about most at each restaurant, identifies the highest-frequency complaint and suggests a corresponding action.

## Features

- Sentiment classification using TF-IDF and Logistic Regression
- Complaint detection using rule-based keyword matching
- Complaint ranking by restaurant
- Rule-based business recommendations for the top complaint
- Interactive Tableau dashboard for exploring restaurant insights

## Workflow

```text
Restaurant Reviews
        ↓
Sentiment Analysis
        ↓
Negative Review Detection
        ↓
Complaint Detection
        ↓
Complaint Ranking
        ↓
Business Recommendations
        ↓
Tableau Dashboard
```

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, Matplotlib, Joblib
- **Techniques:** TF-IDF, Logistic Regression
- **Visualization:** Tableau

## Model Performance

The sentiment model was trained on a public food-review dataset and evaluated on two separate datasets.

| Dataset             | Purpose                             |          Reviews | Accuracy |
| ------------------- | ----------------------------------- | ---------------: | -------: |
| Kaggle Food Reviews | Model training and held-out testing | 200 test reviews |   85.00% |
| Restaurant Reviews  | Cross-dataset validation            |      107 reviews |   86.91% |

> **Note:** The restaurant dataset is a small, curated dataset created for project demonstration. The 86.91% result should not be interpreted as real-world production accuracy.

## Datasets

| Dataset | Description | Used for |
| ------- | ----------- | -------- |
| `food_reviews.csv` | Public Kaggle dataset of 1,000 labeled food reviews (liked / not liked) | Training and testing the sentiment model |
| `restaurant_reviews.csv` | Small custom dataset of 107 reviews for two restaurants (Urban Bites and Spice Garden) | Cross-dataset validation and business analysis |

## Dataset Attribution

The sentiment analysis model was trained using a 1,000-review restaurant sentiment dataset sourced from Kaggle.

The dataset is licensed under the Open Data Commons Database Contents License (DbCL) v1.0.

License: https://opendatacommons.org/licenses/dbcl/1-0/

The dataset is included in this repository for demonstrating the sentiment analysis component of the Restaurant Decision Intelligence Platform.

## Outputs

Running the analysis creates three CSV files in the `data/` folder:

| File | Contents |
| ---- | -------- |
| `dashboard_data.csv` | One row per review: restaurant, rating, review text, reference and predicted sentiment, negative flag, and detected complaint. Used as the Tableau data source. |
| `complaint_analysis.csv` | Complaint counts for each restaurant, ranked from most to least frequent. |
| `business_insights.csv` | One row per restaurant: number of negative reviews, top complaint, complaint count, and recommended action. |

The recommendations are rule-based: each complaint category is mapped to a fixed recommended action.

## Tableau Dashboard

The dashboard shows:

- Total reviews
- Average rating
- Negative reviews
- Top customer complaints
- Complaints by restaurant
- Restaurant rating comparison

**Live dashboard:** [View on Tableau Public](https://public.tableau.com/app/profile/sanika07/viz/Restaurant_Decision_Intelligence_Platform/RestaurantDecisionDashboard)

## Getting Started

**Requirements:** Python 3 (Tableau Public or Tableau Desktop is only needed for the dashboard).

**1. Clone the repository**

```bash
git clone <repository-url>
cd decision-intelligence-platform
```

**2. Install the dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the sentiment model**

```bash
python analysis/sentiment_analysis.py
```

This trains the TF-IDF + Logistic Regression model and saves the model and vectorizer to the `models/` folder.

**4. Run the restaurant review analysis**

```bash
python analysis/review_analysis.py
```

This predicts sentiment, detects and ranks complaints, generates recommendations, and creates the CSV files described above.

**5. Open the dashboard data in Tableau**

Connect Tableau to data/dashboard_data.csv to build or refresh the dashboard.

## Project Structure

```text
decision-intelligence-platform/
│
├── analysis/
│   ├── sentiment_analysis.py
│   └── review_analysis.py
│
├── data/
│   ├── food_reviews.csv
│   ├── restaurant_reviews.csv
│   ├── dashboard_data.csv
│   ├── complaint_analysis.csv
│   └── business_insights.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

- `analysis/`: model training and restaurant review analysis scripts
- `data/`: input datasets and generated analysis outputs
- `models/`: trained sentiment model and TF-IDF vectorizer
- `requirements.txt`: Python dependencies

## Limitations

- The restaurant review dataset is small and curated for demonstration.
- Complaint detection uses rule-based keyword matching, so unusual wording may be missed. Each review is assigned a single complaint category.
- Business recommendations are rule-based, not learned from historical business outcomes.
- The system analyzes customer feedback only. It cannot explain changes in sales, because revenue, order history, and time-series data are not included.

## Future Scope

- Test on larger real-world restaurant review datasets
- Improve complaint detection with more advanced NLP techniques
- Add sales, revenue, order, and customer-retention data
- Add time-based trend analysis and competitor comparison
- Extend the approach to other industries

## Author

**Sanika Kothawade**