import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

def analyze_reviews():

    # 1. LOAD DATA

    df = pd.read_csv("data/restaurant_reviews.csv")

    # Load the sentiment model trained on the Kaggle dataset
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    print("\nSentiment model loaded successfully!")

    # 2. PREDICT SENTIMENT OF RESTAURANT REVIEWS

    review_tfidf = vectorizer.transform(df["review"])

    df["predicted_sentiment"] = model.predict(review_tfidf)

    # Validate model on restaurant review data
    validation_accuracy = accuracy_score(
        df["sentiment"],
        df["predicted_sentiment"]
    )

    print("\nRestaurant Dataset Validation Accuracy:",
        validation_accuracy)

    print("\nRestaurant Dataset Classification Report:")
    print(
        classification_report(
            df["sentiment"],
            df["predicted_sentiment"]
        )
    )

    print("\nSentiment predictions completed!")

    print("\nRestaurant Review Sentiments:")
    print(
        df[
            ["restaurant", "review", "predicted_sentiment"]
        ].head()
    )

    # 3. IDENTIFY COMPLAINTS

    complaint_phrases = {
        "slow delivery": [
            "slow delivery",
            "delivery was slow",
            "delivery is slow",
            "delivery was late",
            "delivery is late",
            "late delivery",
            "delivery delay",
            "delivery was delayed",
            "delivery is delayed",
            "delivered late",
            "arrived late",
            "took too long",
            "delivery took too long",
            "order took too long",
            "long delivery time",
            "delay in delivery",
            "delayed delivery"
        ],

        "cold food": [
            "cold food",
            "food was cold",
            "food is cold",
            "meal was cold",
            "meal is cold",
            "not hot",
            "not warm",
            "lukewarm"
        ],

        "high price": [
            "too expensive",
            "very expensive",
            "overpriced",
            "too costly",
            "very costly",
            "high price",
            "high prices",
            "price is high",
            "prices are high",
            "price was high"
        ],

        "small portions": [
            "small portion",
            "small portions",
            "portion was small",
            "portions were small",
            "portion size",
            "small serving",
            "small servings",
            "not enough food",
            "too little food"
        ],

        "rude staff": [
            "rude staff",
            "staff was rude",
            "staff were rude",
            "staff is rude",
            "staff are rude",
            "unfriendly staff",
            "unhelpful staff",
            "staff behaved badly",
            "bad staff behaviour",
            "bad staff behavior"
        ]
    }

    def identify_complaint(review):

        review = review.lower()

        for complaint, phrases in complaint_phrases.items():

            if any(phrase in review for phrase in phrases):
                return complaint

        return "other"

    # Create data for Tableau dashboard
    dashboard_data = df.copy()

    dashboard_data["negative"] = (
        dashboard_data["predicted_sentiment"] == "negative"
    )

    dashboard_data["complaint"] = "other"

    negative_mask = (
        dashboard_data["predicted_sentiment"] == "negative"
    )

    dashboard_data.loc[
        negative_mask,
        "complaint"
    ] = dashboard_data.loc[
        negative_mask,
        "review"
    ].apply(identify_complaint)

    # Save data for Tableau
    dashboard_data.to_csv(
        "data/dashboard_data.csv",
        index=False
    )

    print("\nDashboard data created successfully!")

    print("\nTotal Reviews:", len(df))
    print("\nAverage Rating:", round(df["rating"].mean(), 2))

    # 4. FIND NEGATIVE REVIEWS USING ML PREDICTIONS

    negative_reviews = df[
        df["predicted_sentiment"] == "negative"
    ]

    print("\nNegative Reviews:")
    print(negative_reviews)

    # Count negative reviews for each restaurant
    negative_count = negative_reviews.groupby("restaurant").size()

    print("\nNegative Reviews by Restaurant:")
    print(negative_count)

    # 5. COMPLAINT ANALYSIS

    complaint_data = []

    # Analyze each negative review
    for _, row in negative_reviews.iterrows():

        restaurant = row["restaurant"]
        review = row["review"]

        complaint = identify_complaint(review)

        complaint_data.append({
            "restaurant": restaurant,
            "complaint": complaint,
            "review": review
        })

    # Create DataFrame from complaint analysis
    complaint_df = pd.DataFrame(complaint_data)

    print("\nComplaint Analysis:")
    print(complaint_df)

    # Count complaints for each restaurant
    ranked_complaints = (
        complaint_df[
            complaint_df["complaint"] != "other"
        ]
        .groupby(["restaurant", "complaint"])
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["restaurant", "count"],
            ascending=[True, False]
        )
    )

    print("\nRanked Complaints:")
    print(ranked_complaints)

    # Save complaint analysis
    ranked_complaints.to_csv(
        "data/complaint_analysis.csv",
        index=False
    )

    print("\nComplaint analysis saved successfully!")

    # 6. BUSINESS RECOMMENDATIONS

    recommendations = {
        "cold food":
            "Improve food packaging and delivery handling to maintain food temperature.",

        "slow delivery":
            "Review the order-to-delivery process and reduce delivery delays.",

        "high price":
            "Compare pricing with competitors and review the pricing strategy.",

        "small portions":
            "Review portion sizes and compare them with customer expectations.",

        "rude staff":
            "Provide staff training focused on customer service and communication."
    }

    print("\nBUSINESS INSIGHTS")

    insights_data = []

    for restaurant in ranked_complaints["restaurant"].unique():

        restaurant_data = ranked_complaints[
            ranked_complaints["restaurant"] == restaurant
        ]

        top_row = restaurant_data.iloc[0]

        top_complaint = top_row["complaint"]
        complaint_count = top_row["count"]

        negative_reviews_count = negative_count.get(
            restaurant, 0
        )

        recommended_action = recommendations.get(
            top_complaint,
            "Investigate the issue further."
        )

        print("\nRestaurant:", restaurant)
        print("Negative Reviews:", negative_reviews_count)
        print("Top Complaint:", top_complaint)
        print("Complaint Count:", complaint_count)
        print("Recommended Action:", recommended_action)

        insights_data.append({
            "restaurant": restaurant,
            "negative_reviews": negative_reviews_count,
            "top_complaint": top_complaint,
            "complaint_count": complaint_count,
            "recommended_action": recommended_action
        })

    # 7. SAVE FINAL BUSINESS INSIGHTS

    business_insights = pd.DataFrame(insights_data)

    business_insights.to_csv(
        "data/business_insights.csv",
        index=False
    )

    print("\nBusiness insights created successfully!")
    print(business_insights)

analyze_reviews()