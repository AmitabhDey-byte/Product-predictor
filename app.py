import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Product Predictor", layout="wide")
st.title("Product Predictor Dashboard")

log_model = joblib.load(open('model1.pkl', 'rb'))
rf_model = joblib.load(open('model2.pkl', 'rb'))
st.sidebar.title("Settinhs")

model_choice = st.sidebar.selectbox("Choose Model",["Logistic Regression","Random Forest"])

st.subheader("Enter product details")

col1,col2 = st.columns(2)
with col1:
    discounted_price = st.number_input("Discounted Price",min_value=0.0)

with col2:
    actual_price = st.number_input("Actual Price",min_value=0.0)

    rating_count = st.number_input("Rating Count",min_value=0)

    price_diff = actual_price - discounted_price
    discounted_ratio = discounted_price/actual_price if actual_price !=0 else 0
    log_rating_count =np.log1p(rating_count)

    features = np.array([[discounted_price, discounted_ratio,log_rating_count,price_diff,actual_price]])

    st.subheader("Prediction")

    if st.button("Predict"):
        model = log_model if model_choice == "Logistic Regression" else rf_model

        prediction = model.predict(features)
        if prediction[0] ==1:
          st.success("Good Product..rating high hoga")
        else:
          st.error("Bad Product..mat low yeh sab")
        ingest = pd.read_csv('data/amazon.csv')

        ingest['discounted_price'] = ingest['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        ingest['actual_price'] = ingest['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        ingest['discount_percentage'] = ingest['discount_percentage'].str.replace('%', '').astype(float)
        ingest['rating_count'] = ingest['rating_count'].str.replace(',', '').astype(float)

        ingest = ingest.dropna()
        ingest['price_diff'] = ingest['actual_price'] - ingest['discounted_price']
        ingest['discounted_ratio'] = ingest['discounted_price'] / ingest['actual_price']
        ingest['log_rating_count'] = np.log1p(ingest['rating_count'])

        st.subheader("Smart Recommendation")
        features_ingest = ingest[['discounted_price','discounted_ratio','log_rating_count','price_diff','actual_price']]
        scaler = StandardScaler()
        scaler_data = scaler.fit_transform(features_ingest)

        user_scaled = scaler.transform(features)
        similarity = cosine_similarity(user_scaled, scaler_data)

        top_indices = similarity[0].argsort()[-15:][::-1]
        candidates= ingest.iloc[top_indices].copy()

        candidates_features= candidates[['discounted_price','discounted_ratio','log_rating_count','price_diff','actual_price']]

        preds = rf_model.predict(candidates_features)

        candidates['prediction'] = preds

        good_products = candidates[candidates['prediction']==1]

        final_recommendations = good_products.head(5)

        if len(final_recommendations)>0:
         st.dataframe(final_recommendations[['product_name','discounted_price','rating','rating_count']])
        else:
         st.warning("No strong recommendations found")









