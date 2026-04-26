
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


# ============================================================
# Custom classes used when the models were saved
# These MUST exist before loading xgboost.pkl / knn.pkl
# ============================================================

class AirlineFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Recreates the feature engineering used during training.

    It creates:
    - ArrivalDelay_missing
    - Total Delay Minutes

    The saved pipeline then uses these engineered columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Arrival Delay in Minutes" in X.columns:
            X["ArrivalDelay_missing"] = X["Arrival Delay in Minutes"].isna().astype(int)

        if "Departure Delay in Minutes" in X.columns and "Arrival Delay in Minutes" in X.columns:
            X["Total Delay Minutes"] = (
                X["Departure Delay in Minutes"].fillna(0)
                + X["Arrival Delay in Minutes"].fillna(0)
            )

        return X


class EarlyStoppingXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Lightweight wrapper needed only for loading the saved XGBoost pipeline.
    The real fitted XGBoost model is stored inside self.model_.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def __getattr__(self, name):
        if name != "model_" and hasattr(self, "model_"):
            return getattr(self.model_, name)
        raise AttributeError(name)


# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="Airline Customer Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Customer Satisfaction Predictor")
st.write(
    "This app predicts whether an airline passenger is likely to be **satisfied** "
    "or **neutral/dissatisfied** based on customer profile, travel information, "
    "service ratings, and delay details."
)


# ============================================================
# Load models
# ============================================================

@st.cache_resource
def load_models():
    xgb_model = joblib.load("xgboost.pkl")
    knn_model = joblib.load("knn.pkl")
    return {
        "XGBoost": xgb_model,
        "KNN": knn_model
    }


try:
    models = load_models()
except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Choose prediction model",
    ["XGBoost", "KNN"],
    index=0
)

model = models[model_name]

st.sidebar.info(
    "Use XGBoost as the main model for final prediction. "
    "KNN is useful for comparison."
)


# ============================================================
# Input form
# ============================================================

with st.form("prediction_form"):
    st.subheader("👤 Passenger Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer", "disloyal Customer"]
        )

    with col3:
        age = st.slider("Age", 7, 85, 35)

    with col4:
        travel_type = st.selectbox(
            "Type of Travel",
            ["Business travel", "Personal Travel"]
        )

    st.subheader("🛫 Flight Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        travel_class = st.selectbox(
            "Class",
            ["Business", "Eco", "Eco Plus"]
        )

    with col2:
        flight_distance = st.number_input(
            "Flight Distance",
            min_value=0,
            max_value=10000,
            value=1000,
            step=50
        )

    with col3:
        missing_arrival_delay = st.checkbox(
            "Arrival delay is missing / unknown",
            value=False
        )

    col1, col2 = st.columns(2)

    with col1:
        departure_delay = st.number_input(
            "Departure Delay in Minutes",
            min_value=0,
            max_value=2000,
            value=0,
            step=5
        )

    with col2:
        if missing_arrival_delay:
            arrival_delay = np.nan
            st.number_input(
                "Arrival Delay in Minutes",
                min_value=0,
                max_value=2000,
                value=0,
                step=5,
                disabled=True
            )
        else:
            arrival_delay = st.number_input(
                "Arrival Delay in Minutes",
                min_value=0,
                max_value=2000,
                value=0,
                step=5
            )

    st.subheader("⭐ Service Ratings")
    st.caption("Ratings usually range from 0 to 5, where higher means better service.")

    service_cols = st.columns(3)

    with service_cols[0]:
        inflight_wifi = st.slider("Inflight wifi service", 0, 5, 3)
        dep_arr_time = st.slider("Departure/Arrival time convenient", 0, 5, 3)
        ease_booking = st.slider("Ease of Online booking", 0, 5, 3)
        gate_location = st.slider("Gate location", 0, 5, 3)
        food_drink = st.slider("Food and drink", 0, 5, 3)

    with service_cols[1]:
        online_boarding = st.slider("Online boarding", 0, 5, 3)
        seat_comfort = st.slider("Seat comfort", 0, 5, 3)
        inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
        onboard_service = st.slider("On-board service", 0, 5, 3)
        leg_room = st.slider("Leg room service", 0, 5, 3)

    with service_cols[2]:
        baggage_handling = st.slider("Baggage handling", 0, 5, 3)
        checkin_service = st.slider("Checkin service", 0, 5, 3)
        inflight_service = st.slider("Inflight service", 0, 5, 3)
        cleanliness = st.slider("Cleanliness", 0, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction", use_container_width=True)




# ============================================================
# SHAP explanation helpers
# ============================================================

def get_final_estimator(pipeline):
    """Return the final classifier from a sklearn Pipeline."""
    if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps:
        clf = pipeline.named_steps["clf"]
        return getattr(clf, "model_", clf)
    return pipeline


def get_transformed_feature_names(pipeline, X_transformed):
    """
    Try to recover feature names after preprocessing/selection.
    If PCA exists, SHAP is explained over principal components.
    """
    n_features = X_transformed.shape[1]

    if not hasattr(pipeline, "named_steps"):
        return [f"feature_{i}" for i in range(n_features)]

    steps = pipeline.named_steps

    # If PCA exists before classifier, transformed features are components.
    if "pca" in steps:
        return [f"PC_{i + 1}" for i in range(n_features)]

    try:
        if "preprocessor" in steps:
            names = steps["preprocessor"].get_feature_names_out()
            names = np.array(names, dtype=object)

            if "selector" in steps:
                mask = steps["selector"].get_support()
                names = names[mask]

            if len(names) == n_features:
                return [str(name) for name in names]
    except Exception:
        pass

    return [f"feature_{i}" for i in range(n_features)]


def explain_xgboost_prediction(pipeline, input_df):
    """
    Compute SHAP values for one prediction using the fitted XGBoost model
    inside the saved sklearn pipeline.
    """
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("SHAP explanation requires a sklearn Pipeline.")

    # Transform raw app input using all pipeline steps except the classifier.
    X_transformed = pipeline[:-1].transform(input_df)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)

    feature_names = get_transformed_feature_names(pipeline, X_transformed)

    final_model = get_final_estimator(pipeline)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_transformed)

    # SHAP may return either:
    # - list[class_0_values, class_1_values]
    # - ndarray with shape (n_samples, n_features)
    # - ndarray with shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            values = shap_values[0, :, 1]
        elif shap_values.ndim == 2:
            values = shap_values[0]
        else:
            values = shap_values.reshape(-1)

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "absolute_impact": np.abs(values)
    }).sort_values("absolute_impact", ascending=False)

    return explanation_df


def plot_shap_bar(explanation_df, top_n=12):
    """
    Simple matplotlib SHAP bar chart.
    Positive SHAP values push prediction toward satisfaction;
    negative values push prediction toward neutral/dissatisfied.
    """
    top = explanation_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["shap_value"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP value")
    ax.set_title("Top SHAP Drivers for This Prediction")
    fig.tight_layout()
    return fig


# ============================================================
# Prediction
# ============================================================

input_data = pd.DataFrame([{
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": travel_class,
    "Flight Distance": flight_distance,
    "Inflight wifi service": inflight_wifi,
    "Departure/Arrival time convenient": dep_arr_time,
    "Ease of Online booking": ease_booking,
    "Gate location": gate_location,
    "Food and drink": food_drink,
    "Online boarding": online_boarding,
    "Seat comfort": seat_comfort,
    "Inflight entertainment": inflight_entertainment,
    "On-board service": onboard_service,
    "Leg room service": leg_room,
    "Baggage handling": baggage_handling,
    "Checkin service": checkin_service,
    "Inflight service": inflight_service,
    "Cleanliness": cleanliness,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay,
}])


if submitted:
    st.divider()

    prediction = model.predict(input_data)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(input_data)[0]
        except Exception:
            probability = None

    label_map = {
        0: "Neutral / Dissatisfied",
        1: "Satisfied"
    }

    predicted_label = label_map.get(int(prediction), str(prediction))

    if predicted_label == "Satisfied":
        st.success("✅ Predicted Customer Satisfaction: Satisfied")
    else:
        st.warning("⚠️ Predicted Customer Satisfaction: Neutral / Dissatisfied")

    if probability is not None:
        satisfied_prob = float(probability[1])
        dissatisfied_prob = float(probability[0])

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probability of Satisfaction",
                f"{satisfied_prob:.1%}"
            )

        with col2:
            st.metric(
                "Probability of Neutral / Dissatisfied",
                f"{dissatisfied_prob:.1%}"
            )

        prob_df = pd.DataFrame({
            "Class": ["Neutral / Dissatisfied", "Satisfied"],
            "Probability": [dissatisfied_prob, satisfied_prob]
        })

        st.bar_chart(prob_df.set_index("Class"))

    with st.expander("See input data used for prediction"):
        st.dataframe(input_data, use_container_width=True)


    st.subheader("🔍 SHAP Explanation")

    if model_name == "XGBoost":
        try:
            explanation_df = explain_xgboost_prediction(model, input_data)

            st.caption(
                "Positive SHAP values push the prediction toward **Satisfied**. "
                "Negative SHAP values push it toward **Neutral / Dissatisfied**."
            )

            top_explanation = explanation_df.head(12).copy()
            top_explanation["shap_value"] = top_explanation["shap_value"].round(4)
            top_explanation["absolute_impact"] = top_explanation["absolute_impact"].round(4)

            st.dataframe(
                top_explanation,
                use_container_width=True,
                hide_index=True
            )

            fig = plot_shap_bar(explanation_df, top_n=12)
            st.pyplot(fig)

        except Exception as e:
            st.error("SHAP explanation could not be generated for this prediction.")
            st.exception(e)
    else:
        st.info(
            "SHAP explanation is currently enabled for the XGBoost model only. "
            "KNN explanations require slower model-agnostic SHAP, which is not ideal for a live Streamlit app."
        )



# ============================================================
# Batch prediction section
# ============================================================

st.divider()
st.subheader("📄 Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file with the same feature columns as the training dataset",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(batch_df.head(), use_container_width=True)

    if st.button("Run Batch Prediction", use_container_width=True):
        try:
            batch_predictions = model.predict(batch_df)

            result_df = batch_df.copy()
            result_df["prediction"] = batch_predictions
            result_df["prediction_label"] = result_df["prediction"].map({
                0: "Neutral / Dissatisfied",
                1: "Satisfied"
            })

            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(batch_df)
                    result_df["prob_neutral_dissatisfied"] = probs[:, 0]
                    result_df["prob_satisfied"] = probs[:, 1]
                except Exception:
                    pass

            st.success("Batch prediction completed.")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="airline_satisfaction_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error("Batch prediction failed. Please check that your CSV has the correct columns.")
            st.exception(e)
