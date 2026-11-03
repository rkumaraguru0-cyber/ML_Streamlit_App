import streamlit as st
import joblib
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# Load model and evaluation results
model = joblib.load("iris_model.pkl")

with open("evaluation.json", "r") as file:
    results = json.load(file)

iris = load_iris()

# Prepare dataset
df = pd.DataFrame(
    iris.data,
    columns=["Sepal Length", "Sepal Width",
             "Petal Length", "Petal Width"]
)

df["Flower"] = [
    iris.target_names[i] for i in iris.target
]

# Page title
st.title("Iris Flower Prediction")
st.write("Machine Learning Model Evaluation and Prediction")

# Accuracy
st.subheader("Model Accuracy")
st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")

# Scatter plot
st.subheader("Iris Flower Dataset - Scatter Plot")

fig = px.scatter(
    df,
    x="Petal Length",
    y="Petal Width",
    color="Flower",
    title="Petal Length vs Petal Width",
    labels={
        "Petal Length": "Petal Length (cm)",
        "Petal Width": "Petal Width (cm)"
    }
)

# Prediction inputs
st.subheader("Predict Iris Flower")

sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    predicted_flower = flowers[pred[0]]

    st.success(f"Predicted Flower: {predicted_flower}")

    # Add red prediction point
    fig.add_trace(
        go.Scatter(
            x=[pl],
            y=[pw],
            mode="markers",
            name="Prediction",
            marker=dict(
                color="red",
                size=14,
                symbol="circle"
            )
        )
    )

# Display graph
st.plotly_chart(fig, width="stretch")
