
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#---css----
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.4);
}

/* Subheaders */
h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

/* Card effect */
.block-container {
    padding: 2rem 3rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}



/* Button */
.stButton > button {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    color: white;
    border-radius: 25px;
    height: 3em;
    width: 220px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #ee0979, #ff6a00);
}

/* Success box */
.stSuccess {
    background: rgba(0, 255, 150, 0.2);
    border-radius: 10px;
    padding: 15px;
    font-size: 20px;
}

/* Dataframe */
.css-1d391kg {
    border-radius: 10px;
}

/* Heatmap spacing */
.element-container {
    margin-top: 20px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #ff6a00;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# css End---


# Page 
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App")

# Load Dataset
df = pd.read_csv("Cars Dataset.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Dataset Preview
st.subheader("Dataset Preview")
st.write(df.head())

# Heatmap
st.subheader("Heatmap Correlation")

numeric_df = df.select_dtypes(include=['int64','float64'])

fig, ax = plt.subplots()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

# Model Training
X = df.drop('price', axis=1)
y = df['price']

ohe = OneHotEncoder(handle_unknown='ignore')

ct = make_column_transformer(
    (ohe, ['name','company','fuel_type']),
    remainder='passthrough'
)

model = LinearRegression()

pipe = make_pipeline(ct, model)

pipe.fit(X, y)

# User Input
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "Select Company",
        sorted(df['company'].dropna().unique())
    )

    car_models = df[df['company'] == company]['name'].unique()

    name = st.selectbox(
        "Select Car Name",
        sorted(car_models)
    )

    fuel = st.selectbox(
        "Fuel Type",
        df['fuel_type'].dropna().unique()
    )

with col2:

    year = st.number_input("Manufacturing Year", 1995, 2025)

    kms = st.number_input("Kilometers Driven", 0, 500000)

# Prediction
if st.button("Predict Price"):

    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=['name','company','year','kms_driven','fuel_type']
    )

    prediction = pipe.predict(input_df)[0]

    prediction = int(prediction)

    st.success(f"Estimated Car Price: ₹ {prediction:,}")