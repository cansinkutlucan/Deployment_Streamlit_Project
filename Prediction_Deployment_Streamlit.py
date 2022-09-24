# !pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv("final_scout_not_get_dummy.csv")
df_ = df.copy()
df.columns = [x.lower() for x in df.columns]
df.head()
df = df[["age", "hp_kw", "km", "gears", "make_model"]]
df.describe()

st.title("Machine learning Prediction Project in Streamlit")

import pickle
filename = 'rf_model_new1'
model = pickle.load(open(filename, 'rb'))

make_model = st.sidebar.radio("Name of the Car Model:", df["make_model"].unique())
hp_kw = st.sidebar.slider("hp_kw:",min_value=40.00, max_value=295.00,value=40.00, step=5.0)
age = st.sidebar.slider("age:",min_value=0.0, max_value=3.0,value=2.0, step=1.0)
km = st.sidebar.slider("km:",min_value=0.0, max_value=320.000,value=0.0, step=10.000)
gears = st.sidebar.slider("gears:",min_value=5.0, max_value=8.0,value=5.0, step=1.0)

my_dict = {"make_model":make_model, "hp_kW":hp_kw, "age":age, "km":km, "gears":gears}
df = pd.DataFrame.from_dict([my_dict])

cols = {"make_model":"make_model","hp_kw": "Horse Power","age": "age", "km": "km Traveled",'gears':'gears'}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)
st.write("Selected Specs: \n")
st.table(df_show)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
label_encoder(df, 'make_model')


if st.button("Predict"):
    pred = model.predict(df)
    pred

st.write("\n\n")

