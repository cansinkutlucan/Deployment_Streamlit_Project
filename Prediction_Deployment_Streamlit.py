# !pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("Machine learning Prediction Project in Streamlit")

# !pip install sklearn
import sklearn
import pickle
filename = 'rf_model_new3'
model = pickle.load(open(filename, 'rb'))

Age = st.sidebar.number_input("Age:",min_value=0, max_value=3)
hp_kW = st.sidebar.number_input("hp_kW:",min_value=40, max_value=239)
km = st.sidebar.number_input("km:",min_value=0, max_value=317000)
Gears = st.sidebar.number_input("Gears:",min_value=5, max_value=8)

my_dict = {"hp_kW":hp_kW, "Age":Age, "km":km, "Gears":Gears}
df = pd.DataFrame.from_dict([my_dict])

cols = {"hp_kW": "Horse Power", "Age": "Age", "km": "km Traveled","Gears": "Gears"}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)
st.write("Selected Specs: \n")
st.table(df_show)

if st.button("Predict"):
    pred = model.predict(df)
    pred


st.write("\n\n")

