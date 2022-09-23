# !pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("Machine learning Prediction Project in Streamlit")

# !pip install sklearn
import sklearn
import pickle
filename = 'xgb_model_new1'
model = pickle.load(open(filename, 'rb'))

# "age", "hp_kw", "km", "gears", "make_model"

age = st.sidebar.number_input("age:",min_value=0, max_value=3)
hp_kw = st.sidebar.number_input("hp_kw:",min_value=40, max_value=239)
km = st.sidebar.number_input("km:",min_value=0, max_value=317000)
gears = st.sidebar.number_input("gears:",min_value=5, max_value=8)
make_model = st.sidebar.number_input("make_model:",min_value=0, max_value=7)


my_dict = {
    "age": age,
    "hp_kw": hp_kw,
    "km": km,
    "gears": gears,
    "make_model": make_model}
​
df=pd.DataFrame.from_dict([my_dict])
st.table(df)
​
if st.button("Predict"): 
    pred = model.predict(df)
    st.write(pred[0])
