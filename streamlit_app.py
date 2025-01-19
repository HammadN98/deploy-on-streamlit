import streamlit as st
import pandas as pd
st.title('Deploy nas Nuvens')

st.info("Este app esta tocando nas nuvens")

url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv"
dados = pd.read_csv(url)
dados.head()
