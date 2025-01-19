import streamlit as st
import pandas as pd
st.title('Deploy nas Nuvens')

st.info("Este app esta tocando nas nuvens")

with st.expander('Data'):
  st.write('**Dados**')
  dados = pd.read_csv(https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv)
  dados
