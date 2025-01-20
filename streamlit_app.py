import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Deploy nas Nuvens')

st.info("Este app esta tocando nas nuvens")

with st.expander('Data'):
  st.write('**Dados**')
  dados = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")

  X_raw = dados.drop('species', axis =1)
  y_raw = dados.species
  
  col1, col2 = st.columns([4,1])
    
  with col1:
      st.write('**X**')
      st.dataframe(X_raw, width=600) 

  with col2:
      st.write('**y**')
      st.dataframe(y_raw, width=300)  


with st.expander("Visualizacoes"):
  st.scatter_chart(data=dados, x='bill_length_mm', y='body_mass_g', color='species')

with  st.sidebar: #dados de entrada
  st.header('Variaveis de Entrada')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen' 'Torgersen'))
  bill_length_mm = st.slider('Bill lenght (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper lenght (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))


#CRiando um df dos dados de entrada
  entrada = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': gender
  }
  df_entrada = pd.DataFrame(entrada, index=[0])
  penguins_entrada = pd.concat([df_entrada, X_raw], axis=0)
#df_entrada


with st.expander('Input features'):
  st.write('**Pinguin de entrada')
  df_entrada
  st.write('**Penguins combinados')
  penguins_entrada


#Preparando os dados
# categoria -> dummy  - X 
encode = ['island', 'sex']
df_penguis = pd.get_dummies(penguins_entrada, prefix=encode)
X = df_penguis[1:]
input_raw = df_penguis[:1]

#Encode y
target_map = {'Adelie': 0,
              'Chinstrap': 1,
              'Gentoo': 2}

def target_encode(val):
  return target_map[val]

y = y_raw.apply(target_encode)


with st.expander('Preparacao'):

  st.write('Saida com encoded (X)')
  input_raw 
  st.write('**Encoded y**')
  y

#TReinamento do modelo e inferindo
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_raw)
prediction_proba = clf.predict_proba(input_raw)

df_predict_proba = pd.DataFrame(prediction_proba)
df_predict_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_predict_proba.rename(columns={0 : 'Adelie',
                                 1: 'Chinstrap',
                                 2 : 'Gentoo'})
#df_predict_proba

#MOstrando as probabilidades
st.subheader('Predizendo as Especies')
st.dataframe(df_predict_proba,
             column_config={
              'Adelie' : st.column_config.ProgressColumn(
              'Adelie',
               format='%f',
               width='medium',
               min_value=0,
               max_value=1
             ), 
              'Chinstrap' : st.column_config.ProgressColumn(            
               'Chinstrap',
               format='%f',
               width='medium',
               min_value=0,
               max_value=1
               ),

              'Gentoo' : st.column_config.ProgressColumn(            
               'Gentoo',
               format='%f',
               width='medium',
               min_value=0,
               max_value=1
               ),              
               
             }, hide_index=True)


#especies_p = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
especies_p = df_predict_proba.columns
st.success(str(especies_p[prediction][0]))
#st.info("Debug1")
