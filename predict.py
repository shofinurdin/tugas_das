
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import numpy as np


# def load_model():
# 	model_gb=load('model.joblib')
# 	return model_gb

def run_predict_app():
    
    st.subheader("Classification Prediction")
    
    #model_upload = st.file_uploader('Upload Dataset ', type=['joblib'])
    model_file=joblib.load('model.joblib')
    #modelfile = open("model.pkl", "rb")
    
    
    st.sidebar.title("Features")
    #Intializing
    sl = st.sidebar.slider(label="Sepal Length (cm)",value=5.2,min_value=0.0, max_value=8.0, step=0.1)
    sw = st.sidebar.slider(label="Sepal Width (cm)",value=3.2,min_value=0.0, max_value=8.0, step=0.1)
    pl = st.sidebar.slider(label="Petal Length (cm)",value=4.2,min_value=0.0, max_value=8.0, step=0.1)
    pw = st.sidebar.slider(label="Petal Width (cm)",value=1.2,min_value=0.0, max_value=8.0, step=0.1)

    if model_file is not None:
        if st.button("Click Here to Classify"):
            dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            input_variables = np.array(dfvalues[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
            st.write('Input :')
            data_input=pd.DataFrame(data=input_variables, columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
            st.dataframe(data_input)
            #model = joblib.load(model_upload)
            prediction = model_file.predict(input_variables)
            st.write('Prediction :')
            st.success(prediction[0])
            if prediction == 'Iris-setosa':
                st.image('setosa.png')
            elif prediction == 'Iris-versicolor':
                st.image('versicolor.png')
            elif prediction == 'Iris-virginica':
                st.image('virginica.png')
    else:
        st.write('model belum diupload')