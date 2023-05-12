
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def load_data(data):
    data_iris=pd.read_csv(data, index_col='Id')
    return data_iris


def rename_columns(dataframe):
    old_names = ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    new_names = ['sepal_length', 'sepal_width', 'petal_length','petal_width','species']
    new_columns = dict(zip(old_names, new_names))
    return dataframe.rename(columns=new_columns)
    


def run_eda_app():
    st.subheader('EDA Menu')
    data_= st.file_uploader('Upload Dataset ', type=['csv'])

    if data_ is not None:
        data_frame=load_data(data_)
        # with st.expander('Data Frame'):
        #st.dataframe(data_frame)
        with st.expander('Data Frame'):
            st.write(data_frame)

        with st.expander('Rename Column'):
            df_new = rename_columns(data_frame)
            st.dataframe(df_new)

        with st.expander('Data Info'):
            col1,col2 = st.columns([1,1])
            with col1:
                st.write('Data Shape')
                st.dataframe(df_new.shape)
            with col2:
                st.write('Data Types')
                st.dataframe(df_new.dtypes)

        with st.expander('Null Detection'):
            st.dataframe(df_new.isna().sum())

        with st.expander('Statistic'):
            col1,col2 = st.columns([1,1])
            with col1:
                st.write('Describe')
                st.dataframe(df_new.describe())
            with col2:
                st.write('Median')
                st.dataframe(df_new.groupby('species').median())

        with st.expander('Matrix Correlation'):
            corr = df_new.corr()
            fig, ax = plt.subplots(figsize=(3,3))
            sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax, annot_kws={"size":5})
            ax.set_title('Matriks Korelasi')
            plt.xticks(rotation=80)
            st.pyplot(fig)
       
        with st.expander('Pairplot'):
            sns.pairplot(data=df_new, hue='species')
            st.pyplot(plt)

        with st.expander('Scatterplot'):
            col1,col2 = st.columns([1,1])
            with col1:
                fig, ax= plt.subplots()
                plt.title('scatter plot sepal')
                sns.scatterplot(data=df_new, x='sepal_length', y='sepal_width', hue='species')
                st.pyplot(fig)
            with col2:
                fig, ax= plt.subplots()
                plt.title('scatter plot petal')
                sns.scatterplot(data=df_new, x='petal_length', y='petal_width', hue='species')
                st.pyplot(fig)

        with st.expander('Boxplot'):
            fig, ax= plt.subplots()
            sns.boxplot(data=df_new, orient='h')
            plt.title('Boxplot Chart')
            st.pyplot(fig)



    else:
        st.write('data belum ada')
    
