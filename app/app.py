import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='Churn Analytics', layout='wide')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'src', 'lgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'src', 'feature_names.json'), 'r') as f:
        features = json.load(f)
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_model.csv'))
    return df

model, feature_names = load_model()
df = load_data()

# Sidebar
st.sidebar.title('Churn Analytics')
page = st.sidebar.radio('Розділ', ['Огляд', 'Сегменти', 'Прогноз'])

# ── Сторінка 1: Огляд ──
if page == 'Огляд':
    st.title('Огляд продуктових метрик')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Всього юзерів', len(df))
    col2.metric('Churned', df['Churn'].sum())
    col3.metric('Churn rate', f"{df['Churn'].mean()*100:.1f}%")
    col4.metric('Avg Tenure', f"{df['Tenure'].mean():.1f} міс")

    st.subheader('Churn по tenure')
    fig1 = px.histogram(df, x='Tenure', color='Churn',
                        barmode='overlay', nbins=30,
                        color_discrete_map={0: '#1D9E75', 1: '#D85A30'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('Середні метрики: retained vs churned')
    means = df.groupby('Churn')[['Tenure', 'CashbackAmount',
                                  'SatisfactionScore', 'Complain']].mean().round(2)
    st.dataframe(means)

# ── Сторінка 2: Сегменти ──
elif page == 'Сегменти':
    st.title('Сегментація за ризиком churnu')

    X = df.drop(columns=['Churn'])
    df['ChurnProba'] = model.predict_proba(X)[:, 1]

    df['Segment'] = pd.cut(df['ChurnProba'],
                           bins=[0, 0.3, 0.6, 1.0],
                           labels=['Low risk', 'Medium risk', 'High risk'])

    fig2 = px.histogram(df, x='Segment', color='Segment',
                        color_discrete_map={
                            'Low risk': '#1D9E75',
                            'Medium risk': '#EF9F27',
                            'High risk': '#D85A30'
                        })
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Характеристики сегментів')
    seg_stats = df.groupby('Segment')[['Tenure', 'CashbackAmount',
                                        'SatisfactionScore', 'Complain']].mean().round(2)
    st.dataframe(seg_stats)

# ── Сторінка 3: Прогноз ──
elif page == 'Прогноз':
    st.title('Прогноз churnu для нового юзера')

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (місяців)', 0, 60, 6)
        satisfaction = st.slider('Satisfaction Score', 1, 5, 3)
        cashback = st.slider('Cashback Amount', 0, 400, 150)
        complain = st.selectbox('Скарга', [0, 1])
    with col2:
        warehouse = st.slider('Warehouse to Home (км)', 5, 100, 20)
        hours_app = st.slider('Hours on App', 0, 10, 3)
        order_count = st.slider('Order Count', 1, 20, 3)
        days_last = st.slider('Days Since Last Order', 0, 30, 5)

    if st.button('Розрахувати ризик'):
        input_data = pd.DataFrame(
            np.zeros((1, len(feature_names))), columns=feature_names
        )
        input_data['Tenure'] = tenure
        input_data['SatisfactionScore'] = satisfaction
        input_data['CashbackAmount'] = cashback
        input_data['Complain'] = complain
        input_data['WarehouseToHome'] = warehouse
        input_data['HourSpendOnApp'] = hours_app
        input_data['OrderCount'] = order_count
        input_data['DaySinceLastOrder'] = days_last

        proba = model.predict_proba(input_data)[0][1]

        if proba < 0.3:
            st.success(f'Низький ризик churnu: {proba*100:.1f}%')
        elif proba < 0.6:
            st.warning(f'Середній ризик churnu: {proba*100:.1f}%')
        else:
            st.error(f'Високий ризик churnu: {proba*100:.1f}%')