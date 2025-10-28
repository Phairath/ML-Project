import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import joblib
import gdown,os

st.set_page_config(
    page_title = 'Machine Learning',
    page_icon = '🤖',
    initial_sidebar_state = 'auto',
    layout = 'centered' 
)

st.title('🛒 Favorita Grocery Sales Forecasting')

@st.cache_data(persist=True)
def load_model():
    try:
        file_id = '1D5wXF0ISvuwkIWtfv-IIPhJ3_rXCMQR3'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = 'xgb_best_model.joblib'
        gdown.download(url, output, quiet=False)
        model = joblib.load(./output)
        # return model
    except Exception:
        st.markdown('<span style="color: red; font-size: 20px;">\***An error occurred while loading the model**</span> ', unsafe_allow_html=True)
    return model

st.markdown('#### **Input your data for predict**')
cols = st.columns(3)
with cols[0]:
    # st.write('General Information')
    selected_date = st.date_input('Date',value=date.today(),min_value=date(2017, 1, 1),max_value=date(2030, 12, 31))
    dcoilwtico = st.number_input('Crude Oil Price (WTI)',min_value=0.0,value=0.0,format="%.2f")
    is_holiday = st.selectbox('Holiday',('Yes','No'))

city_name = ['Quito', 'Santo Domingo', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra', 'Guaranda', 
             'Puyo', 'Ambato', 'Guayaquil', 'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 
             'Libertad', 'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen']
with cols[1]:
    # st.write('Store Information')
    store_nbr = st.selectbox('Store Number',(range(1,55))) #st.number_input('Store Number',min_value=1,max_value=54,value=1)
    city = st.selectbox('City',(city_name))
    onpromotion = st.selectbox('Promotion',('Yes','No'))

family_name = ['GROCERY I', 'CLEANING', 'BREAD/BAKERY', 'DELI', 'POULTRY', 'EGGS', 'PERSONAL CARE', 
               'LINGERIE', 'BEVERAGES', 'AUTOMOTIVE', 'DAIRY', 'GROCERY II', 'MEATS', 'FROZEN FOODS', 
               'HOME APPLIANCES', 'SEAFOOD', 'PREPARED FOODS', 'LIQUOR,WINE,BEER', 'BEAUTY', 'HARDWARE', 
               'LAWN AND GARDEN', 'PRODUCE', 'HOME AND KITCHEN II', 'HOME AND KITCHEN I', 'MAGAZINES', 
               'HOME CARE', 'PET SUPPLIES', 'BABY CARE', 'SCHOOL AND OFFICE SUPPLIES', 'PLAYERS AND ELECTRONICS', 
               'CELEBRATION', 'LADIESWEAR', 'BOOKS']

with cols[2]:
    # st.write('Item Information')
    item_nbr = st.number_input('Item Number',min_value=1000866,value=1000866)
    family = st.selectbox('Item Type',(family_name))
    perishable = st.selectbox('Perishable',('Yes','No'))
    

btn1 = st.button('Predict!')
model = load_model()
if btn1:
    onpromotion = 1 if onpromotion == 'Yes' else 0
    perishable = 1 if perishable == 'Yes' else 0
    is_holiday = 1 if is_holiday == 'Yes' else 0 
    day_of_week = int(selected_date.weekday())
    day = int(selected_date.strftime('%d'))
    month = int(selected_date.strftime('%m'))
    year = int(selected_date.strftime('%Y'))
    is_weekend = int(day_of_week > 4)
    x_new = {'store_nbr':[store_nbr],
            'item_nbr': [item_nbr],
            'onpromotion':[onpromotion],
            'dcoilwtico':[dcoilwtico],
            'city':[city],
            'family':[family],
            'perishable':[perishable],
            'is_holiday':[int(is_holiday)],
            'day_of_week':[day_of_week],
            'day':[day],
            'month':[month],
            'year':[year],
            'is_weekend':[is_weekend]}
    x_new = pd.DataFrame(x_new)		
    x_new[['store_nbr','item_nbr','city','family']] = x_new[['store_nbr','item_nbr','city','family']].astype('category')
    x_new[['perishable','is_holiday','day_of_week','day','month','is_weekend','onpromotion']] = x_new[['perishable','is_holiday','day_of_week','day','month','is_weekend','onpromotion']].astype('int8')
    x_new['year'] = x_new['year'].astype('int16')
    x_new['dcoilwtico'] = x_new['dcoilwtico'].astype('float32')
    prediction = model.predict(x_new)
    prediction = np.maximum(0,prediction)
    st.markdown(
                f"""<p style="font-size: 26px; font-weight: bold;">
                <span style="color: black;">Predicted Unit Sales: </span>
                <span style="color: green;">{prediction[0]:.5f} pieces/kg</span></p>""" 

                , unsafe_allow_html=True)


