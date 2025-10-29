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

file_id = '1D5wXF0ISvuwkIWtfv-IIPhJ3_rXCMQR3'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'xgb_best_model.joblib'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

@st.cache_data(persist=True)
def load_model():
    try:
        model = joblib.load(output)
    except Exception:
        st.markdown('<span style="color: red; font-size: 20px;">\***An error occurred while loading the model**</span> ', unsafe_allow_html=True)
    return model

# st.markdown('#### **Input your data for predict**')
cols = st.columns(3)
with cols[0]:
    # st.write('General Information')
    selected_date = st.date_input('Date',value=date(2017, 8, 16),min_value=date(2017, 1, 1),max_value=date(2030, 12, 31))
    dcoilwtico = st.number_input('Crude Oil Price (WTI)',min_value=0.00,value=0.00,format="%.2f")
    is_holiday = st.selectbox('Holiday',('Yes','No'))

CITY_NAME = ['Ambato', 'Babahoyo', 'Cayambe', 'Cuenca', 'Daule', 'El Carmen',
             'Esmeraldas', 'Guaranda', 'Guayaquil', 'Ibarra', 'Latacunga',
             'Libertad', 'Loja', 'Machala', 'Manta', 'Playas', 'Puyo', 'Quevedo',
             'Quito', 'Riobamba', 'Salinas', 'Santo Domingo']

STORE_NAME = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30',
              '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41',
              '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52',
              '53', '54', '6', '7', '8', '9']

with cols[1]:
    # st.write('Store Information')
    store_nbr = st.selectbox('Store Number',sorted(STORE_NAME, key=lambda x: int(x))) #st.number_input('Store Number',min_value=1,max_value=54,value=1)
    city = st.selectbox('City',(CITY_NAME))
    onpromotion = st.selectbox('Promotion',('Yes','No'))

FAMILY_NAME = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
               'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
               'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
               'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
               'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
               'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
               'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS',
               'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD']

items = pd.read_parquet('./data/items_categories.parquet')
ITEM_NAME = items['item_nbr'].cat.categories

with cols[2]:
    # st.write('Item Information')
    item_nbr = st.selectbox('Item Number',ITEM_NAME)
    family = st.selectbox('Item Type',(FAMILY_NAME))
    perishable = st.selectbox('Perishable',('Yes','No'))


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
x_new['store_nbr'] = pd.Categorical(x_new['store_nbr'],categories=STORE_NAME)
x_new['city'] = pd.Categorical(x_new['city'],categories=CITY_NAME)
x_new['family'] = pd.Categorical(x_new['family'],categories=FAMILY_NAME)
x_new['item_nbr'] = pd.Categorical(x_new['item_nbr'],categories=items['item_nbr'].cat.categories)
st.write(x_new)

btn1 = st.button('Predict!')
if btn1:
    model = load_model()
    prediction = model.predict(x_new)
    prediction = np.maximum(0,prediction)
    st.markdown(
                f"""<p style="font-size: 26px; font-weight: bold;">
                <span style="color: black;">Predicted Unit Sales: </span>
                <span style="color: green;">{prediction[0]:.5f} pieces/kg</span></p>""" 

                , unsafe_allow_html=True)
