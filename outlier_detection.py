import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pandas.io.json import json_normalize 
import statistics
from scipy import stats
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")

st.title('Outlier Detection')
st.subheader('Description')
st.info("""
An outlier may indicate bad data, but it does not necessarily indicate something bad has happened. 
Anomaly detection is the process of identifying unexpected data points in data sets, which differ from the norm. 
Traditional manual anomaly detection is humanely impossible. Thus, the aim of this web app is to 
share insights about the different methods of detecting outlier in time series including 
machine learning techniques by other central banks.

We will be using GDP data and IPI data from apikijangportal.bnm.my.
""")

# To import data from apikijangportal.bnm.gov.my (MSB 3.3: Gross Domestic Product by Expenditure Components at Constant 2010 Prices (Annual Change))

df = pd.DataFrame()
for y in range(2006,2022):
    response = requests.get('https://api.bnm.gov.my/public/msb/3.3/year/'+str(y),
                            headers = {'Accept': 'application/vnd.BNM.API.v1+json'},)
    data = json.loads(response.text)
    df = df.append(pd.json_normalize(data['data']))
df.rename(columns = {'gro_dom_pro': 'gdp_growth'}, inplace=True)
df = df[df.period.notnull()]
df['date'] = df.year_dt.astype('str') +'Q' + df.period.str[0]
#df = df.drop(['period'] , axis=1)
df.sort_values(by = 'date', inplace = True)
df.rename(columns={'fin_com_exp_tot': 'total_consumption_expenditure',
                    'fin_com_exp_pri_sec': 'total_consumption_expenditure_private', 
                    'fin_com_exp_pub_sec': 'total_consumption_expenditure_public', 
                    'gro_fix_cap_inf_tot': 'total_gross_fixed_capital_formation',
                    'gro_fix_cap_inf_pri_sec': 'total_gross_fixed_capital_formation_private', 
                    'gro_fix_cap_inf_pub_sec': 'total_gross_fixed_capital_formation_public', 
                    'exp_of_goo_and_ser': 'exports_of_goods_and_services', 
                    'imp_of_goo_and_ser': 'imports_of_goods_and_services'}, 
                   inplace=True)
df = df.astype({'gdp_growth':'float','total_consumption_expenditure' :'float' , 
                'total_consumption_expenditure_private':'float',
     'total_consumption_expenditure_public':'float', 'total_gross_fixed_capital_formation':'float', 
     'total_gross_fixed_capital_formation_private':'float', 'total_gross_fixed_capital_formation_public':'float',
     'exports_of_goods_and_services':'float', 'imports_of_goods_and_services':'float'})

# To import data from apikijangportal.bnm.gov.my (MSB 3.5.1: Industrial Production Index (Annual Change))

df2 = pd.DataFrame()
for y in range(2016,2022):
    response = requests.get('https://api.bnm.gov.my/public/msb/3.5.1/year/'+str(y),
                            headers = {'Accept': 'application/vnd.BNM.API.v1+json'},)
    data = json.loads(response.text)
    df2 = df2.append(pd.json_normalize(data['data']))
    
df2['all_div'] = pd.to_numeric(df2['all_div'], errors='coerce')
df2['period'] = (((df2['month_dt']-1)//3)+1)
df2['date'] = df2.year_dt.astype('str') +'Q' + df2.period.astype('str')
df2.sort_values(by = 'date', inplace = True)

#To find the quarterly average IPI based on the 3.5.1 data in month.

x = pd.Series(round((df2.groupby(['date'])['all_div'].mean()),1))
xdate = list(set(df2['date']))
ipi_df = pd.DataFrame(list(zip(xdate, x)), columns = ['date', 'all_division'])
ipi_df['date'] = ipi_df['date'].sort_values(ascending=True).values
ipi_df.drop(ipi_df.tail(1).index,inplace=True)

if st.checkbox('Show raw data'):
    st.subheader('GDP data')
    st.write(df.head(n=5))
    st.subheader('IPI data')
    st.write(ipi_df.head(n=5))

# summary statistics for each data
st.subheader('Summary of GDP data')
st.table(df.describe())

st.subheader('1. Traditional method - Measures of Variability')
"""
- Standard deviation
- Median absolute deviation
- Interquartile range
"""
option = st.selectbox(
     'Choose 1 variable.',
     ('gdp_growth', 'total_consumption_expenditure','total_consumption_expenditure_private',
     'total_consumption_expenditure_public', 'total_gross_fixed_capital_formation', 
     'total_gross_fixed_capital_formation_private', 'total_gross_fixed_capital_formation_public',
     'exports_of_goods_and_services', 'imports_of_goods_and_services'
    ))
     
meddf = round((statistics.median(df[option])),4)
meandf = round((statistics.mean(df[option])),4)
stddf = round((statistics.stdev(df[option])),4)
maddf = round((stats.median_absolute_deviation(df[option])),4)
iqrdf = round((stats.iqr(df[option], rng=(25, 75))),4)

st.write('The median value                     : ', meddf)
st.write('The mean value                       : ', meandf)
st.write('The standard deviation value         : ', stddf)
st.write('The median absolute deviation value  : ', maddf)
st.write('The interquartile range value        : ', iqrdf)

st.subheader('2. Graphing your data to identify outliers')
st.info("""
The simplest way to detect an outlier is by graphing the features or the data points. 
Visualization is one of the best and easiest ways to have an inference about the overall data and the outliers.

Scatter plots and box plots are the most preferred visualization tools to detect outliers.
""")

"""
- Scatter plot
- Box plot
- Histogram
"""
option2 = st.selectbox(
     'Choose 1 variable:',
     ('gdp_growth', 'total_consumption_expenditure','total_consumption_expenditure_private',
     'total_consumption_expenditure_public', 'total_gross_fixed_capital_formation', 
     'total_gross_fixed_capital_formation_private', 'total_gross_fixed_capital_formation_public',
     'exports_of_goods_and_services', 'imports_of_goods_and_services'
    ))

graph1 = px.scatter(x=df['date'] , y=df[option2])
graph1.update_layout(
        xaxis_title= "Date",
    ) 

graph2 = px.box(x=df[option2])

graph3 = px.histogram(x=df['date'] , y=df[option2])
graph3.update_layout(
        xaxis_title= "Date",
    ) 

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Scatter plot")
    st.plotly_chart(graph1 , use_container_width=True)
with col2:
    st.subheader("Box plot")
    st.plotly_chart(graph2 , use_container_width=True)
with col3:
    st.subheader("Histogram")
    st.plotly_chart(graph3 , use_container_width=True)


st.subheader('3. Machine learning techniques')
st.info("""
Machine learning algorithms have the ability to learn from data and make predictions based on that data. 
Machine learning for anomaly detection includes techniques that provide a promising alternative for detection 
and classification of anomalies based on an initially large set of features. Using the GDP data we will 
focus on using 2 machine learning techniques.
""")

df_dbscan = df.copy()
df_if = df.copy()

st.subheader('- Density-based spatial clustering of applications with noise (DBSCAN)​')

col4, col5= st.columns(2)

with col4:
    eps1 = st.slider('Choose the eps value:', min_value = 5, max_value= 50, value= 5, step=1 )
with col5:
    minsample1 = st.slider('Choose the min sample:', min_value = 3, max_value= 10, value= 3 )


# Separate the df from variable (datatype date format)
df_model = df.drop(['year_dt','period', 'date'], axis=1)

model = DBSCAN(eps = eps1, min_samples =minsample1).fit(df_model)
colors = model.labels_
graph4 = px.scatter(df['total_consumption_expenditure'], df['gdp_growth'], color =colors)

st.plotly_chart(graph4 , use_container_width=True)

df_dbscan["dbscan_outliers"] = model.fit_predict(df_model)

st.write(df_dbscan[df_dbscan["dbscan_outliers"]==-1])

st.subheader('- Isolation forest')

model2=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2)).fit(df_model)
df_if['scores'] = model2.decision_function(df_model)
df_if['if_outliers'] = model2.predict(df_model)
df_if['anomaly']=df_if['if_outliers'].apply(lambda x: 'outlier' if x==-1  else 'inlier')

graph5 = px.scatter_3d(df_if,x='total_gross_fixed_capital_formation',
                       y='total_consumption_expenditure',
                       z='gdp_growth',
                       color='anomaly')

st.plotly_chart(graph5 , use_container_width=True)

st.write(df_if[df_if['if_outliers']==-1])


#If we filter the dataset by anomaly_label equal to -1, we can observe that all the scores are negative near zero. 
#In the opposite case, with the anomaly label equal to 1, we found all the positive scores.
