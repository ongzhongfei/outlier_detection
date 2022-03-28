import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pandas.io.json import json_normalize 
import statistics
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")

st.title('Outlier Detection')
st.subheader('Description')
st.info("""
An outlier may indicate bad data, but it does not necessarily indicate something bad has happened. 
Anomaly detection is the process of identifying unexpected data points in data sets, which differ from the norm. 
Traditional manual anomaly detection is humanely impossible. Thus, the aim of this web app is to 
share insights about the different methods of detecting outlier including machine learning techniques by other central banks.

We will be using GDP data and IPI data from apikijangportal.bnm.my.
""")

# To import data from apikijangportal.bnm.gov.my (MSB 3.3: Gross Domestic Product by Expenditure Components at Constant 2010 Prices (Annual Change))
@st.cache
def loaddata():
  df = pd.DataFrame()
  for y in range(2006,2022):
    response = requests.get('https://api.bnm.gov.my/public/msb/3.3/year/'+str(y),
                            headers = {'Accept': 'application/vnd.BNM.API.v1+json'},)
    data = json.loads(response.text)
    df = df.append(pd.json_normalize(data['data']))
  df.rename(columns = {'gro_dom_pro': 'gdp_growth'}, inplace=True)
  df = df[df.period.notnull()]
  df['date'] = df.year_dt.astype('str') +'Q' + df.period.str[0]
  df.sort_values(by = 'date', inplace = True)
  df['gdp_growth'] = df['gdp_growth'].astype(float)
  df = df[['year_dt', 'date', 'gdp_growth']]
  df['date'] = pd.PeriodIndex(df['date'], freq="Q")
  df.reset_index(drop = True, inplace = True)
  return df

df = loaddata()

# To import data from apikijangportal.bnm.gov.my (MSB 3.5.1: Industrial Production Index (Annual Change))
@st.cache
def loaddata2():
    df2 = pd.DataFrame()
    for y in range(2016,2022):
        response = requests.get('https://api.bnm.gov.my/public/msb/3.5.1.1/year/'+str(y),
                                headers = {'Accept': 'application/vnd.BNM.API.v1+json'},)
        data = json.loads(response.text)
        df2 = df2.append(pd.json_normalize(data['data']))
    df2['all_div'] = pd.to_numeric(df2['all_div'], errors='coerce')
    df2['period'] = (((df2['month_dt']-1)//3)+1)
    df2['date'] = df2.year_dt.astype('str') +'Q' + df2.period.astype('str')
    df2.sort_values(by = 'date', inplace = True)
    df2 = df2.astype({'all_div':'float','min' :'float' , 'ele':'float', 'man':'float', 
                    'exp_ind_tot':'float', 'exp_ind_ele_clu_tot':'float', 'exp_ind_ele_clu_ele':'float',
                    'exp_ind_ele_clu_ele_pro':'float', 'exp_ind_ele_clu_mac':'float', 'exp_ind_pri_clu_tot':'float',
                    'exp_ind_pri_clu_che':'float', 'exp_ind_pri_clu_pet':'float', 'exp_ind_pri_clu_tex':'float',
                    'exp_ind_pri_clu_woo':'float', 'exp_ind_pri_clu_rub':'float', 'exp_ind_pri_clu_off':'float',
                    'exp_ind_pri_clu_pap':'float', 'dom_ind_tot':'float', 'dom_ind_con_clu_tot':'float',
                    'dom_ind_con_clu_non':'float', 'dom_ind_con_clu_iro':'float', 'dom_ind_con_clu_fab':'float',
                    'dom_ind_csm_clu_tot':'float', 'dom_ind_csm_clu_foo':'float', 'dom_ind_csm_clu_tra':'float', 
                    'dom_ind_csm_clu_bev':'float', 'dom_ind_csm_clu_tob':'float', 'dom_ind_csm_clu_oth':'float' })
    return df2
df2 = loaddata2()

#To find the quarterly average IPI data based on the 3.5.1 data in month.

x = pd.Series(round((df2.groupby(['date'])['all_div'].mean()),1))
x2 = pd.Series(round((df2.groupby(['date'])['min'].mean()),4))
x3 = pd.Series(round((df2.groupby(['date'])['ele'].mean()),4))
x4 = pd.Series(round((df2.groupby(['date'])['man'].mean()),4))
x5 = pd.Series(round((df2.groupby(['date'])['exp_ind_tot'].mean()),1))
x6 = pd.Series(round((df2.groupby(['date'])['exp_ind_ele_clu_tot'].mean()),4))
x7 = pd.Series(round((df2.groupby(['date'])['exp_ind_ele_clu_ele'].mean()),4))
x8 = pd.Series(round((df2.groupby(['date'])['exp_ind_ele_clu_ele_pro'].mean()),4))
x9 = pd.Series(round((df2.groupby(['date'])['exp_ind_ele_clu_mac'].mean()),1))
x10 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_tot'].mean()),4))
x11 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_che'].mean()),4))
x12 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_pet'].mean()),4))
x13 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_tex'].mean()),1))
x14 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_woo'].mean()),4))
x15 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_rub'].mean()),4))
x16 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_off'].mean()),4))
x17 = pd.Series(round((df2.groupby(['date'])['exp_ind_pri_clu_pap'].mean()),1))
x18 = pd.Series(round((df2.groupby(['date'])['dom_ind_tot'].mean()),4))
x19 = pd.Series(round((df2.groupby(['date'])['dom_ind_con_clu_tot'].mean()),4))
x20 = pd.Series(round((df2.groupby(['date'])['dom_ind_con_clu_non'].mean()),4))
x21 = pd.Series(round((df2.groupby(['date'])['dom_ind_con_clu_iro'].mean()),1))
x22 = pd.Series(round((df2.groupby(['date'])['dom_ind_con_clu_fab'].mean()),4))
x23 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_tot'].mean()),4))
x24 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_foo'].mean()),4))
x25 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_tra'].mean()),1))
x26 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_bev'].mean()),4))
x27 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_tob'].mean()),4))
x28 = pd.Series(round((df2.groupby(['date'])['dom_ind_csm_clu_oth'].mean()),4))

xdate = list(set(df2['date']))

@st.cache
def loaddata3():
    dataset = pd.DataFrame(list(zip(xdate, df['gdp_growth'], x, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                                x21, x22, x23, x24, x25, x26, x27, x28)), 
                        columns = ['date', 'gdp_growth', 'IPI All', 'mining', 'electricity', 'manufacturing',
                                    'exp_ind_tot', 'exp_ind_ele_clu_tot', 'exp_ind_ele_clu_ele', 'exp_ind_ele_clu_ele_pro', 
                                    'exp_ind_ele_clu_mac', 'exp_ind_pri_clu_tot', 'exp_ind_pri_clu_che', 'exp_ind_pri_clu_pet', 
                                    'exp_ind_pri_clu_tex','exp_ind_pri_clu_woo', 'exp_ind_pri_clu_rub', 'exp_ind_pri_clu_off',
                                    'exp_ind_pri_clu_pap', 'dom_ind_tot', 'dom_ind_con_clu_tot', 'dom_ind_con_clu_non', 
                                    'dom_ind_con_clu_iro', 'dom_ind_con_clu_fab','dom_ind_csm_clu_tot', 'dom_ind_csm_clu_foo', 
                                    'dom_ind_csm_clu_tra', 'dom_ind_csm_clu_bev', 'dom_ind_csm_clu_tob', 'dom_ind_csm_clu_oth'])
    dataset['date'] = dataset['date'].sort_values(ascending=True).values
    dataset.drop(dataset.tail(1).index,inplace=True)
    qs = dataset['date'].str.replace(r'(Q\d) (\d+)', r'\2-\1')
    dataset['datetime'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()
    return dataset
dataset = loaddata3()


if st.checkbox('Show raw data'):
    st.subheader('GDP and IPI data')
    st.write(dataset)

# summary statistics
st.subheader('Summary of data')
st.table(dataset.describe())
with st.expander("See variables"):
     st.write("""
        - gdp_growth : Gross domestic product (GDP)
        - all_division : IPI All
        - mining : Mining
        - electricity : Electricity
        - manufacturing : Manufacturing
        - exp_ind_tot : Export-oriented industries - Total
        - exp_ind_ele_clu_tot : Export-oriented industries - Electronic and electrical cluster - Total
        - exp_ind_ele_clu_ele : Export-oriented industries - Electronic and electrical cluster - Electronics
        - exp_ind_ele_clu_ele_pro : Export-oriented industries - Electronic and electrical cluster - Electricals products
        - exp_ind_ele_clu_mac : Export-oriented industries - Electronic and electrical cluster - Machineries
        - exp_ind_pri_clu_tot : Export-oriented industries - Primary-related cluster - Total
        - exp_ind_pri_clu_che : Export-oriented industries - Primary-related cluster - Chemicals and chemical products
        - exp_ind_pri_clu_pet : Export-oriented industries - Primary-related cluster - Petroleum products
        - exp_ind_pri_clu_tex : Export-oriented industries - Primary-related cluster - Textiles wearing apparel and footwear
        - exp_ind_pri_clu_woo : Export-oriented industries - Primary-related cluster - Wood and wood products
        - exp_ind_pri_clu_rub : Export-oriented industries - Primary-related cluster - Rubber products
        - exp_ind_pri_clu_off : Export-oriented industries - Primary-related cluster - Off-estate processing
        - exp_ind_pri_clu_pap : Export-oriented industries - Primary-related cluster - Paper products
        - dom_ind_tot : Domestic-oriented industries - Total
        - dom_ind_con_clu_tot : Domestic-oriented industries - Construction-related cluster - Total
        - dom_ind_con_clu_non : Domestic-oriented industries - Construction-related cluster - Non-metallic mineral products
        - dom_ind_con_clu_iro : Domestic-oriented industries - Construction-related cluster - Iron and steel
        - dom_ind_con_clu_fab : Domestic-oriented industries - Construction-related cluster - Fabricated metal products
        - dom_ind_csm_clu_tot : Domestic-oriented industries - Consumer-related cluster - Total
        - dom_ind_csm_clu_foo : Domestic-oriented industries - Consumer-related cluster - Food products
        - dom_ind_csm_clu_tra : Domestic-oriented industries - Consumer-related cluster - Transport equipment
        - dom_ind_csm_clu_bev : Domestic-oriented industries - Consumer-related cluster - Beverages
        - dom_ind_csm_clu_tob : Domestic-oriented industries - Consumer-related cluster - Tobacco products
        - dom_ind_csm_clu_oth : Domestic-oriented industries - Consumer-related cluster - Others
     """)

st.subheader('1. Traditional method - Measures of Variability')
"""
- Standard deviation
- Median absolute deviation
- Interquartile range
"""
option = st.selectbox(
     'Choose 1 variable.',
     ('gdp_growth', 'IPI All', 'mining', 'electricity', 'manufacturing',
        'exp_ind_tot', 'exp_ind_ele_clu_tot', 'exp_ind_ele_clu_ele', 'exp_ind_ele_clu_ele_pro', 
        'exp_ind_ele_clu_mac', 'exp_ind_pri_clu_tot', 'exp_ind_pri_clu_che', 'exp_ind_pri_clu_pet', 
        'exp_ind_pri_clu_tex','exp_ind_pri_clu_woo', 'exp_ind_pri_clu_rub', 'exp_ind_pri_clu_off',
        'exp_ind_pri_clu_pap', 'dom_ind_tot', 'dom_ind_con_clu_tot', 'dom_ind_con_clu_non', 
        'dom_ind_con_clu_iro', 'dom_ind_con_clu_fab','dom_ind_csm_clu_tot', 'dom_ind_csm_clu_foo', 
        'dom_ind_csm_clu_tra', 'dom_ind_csm_clu_bev', 'dom_ind_csm_clu_tob', 'dom_ind_csm_clu_oth'))
     
meddf = round((statistics.median(dataset[option])),4)
meandf = round((statistics.mean(dataset[option])),4)
stddf = round((statistics.stdev(dataset[option])),4)
maddf = round((stats.median_absolute_deviation(dataset[option])),4)
iqrdf = round((stats.iqr(dataset[option], rng=(25, 75))),4)

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
"""
option2 = st.selectbox(
     'Choose 1 variable:',
     ('gdp_growth', 'IPI All', 'mining', 'electricity', 'manufacturing',
        'exp_ind_tot', 'exp_ind_ele_clu_tot', 'exp_ind_ele_clu_ele', 'exp_ind_ele_clu_ele_pro', 
        'exp_ind_ele_clu_mac', 'exp_ind_pri_clu_tot', 'exp_ind_pri_clu_che', 'exp_ind_pri_clu_pet', 
        'exp_ind_pri_clu_tex','exp_ind_pri_clu_woo', 'exp_ind_pri_clu_rub', 'exp_ind_pri_clu_off',
        'exp_ind_pri_clu_pap', 'dom_ind_tot', 'dom_ind_con_clu_tot', 'dom_ind_con_clu_non', 
        'dom_ind_con_clu_iro', 'dom_ind_con_clu_fab','dom_ind_csm_clu_tot', 'dom_ind_csm_clu_foo', 
        'dom_ind_csm_clu_tra', 'dom_ind_csm_clu_bev', 'dom_ind_csm_clu_tob', 'dom_ind_csm_clu_oth'))

graph1 = px.box(dataset, x=option2)


iqr = stats.iqr(dataset[option2], rng=(25, 75))
min_fence = np.percentile(dataset[option2], 25) - (1.5*iqr)
max_fence = np.percentile(dataset[option2], 75) + (1.5*iqr)

# graph1 = px.scatter(x=dataset['date'] , y=dataset[option2])
graph2 = go.Figure(data=go.Scatter(x=dataset['datetime'], y=dataset[option2], mode='lines+markers'))
graph2.add_shape(
        type='line',
        x0=dataset['datetime'].min(),
        y0=min_fence,
        x1=dataset['datetime'].max(),
        y1=min_fence,
        line=dict(
            color='Red',
        ),
        opacity =0.3
)
graph2.add_shape(
        type='line',
        x0=dataset['datetime'].min(),
        y0=max_fence,
        x1=dataset['datetime'].max(),
        y1=max_fence,
        line=dict(
            color='Red',
        ),
        opacity =0.3
)
graph2.add_annotation(x=dataset['datetime'].max(), y=max_fence,
            text=str(round(max_fence,1)),
            font = {'color':'red'},
            showarrow=False,yshift=10)
graph2.add_annotation(x=dataset['datetime'].max(), y=min_fence,
            text=str(round(min_fence,1)),
            font = {'color':'red'},
            showarrow=False,yshift=10)
graph2.update_layout(
        xaxis_title= "Date",
        yaxis_title= option2,
    ) 


col1, col2 = st.columns(2)

with col1:
    st.subheader("Scatter plot")
    st.plotly_chart(graph1 , use_container_width=True)
with col2:
    st.subheader("Box plot")
    st.plotly_chart(graph2 , use_container_width=True)


st.subheader('3. Machine learning techniques')
st.info("""
Machine learning algorithms have the ability to learn from data and make predictions based on that data. 
Machine learning for anomaly detection includes techniques that provide a promising alternative for detection 
and classification of anomalies based on an initially large set of features. Using the GDP data we will 
focus on using 2 machine learning techniques.
""")
# Separate the df from variable (datatype date format)

with st.form("Outlier variables selection: "):
    option3 = st.multiselect(
        'Choose variables detect outlier: (Please select at least 2 variables)',
        ['gdp_growth', 'IPI All', 'mining', 'electricity', 'manufacturing',
            'exp_ind_tot', 'exp_ind_ele_clu_tot', 'exp_ind_ele_clu_ele', 'exp_ind_ele_clu_ele_pro', 
            'exp_ind_ele_clu_mac', 'exp_ind_pri_clu_tot', 'exp_ind_pri_clu_che', 'exp_ind_pri_clu_pet', 
            'exp_ind_pri_clu_tex','exp_ind_pri_clu_woo', 'exp_ind_pri_clu_rub', 'exp_ind_pri_clu_off',
            'exp_ind_pri_clu_pap', 'dom_ind_tot', 'dom_ind_con_clu_tot', 'dom_ind_con_clu_non', 
            'dom_ind_con_clu_iro', 'dom_ind_con_clu_fab','dom_ind_csm_clu_tot', 'dom_ind_csm_clu_foo', 
            'dom_ind_csm_clu_tra', 'dom_ind_csm_clu_bev', 'dom_ind_csm_clu_tob', 'dom_ind_csm_clu_oth'],
        default = ['gdp_growth' , 'IPI All'])
    submitted = st.form_submit_button("Submit")
st.subheader('3.1 Density-based spatial clustering of applications with noise (DBSCAN)​')
st.info("""
DBSCAN is able to find arbitrary shaped clusters and clusters with noise (i.e. outliers).
The main idea behind DBSCAN is that a point belongs to a cluster if it is close to many points from that cluster.

There are two key parameters of DBSCAN:
- eps     : The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
- minPts  : Minimum number of data points to define a cluster.
""")

st.info("""
Pros and Cons of DBSCAN

Pros:
- Does not require to specify number of clusters beforehand.
- Performs well with arbitrary shapes clusters.
- DBSCAN is robust to outliers and able to detect the outliers.

Cons:
- Determining an appropriate distance of neighborhood (eps) is not easy and it requires domain knowledge.
""")

col4, col5= st.columns(2)

with col4:
    eps1 = st.slider('Choose the eps value:', min_value = 1, max_value= 30, value= 3, step=1 )
with col5:
    minsample1 = st.slider('Choose the min sample:', min_value = 3, max_value= 10, value= 3 )




#df_model = dataset.drop(['date'], axis=1)
df_dbscan = dataset.copy()
df_if = dataset.copy()

model = DBSCAN(eps = eps1, min_samples =minsample1).fit(dataset[option3])
colors = model.labels_
df_dbscan["dbscan_outliers"] = model.fit_predict(dataset[option3])
df_dbscan['anomaly']= df_dbscan['dbscan_outliers'].apply(lambda x: 'outlier' if x==-1  else 'normal')
# graph4 = px.scatter(dataset['gdp_growth'], dataset['all_division'], color =df_dbscan['anomaly'])
if len(option3) == 2:
    graph4 = px.scatter(dataset, x=option3[0], y = option3[1], color =df_dbscan['anomaly'])
    graph4.update_traces(marker={'size': 15})
else:
    graph4 = px.scatter_3d(dataset,x=option3[0],
                        y=option3[1],
                        z=option3[2],
                        color=df_dbscan['anomaly'])


st.plotly_chart(graph4 , use_container_width=True)

st.write('Below are the identified outliers:')
st.write(df_dbscan[df_dbscan["dbscan_outliers"]==-1])

st.subheader('3.2 Isolation forest')
st.info("""
Isolation forests are built using decision trees. 

The isolation forest needs an anomaly score to have an idea of how anomalous a data point is. 
Its values lie between 0 and 1. The anomaly score is defined as:

- A score close to 1 indicates anomalies
- Score much smaller than 0.5 indicates normal observations
- If all scores are close to 0.5 then the entire sample does not seem to have clearly distinct anomalies 

Note: scikit-learn’s isolation forest introduced a modification of the anomaly scores. 
The outliers are indicated by negative scores, while the positive score implies that we have normal observation.
""")

model2=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2)).fit(dataset[option3])
df_if['scores'] = model2.decision_function(dataset[option3])
df_if['if_outliers'] = model2.predict(dataset[option3])
df_if['anomaly']=df_if['if_outliers'].apply(lambda x: 'outlier' if x==-1  else 'normal')

if len(option3) == 2:
    graph5 = px.scatter(df_if, x=option3[0], y = option3[1], color =df_dbscan['anomaly'])
    graph5.update_traces(marker={'size': 15})

else:
    graph5 = px.scatter_3d(df_if,x=option3[0],
                        y=option3[1],
                        z=option3[2],
                        color='anomaly')

st.plotly_chart(graph5 , use_container_width=True)
st.write('Below are the identified outliers:')
st.write(df_if[df_if['if_outliers']==-1])