import streamlit as st
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize 
import statistics
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pd_profile_viz_app import pd_profile_app

from read_api_data import *
from sweet_viz_app import *
from pd_profile_viz_app import *

st.set_page_config(page_title="Outlier Detection with MHS",
                    page_icon=(":negative_squared_cross_mark:"),
                    layout="wide"
                    )

with st.sidebar:
    st.subheader("Use demo data or upload your own data:")
    data_mode = st.selectbox("",["Use demo data","Upload your own data"])

    if data_mode == "Upload your own data":
        with st.form("upload custom data form."):
            #To upload data in CSV format
            uploaded_file = st.file_uploader("Upload data in CSV format", type=[".csv",".xlsx"])
            st.form_submit_button("Submit")
        if uploaded_file:
            st.write(uploaded_file.name)
            if uploaded_file.name.endswith(".xlsx"):
                df_upload = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswtih(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            # st.markdown("### Data preview")
            # st.dataframe(df_upload.head())


st.title('Outlier Detection')
st.subheader('Description')
st.markdown("""
An outlier may indicate bad data, but it does not necessarily indicate something bad has happened. 
Anomaly detection is the process of identifying unexpected data points in data sets, which differ from the norm. 
Traditional manual anomaly detection is humanely impossible. Thus, the aim of this web app is to 
share insights about the different methods of detecting outlier including machine learning techniques by other central banks.
We will be using GDP data and IPI data from apikijangportal.bnm.my.
""")

if data_mode == "Use demo data":
    #### Function in read_api_data.py
    dataset = loaddata()
elif data_mode == "Upload your own data":
    dataset = df_upload.copy()

if st.checkbox('Show raw data'):
    st.subheader('GDP and IPI data')
    st.write(dataset)
    # summary statistics
    st.subheader('Summary of data')
    st.write(dataset.describe())

    with st.expander("See variables"):
        for k, v in data_name_dict.items():
            st.write("- ",k, " : ",v)
     
st.write("***")

#### Sweet Visualisation
source_code = sweet_app(input_df=dataset, data_mode=data_mode)

components.html(source_code, height=550, scrolling=True)


#### Pandas profiling Takes too long to load
# pd_profile_app(input_df=dataset)



st.subheader('1. Traditional method - Measures of Variability')
"""
- Standard deviation (SD)
- Median absolute deviation (MAD)
- Interquartile range (IQR)
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


iqr = stats.iqr(dataset[pd.notna(dataset[option2])][option2], rng=(25, 75))
min_fence = np.percentile(dataset[pd.notna(dataset[option2])][option2], 25) - (1.5*iqr)
max_fence = np.percentile(dataset[pd.notna(dataset[option2])][option2], 75) + (1.5*iqr)

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
    st.subheader("Box plot")
    st.plotly_chart(graph1 , use_container_width=True)
with col2:
    st.subheader(data_name_dict[option2] + " trend")
    st.plotly_chart(graph2 , use_container_width=True)

st.subheader('Find Potential Outliers using SD, MAD & IQR')
st.write('Define the distance factor α. Setting α=3 for Mean/SD method covers 99% of the distribution assuming the data is normally distributed.')
st.write('Note that alpha is irrelevant when choosing IQR.')

graph3 = px.histogram(dataset, x=dataset['datetime'], y=dataset[option2])

col3, col4 = st.columns([1,3])

with col3:
    option = st.selectbox(
     'Data Transformation',
     ('Log', 'Box-cox'))
    numalpha = st.number_input('Choose Alpha', min_value=1, step=1)
    method = st.radio(
     "Choose Measure of Variability",
     ('Mean/SD', 'Median/MAD', 'IQR'))

#if method == 'Mean/SD':
     #st.write('You selected comedy.')
 #else:
     #st.write("You didn't select comedy.")

with col4:
    st.write('Histogram with Outlier Threshold')
    st.plotly_chart(graph3 , use_container_width=True)

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
        'Choose variables detect outlier. Please select at least 2 variables. (For DBSCAN, the first variable will be used as the target variable, and the others will be used as the regressors.)',
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

st.write("Performing linear regression on the following:")
st.write("Target variable: "+ data_name_dict[option3[0]])
selected_regressor_str = data_name_dict[option3[1]]
for reg in option3[2:]:
    selected_regressor_str += ', ' + data_name_dict[reg]
st.write("Regressors: "+ selected_regressor_str)

#df_model = dataset.drop(['date'], axis=1)
df_dbscan = dataset.copy()
df_if = dataset.copy()
#### Linear Regression
Y = df_dbscan[option3[0]]
X = df_dbscan[option3[1:]]
# st.write(Y)
# st.write(X)
regressor = LinearRegression()  
regressor.fit(X, Y) 
df_dbscan['y_pred'] = regressor.predict(X)
#### Residual from prediction
df_dbscan['pred_residual'] = Y - df_dbscan['y_pred']
#### Plot prediction chart
pred_chart = go.Figure(data=go.Scatter(x=df_dbscan['datetime'], y=df_dbscan[option3[0]], mode='lines+markers', name='actual'))
pred_chart.add_trace(go.Scatter(x=df_dbscan['datetime'], y=df_dbscan['y_pred'],
                    mode='lines+markers',
                    name='predicted'))
pred_chart.add_trace(go.Scatter(x=df_dbscan['datetime'], y=df_dbscan['pred_residual'],
                    mode='lines',
                    name='prediction residual', line_color='black', opacity=0.3))
pred_chart.update_layout(hovermode="x")
pred_chart.update_layout(
        xaxis_title= "Date",
        yaxis_title= data_name_dict[option3[0]],
    ) 
with st.expander("Show prediction results: "):
    st.plotly_chart(pred_chart , use_container_width=True)

#### DBSCAN variables
col4, col5= st.columns(2)

with col4:
    eps1 = st.slider('Choose the eps value:', min_value = 1.0, max_value= 10.0, value= 1.5, step=0.5 )
with col5:
    minsample1 = st.slider('Choose the min sample:', min_value = 2, max_value= 10, value= 3 )

st.caption("Selected eps value: " + str(eps1))
st.caption("Selected min sample: " + str(minsample1))



#### Standardise Y and pred_res
df_dbscan[data_name_dict[option3[0]]+" (normalised)"] = (df_dbscan[option3[0]] - df_dbscan[option3[0]].mean()) / df_dbscan[option3[0]].std()
df_dbscan["Prediction Residual (normalised)"] = (df_dbscan['pred_residual'] - df_dbscan['pred_residual'].mean()) / df_dbscan['pred_residual'].std()

#### DBCSAN on residual
# model = DBSCAN(eps = eps1, min_samples =minsample1).fit(dataset[option3])
model = DBSCAN(eps = eps1, min_samples =minsample1).fit(df_dbscan[[data_name_dict[option3[0]]+" (normalised)", "Prediction Residual (normalised)"]])
colors = model.labels_
df_dbscan["dbscan_outliers"] = model.fit_predict(df_dbscan[[data_name_dict[option3[0]]+" (normalised)", "Prediction Residual (normalised)"]])


# df_dbscan['anomaly']= df_dbscan['dbscan_outliers'].apply(lambda x: 'outlier' if x==-1  else 'normal')
df_dbscan['dbscan_outliers'] = df_dbscan['dbscan_outliers'].replace(-1,"Noise")
df_dbscan['dbscan_outliers'] = df_dbscan['dbscan_outliers'].replace(0,"Core")
df_dbscan['dbscan_outliers'] = df_dbscan['dbscan_outliers'].replace(1,"Border")
# graph4 = px.scatter(dataset['gdp_growth'], dataset['all_division'], color =df_dbscan['anomaly'])
color_discrete_map = {'Core':'blue', 'Border':'green', 'Noise':'red'}

#### Residual scatter plot
res_scatter = px.scatter(df_dbscan, x=data_name_dict[option3[0]]+" (normalised)", y = "Prediction Residual (normalised)",
                        hover_data=["date", option3[0]],
                        color =df_dbscan['dbscan_outliers'],
                        color_discrete_map=color_discrete_map,
                        )
res_scatter.update_traces(marker={'size': 12})
res_scatter.update_layout(
    title={
        'text': data_name_dict[option3[0]]+" (normalised) vs Residual Plot",
        # 'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
st.plotly_chart(res_scatter , use_container_width=True)


if len(option3) == 2:
    graph4 = px.scatter(df_dbscan, x=option3[0], y = option3[1], color =df_dbscan['dbscan_outliers'],
                     labels={
                     option3[0]: data_name_dict[option3[0]],
                     option3[1]: data_name_dict[option3[1]],
                 }, color_discrete_map=color_discrete_map, hover_data=["date"]
    )
    graph4.update_traces(marker={'size': 12})
    graph4.update_layout(
        title={
            'text': data_name_dict[option3[0]]+" vs "+ data_name_dict[option3[1]],
            # 'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
else:
    graph4 = px.scatter_3d(df_dbscan,x=option3[0],
                        y=option3[1],
                        z=option3[2],
                        color=df_dbscan['dbscan_outliers'],
                    labels={
                     option3[0]: data_name_dict[option3[0]],
                     option3[1]: data_name_dict[option3[1]],
                     option3[2]: data_name_dict[option3[2]],
                    }, color_discrete_map=color_discrete_map, hover_data=["date"]
                 )
    graph4.update_layout(width=900,height=700,legend=dict(x=0.7,font_size=20),
            title={
            'text': data_name_dict[option3[0]]+" vs "+ data_name_dict[option3[1]] + " vs " + data_name_dict[option3[2]],
            # 'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


st.write('Below are the identified outliers:')
st.write(df_dbscan[df_dbscan["dbscan_outliers"]=="Noise"])

st.subheader('3.2 Isolation forest')
st.info("""
Isolation forests are built using decision trees. 
This exercise uses isolation forest from the scikit learn package. 
The outliers are indicated by negative scores, while the positive score implies that we have normal observation.
""")


contamination = st.slider('Choose the contamination value:', min_value = 0.1, max_value= 0.5, value= 0.2, step=0.1 )

model2=IsolationForest(n_estimators=100,max_samples='auto',contamination=contamination).fit(dataset[option3])
df_if['scores'] = model2.decision_function(dataset[option3])
df_if['if_outliers'] = model2.predict(dataset[option3])
df_if['anomaly']=df_if['if_outliers'].apply(lambda x: 'outlier' if x==-1  else 'normal')


if len(option3) == 2:
    graph5 = px.scatter(df_if, x=option3[0], y = option3[1], color ='anomaly',
                     labels={
                     option3[0]: data_name_dict[option3[0]],
                     option3[1]: data_name_dict[option3[1]],
                 }, hover_data=["date"])
    graph5.update_traces(marker={'size': 12})
    graph5.update_layout(
        title={
            'text': data_name_dict[option3[0]]+" vs "+ data_name_dict[option3[1]],
            # 'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
else:
    graph5 = px.scatter_3d(df_if,x=option3[0],
                        y=option3[1],
                        z=option3[2],
                        color='anomaly',
                    labels={
                     option3[0]: data_name_dict[option3[0]],
                     option3[1]: data_name_dict[option3[1]],
                     option3[2]: data_name_dict[option3[2]],
                    }, hover_data=["date"])
    graph5.update_layout(width=900,height=700,legend=dict(x=0.7,font_size=20),
            title={
            'text': data_name_dict[option3[0]]+" vs "+ data_name_dict[option3[1]] + " vs " + data_name_dict[option3[2]],
            # 'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


st.plotly_chart(graph5 , use_container_width=True)
st.write('Below are the identified outliers:')
st.write(df_if[df_if['if_outliers']==-1])