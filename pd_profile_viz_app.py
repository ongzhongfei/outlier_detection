import pandas as pd

import pandas_profiling

import streamlit as st

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
# df = pd.read_csv("crops data.csv", na_values=['='])

def pd_profile_app(input_df):
    profile = ProfileReport(input_df,
    vars={"num": {"low_categorical_threshold": 0}} ,
    progress_bar=True
    )

    st.title("Pandas Profiling in Streamlit!")
    st_profile_report(profile)
