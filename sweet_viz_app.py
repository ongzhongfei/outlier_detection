import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
# https://pypi.org/project/sweetviz/
# https://github.com/fbdesignpro/sweetviz
import sweetviz as sv

@st.experimental_memo
def sweet_app(input_df, data_mode):

    # st.title(title)
    df = input_df.copy()
    # st.write(df.head())
    if data_mode == "Use demo data":
        # skip_columns_years = [str(y) for y in range(2016, 2026)]
        skip_columns_datetime = ['date', 'year_dt', 'month_dt', 'period', 'datetime'] 

        # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
        analysis = sv.analyze([df,'EDA'], feat_cfg=sv.FeatureConfig(
            skip=skip_columns_datetime,
            force_text=[]), target_feat='gdp_growth')
    elif data_mode == "Upload your own data":
        analysis = sv.analyze([df,'EDA'], feat_cfg=sv.FeatureConfig(
            force_text=[]))

    # Render the output on a web page.
    analysis.show_html(filepath='./EDA.html', open_browser=False, layout='vertical', scale=0.9)

    HtmlFile = open("EDA.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    # print(source_code)
    return source_code
    # components.html(source_code, scrolling=True)
    # components.html(source_code, width=1600, height=800, scrolling=True)
    # components.iframe(src='http://localhost:3001/EDA.html', width=1100, height=1200, scrolling=True)