import streamlit as st
import pandas as pd
import requests
import json


@st.experimental_memo(ttl = 86400)
def loaddata():
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
    df.sort_values(by = 'date', inplace = True)
    df['gdp_growth'] = df['gdp_growth'].astype(float)
    df = df[['year_dt', 'date', 'gdp_growth']]
    #   df['date'] = pd.PeriodIndex(df['date'], freq="Q")
    df.reset_index(drop = True, inplace = True)

# # To import data from apikijangportal.bnm.gov.my (MSB 3.5.1: Industrial Production Index (Annual Change))
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

    #### Combine df1 and df2
    dataset = df2.groupby(['date']).mean()
    dataset = dataset.rename(columns = {'all_div':'IPI All', 'min':'mining','ele':'electricity','man':'manufacturing'})

    dataset = dataset.merge(df[['date','gdp_growth']], on='date', how='outer')
    dataset['date'] = dataset['date'].sort_values(ascending=True).values
    # dataset.drop(dataset.tail(1).index,inplace=True)
    qs = dataset['date'].str.replace(r'(Q\d) (\d+)', r'\2-\1')
    dataset['datetime'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()

    #### Drop nan (for ML)
    dataset = dataset.dropna()
    return dataset


data_name_dict = {
        'gdp_growth' : 'Gross domestic product (GDP)',
        'IPI All' : 'IPI All Division',
        'mining' : 'Mining',
        'electricity' : 'Electricity',
        'manufacturing' : 'Manufacturing',
        'exp_ind_tot' : 'Export-oriented industries - Total',
        'exp_ind_ele_clu_tot' : 'Export-oriented industries - Electronic and electrical cluster - Total',
        'exp_ind_ele_clu_ele' : 'Export-oriented industries - Electronic and electrical cluster - Electronics',
        'exp_ind_ele_clu_ele_pro' : 'Export-oriented industries - Electronic and electrical cluster - Electricals products',
        'exp_ind_ele_clu_mac' : 'Export-oriented industries - Electronic and electrical cluster - Machineries',
        'exp_ind_pri_clu_tot' : 'Export-oriented industries - Primary-related cluster - Total',
        'exp_ind_pri_clu_che' : 'Export-oriented industries - Primary-related cluster - Chemicals and chemical products',
        'exp_ind_pri_clu_pet' : 'Export-oriented industries - Primary-related cluster - Petroleum products',
        'exp_ind_pri_clu_tex' : 'Export-oriented industries - Primary-related cluster - Textiles wearing apparel and footwear',
        'exp_ind_pri_clu_woo' : 'Export-oriented industries - Primary-related cluster - Wood and wood products',
        'exp_ind_pri_clu_rub' : 'Export-oriented industries - Primary-related cluster - Rubber products',
        'exp_ind_pri_clu_off' :' Export-oriented industries - Primary-related cluster - Off-estate processing',
        'exp_ind_pri_clu_pap' : 'Export-oriented industries - Primary-related cluster - Paper products',
        'dom_ind_tot' : 'Domestic-oriented industries - Total',
        'dom_ind_con_clu_tot' : 'Domestic-oriented industries - Construction-related cluster - Total',
        'dom_ind_con_clu_non' : 'Domestic-oriented industries - Construction-related cluster - Non-metallic mineral products',
        'dom_ind_con_clu_iro' : 'Domestic-oriented industries - Construction-related cluster - Iron and steel',
        'dom_ind_con_clu_fab' : 'Domestic-oriented industries - Construction-related cluster - Fabricated metal products',
        'dom_ind_csm_clu_tot' : 'Domestic-oriented industries - Consumer-related cluster - Total',
        'dom_ind_csm_clu_foo' : 'Domestic-oriented industries - Consumer-related cluster - Food products',
        'dom_ind_csm_clu_tra' :' Domestic-oriented industries - Consumer-related cluster - Transport equipment',
        'dom_ind_csm_clu_bev' : 'Domestic-oriented industries - Consumer-related cluster - Beverages',
        'dom_ind_csm_clu_tob' : 'Domestic-oriented industries - Consumer-related cluster - Tobacco products',
        'dom_ind_csm_clu_oth' : 'Domestic-oriented industries - Consumer-related cluster - Others',

}