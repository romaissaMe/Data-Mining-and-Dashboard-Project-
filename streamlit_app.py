import streamlit as st
import plotly
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")
st.sidebar.header("choose a database")
database = st.sidebar.selectbox("select a database", ("sol database", "covid19 database2", "database3"))
balance = 1000

st.metric(
        label="A/C Balance ＄",
        value=f"$ {round(balance,2)} ",
        delta=-round(balance /2) * 100,
    )
