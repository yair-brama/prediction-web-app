# -*- coding: utf-8 -*-
"""
Entry point for the Revenue & Cards Predictor multi-page app.
Uses st.navigation so page titles in the sidebar are explicit strings
rather than being derived from filenames.
"""

import streamlit as st

st.set_page_config(
    page_title="Revenue & Cards Predictor",
    page_icon="💰",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/prediction_app.py",     title="Prediction App",     icon="💰", default=True),
    st.Page("pages/analysis.py",           title="Data Analysis",      icon="📊"),
])

pg.run()
