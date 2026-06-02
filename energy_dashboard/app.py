import streamlit as st
from .db import init_db
from .views import home
from .views.process_view import render as render_process

st.set_page_config(page_title="דשבורד אנרגיה — קיבוץ", layout="wide", page_icon="⚡")

# RTL support
st.markdown("""
<style>
body, .stApp { direction: rtl; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

init_db()

page = st.sidebar.radio(
    "ניווט",
    ["סיכום", "ייצור PV במחלק", "ייצור PV מחוץ למחלק", "השלת גנרטור"],
)

if page == "סיכום":
    home.render()
elif page == "ייצור PV במחלק":
    render_process("pv_inside")
elif page == "ייצור PV מחוץ למחלק":
    render_process("pv_outside")
elif page == "השלת גנרטור":
    render_process("generator")
