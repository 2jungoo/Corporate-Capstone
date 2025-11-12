import streamlit as st
from auth import login, current_user

st.set_page_config(page_title="로그인", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"] {display:none;}
[data-testid="stSidebarNav"] {display:none;}
</style>
""", unsafe_allow_html=True)

if current_user():
    st.switch_page("pages/20_Dashboard.py")

st.title("로그인")

with st.form("login_form"):
    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")
    c1, c2 = st.columns([1, 1])
    with c1:
        ok_login = st.form_submit_button("로그인", use_container_width=True)
    with c2:
        ok_signup = st.form_submit_button("회원가입", use_container_width=True)

if ok_signup:
    st.switch_page("pages/05_Signup.py")
elif ok_login:
    if login(username, password):
        st.switch_page("pages/20_Dashboard.py")
    else:
        st.error("로그인 실패")
