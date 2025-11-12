import streamlit as st
from auth import current_user, logout

def user_header_inline():
    u = current_user()
    c1, c2 = st.columns([2, 8])
    with c1:
        st.markdown(f"**{u['username']}**")
        if st.button("로그아웃", key="logout_inline"):
            logout()
            st.switch_page("login.py")
