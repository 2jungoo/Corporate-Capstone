import streamlit as st
from auth import current_user, create_user, login

st.set_page_config(page_title="회원가입", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarNav']{display:none;}</style>", unsafe_allow_html=True)

if current_user():
    st.switch_page("pages/20_Dashboard.py")

st.title("회원가입")

with st.form("signup_public"):
    a = st.text_input("아이디")
    b = st.text_input("비밀번호", type="password")
    c = st.text_input("비밀번호 확인", type="password")
    ok = st.form_submit_button("가입하기")

if ok:
    if not a or not b or not c:
        st.error("모든 값을 입력")
    elif b != c:
        st.error("비밀번호 불일치")
    else:
        try:
            new = create_user(a, b, None)
            login(a, b)
            st.success(f"{new['username']} 가입 완료")
            st.switch_page("pages/20_Dashboard.py")
        except Exception as e:
            st.error(str(e))
