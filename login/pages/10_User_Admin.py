import streamlit as st
from auth import require_perms, current_user, admin_users, list_roles, admin_set_roles, admin_set_active, admin_reset_password, admin_delete_user, create_user

st.set_page_config(page_title="유저 관리", layout="wide")
require_perms(["manage_users"])

u=current_user()
st.title("유저 관리")

data=admin_users()
if data:
    st.dataframe(data, use_container_width=True, hide_index=True)
else:
    st.info("사용자 없음")

usernames=[d["username"] for d in data] if data else []
colL,colR=st.columns([1,1])

with colL:
    st.subheader("선택한 사용자 수정")
    sel=st.selectbox("사용자", usernames, index=0 if usernames else None, placeholder="사용자 선택")
    if sel:
        selected=[d for d in data if d["username"]==sel][0]
        roles_all=list_roles()
        cur_roles=[r for r in (selected.get("roles","") or "").split(",") if r]
        new_roles=st.multiselect("역할", roles_all, default=cur_roles, key="edit_roles")
        new_active=st.checkbox("활성화", value=bool(selected.get("is_active",1)), key="edit_active")
        c1,c2=st.columns(2)
        with c1:
            if st.button("역할 저장", use_container_width=True):
                try:
                    admin_set_roles(sel, new_roles)
                    st.success("저장 완료"); st.rerun()
                except Exception as e:
                    st.error(str(e))
        with c2:
            if st.button("활성화 상태 저장", use_container_width=True):
                try:
                    admin_set_active(sel, new_active)
                    st.success("저장 완료"); st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader("비밀번호 초기화")
    if sel:
        npw1=st.text_input("새 비밀번호", type="password")
        npw2=st.text_input("새 비밀번호 확인", type="password")
        if st.button("비밀번호 변경", use_container_width=True, type="primary", disabled=not npw1 or not npw2):
            if npw1!=npw2:
                st.error("비밀번호 불일치")
            else:
                try:
                    admin_reset_password(sel, npw1)
                    st.success("변경 완료")
                except Exception as e:
                    st.error(str(e))

with colR:
    st.subheader("새 사용자 생성")
    nu=st.text_input("아이디")
    np1=st.text_input("비밀번호", type="password")
    np2=st.text_input("비밀번호 확인", type="password")
    nroles=st.multiselect("역할", list_roles(), default=["viewer"])
    if st.button("생성", use_container_width=True):
        if not nu or not np1 or not np2:
            st.error("모든 값을 입력")
        elif np1!=np2:
            st.error("비밀번호 불일치")
        else:
            try:
                create_user(nu, np1, nroles)
                st.success("생성 완료"); st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("사용자 삭제")
    if sel:
        st.warning("삭제는 되돌릴 수 없음")
        confirm=st.text_input("확인용으로 사용자 아이디 입력")
        me=current_user()["username"]
        disabled=(confirm!=sel) or (sel==me)
        if st.button("영구 삭제", use_container_width=True, disabled=disabled):
            try:
                admin_delete_user(sel)
                st.success("삭제 완료"); st.rerun()
            except Exception as e:
                st.error(str(e))
