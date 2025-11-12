# pages/10_User_Admin.py
import streamlit as st
from auth import (
    current_user, require_perms,
    admin_users, list_roles, admin_set_roles, admin_set_active,
    admin_reset_password, admin_delete_user, create_user,
    users_summary, count_admins
)

st.set_page_config(page_title="사용자 관리 (Admin)", layout="wide")

# 로그인 상태면 사이드바의 login / Signup 숨김
if current_user():
    st.markdown("""
    <style>
      [data-testid="stSidebar"] a:has(span:contains("login")),
      [data-testid="stSidebar"] a:has(span:contains("Signup")) { display:none !important; }
    </style>
    """, unsafe_allow_html=True)

# 권한 체크(관리자만 접근)
require_perms(["manage_users"])
u = current_user()

st.page_link("pages/20_Dashboard.py", label="◀ 대시보드로")
st.markdown("### 👤 사용자 관리 (Admin)")
st.caption(f"관리자: {u['username']}")

st.divider()

# -------------------------------------------------------------------
# 0) 데이터 로드
# -------------------------------------------------------------------
role_pool = list_roles()                # ['admin','operator','viewer', ...]
rows      = admin_users()               # id, username, is_active, roles(콤마)
created   = {d["username"]: d.get("created_at")
             for d in users_summary()}  # username -> created_at
only_one_admin = (count_admins() <= 1)

# -------------------------------------------------------------------
# 1) (관리자 전용) 새 사용자 생성
# -------------------------------------------------------------------
with st.expander("➕ 새 사용자 생성", expanded=False):
    with st.form("create_user_form"):
        cu_id  = st.text_input("아이디")
        cu_pw1 = st.text_input("비밀번호", type="password")
        cu_pw2 = st.text_input("비밀번호 확인", type="password")
        cu_rs  = st.multiselect("역할", role_pool, default=["viewer"])
        ok_new = st.form_submit_button("생성")
    if ok_new:
        if not cu_id or not cu_pw1 or not cu_pw2:
            st.error("모든 항목을 입력하세요.")
        elif cu_pw1 != cu_pw2:
            st.error("비밀번호가 일치하지 않습니다.")
        else:
            try:
                create_user(cu_id, cu_pw1, cu_rs)
                st.success(f"사용자 '{cu_id}' 생성 완료")
                st.rerun()
            except Exception as e:
                st.error(str(e))

st.subheader("사용자 목록")

# -------------------------------------------------------------------
# 2) 사용자별 관리 카드
# -------------------------------------------------------------------
if not rows:
    st.info("사용자가 없습니다.")
else:
    for idx, r in enumerate(rows):
        uid        = r["id"]
        uname      = r["username"]
        is_active  = bool(r["is_active"])
        roles_csv  = (r["roles"] or "").strip()
        role_list  = [s for s in roles_csv.split(",") if s] if roles_csv else []
        created_at = created.get(uname)

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 3, 3, 2])

            # 기본 정보
            with c1:
                st.markdown(f"**{uname}**")
                st.caption(f"생성: {created_at}" if created_at else "생성: -")

            # 역할 관리
            with c2:
                st.caption("역할")
                new_roles = st.multiselect(
                    f"roles_{uid}", role_pool, default=role_list, label_visibility="collapsed", key=f"roles_{uid}"
                )
                apply_roles = st.button("역할 적용", key=f"apply_roles_{uid}")
                if apply_roles:
                    try:
                        # 마지막 관리자를 admin에서 제거하는 상황 방지
                        if "admin" in role_list and "admin" not in new_roles and only_one_admin and uname != u["username"]:
                            st.warning("현재 유일한 admin입니다. admin 역할을 제거할 수 없습니다.")
                        else:
                            admin_set_roles(uname, new_roles)
                            st.success("역할이 업데이트되었습니다.")
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))

            # 활성/비활성 + 비밀번호 초기화
            with c3:
                st.caption("상태 / 비밀번호")
                new_active = st.toggle("활성화", value=is_active, key=f"active_{uid}")
                apply_active = st.button("상태 적용", key=f"apply_active_{uid}")

                if apply_active:
                    try:
                        # 마지막 admin 비활성화 방지
                        if "admin" in role_list and only_one_admin and not new_active:
                            st.warning("현재 유일한 admin은 비활성화할 수 없습니다.")
                        else:
                            admin_set_active(uname, new_active)
                            st.success("상태가 업데이트되었습니다.")
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))

                with st.popover("비밀번호 초기화", use_container_width=True):
                    npw1 = st.text_input("새 비밀번호", type="password", key=f"npw1_{uid}")
                    npw2 = st.text_input("새 비밀번호 확인", type="password", key=f"npw2_{uid}")
                    if st.button("초기화", key=f"do_reset_{uid}"):
                        if not npw1 or not npw2:
                            st.error("모두 입력하세요.")
                        elif npw1 != npw2:
                            st.error("비밀번호가 일치하지 않습니다.")
                        else:
                            try:
                                admin_reset_password(uname, npw1)
                                st.success("비밀번호가 초기화되었습니다.")
                            except Exception as e:
                                st.error(str(e))

            # 삭제
            with c4:
                st.caption("삭제")
                is_last_admin_target = ("admin" in role_list) and only_one_admin
                disable_delete = (uname == u["username"]) or is_last_admin_target
                tip = "본인 계정은 삭제할 수 없습니다." if uname == u["username"] else (
                      "유일한 admin 계정은 삭제할 수 없습니다." if is_last_admin_target else "")
                if st.button("삭제", key=f"del_{uid}", disabled=disable_delete):
                    if st.session_state.get(f"confirm_{uid}") != "yes":
                        st.session_state[f"confirm_{uid}"] = "yes"
                        st.warning("다시 한 번 '삭제'를 누르면 즉시 삭제됩니다.")
                    else:
                        try:
                            admin_delete_user(uname)
                            st.success(f"'{uname}' 삭제 완료")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                if tip:
                    st.caption(f"※ {tip}")
