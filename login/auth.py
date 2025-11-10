# auth.py
import os
import streamlit as st
import sqlalchemy as sa
from sqlalchemy import text
import bcrypt

@st.cache_resource
def _engine():
    if "database" in st.secrets:
        db=st.secrets["database"]
        url=f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['db_name']}?charset=utf8mb4"
    else:
        url=(os.environ.get("DB_URL") or "").strip()
        if not url: raise RuntimeError("DB_URL not set and secrets.database missing")
    return sa.create_engine(url, pool_pre_ping=True)

def _exec(sql, params=None, many=False):
    with _engine().begin() as cx:
        if many:
            cx.execute(text(sql), params)
            return
        r=cx.execute(text(sql), params or {})
        try: return r.fetchall()
        except: return None

def _tbl_exists(t):
    return _exec("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:t",{"t":t})[0][0]>0

def _col_exists(t,c):
    return _exec("SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:t AND COLUMN_NAME=:c",{"t":t,"c":c})[0][0]>0

def _ensure_schema():
    _exec("""CREATE TABLE IF NOT EXISTS users(
      id INT AUTO_INCREMENT PRIMARY KEY,
      username VARCHAR(64) NOT NULL UNIQUE,
      password_hash VARCHAR(200) NOT NULL,
      is_active TINYINT(1) NOT NULL DEFAULT 1,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")

    _exec("""CREATE TABLE IF NOT EXISTS roles(
      id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(32) NOT NULL UNIQUE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")

    if not _tbl_exists("permissions"):
        _exec("""CREATE TABLE permissions(
          id INT AUTO_INCREMENT PRIMARY KEY,
          code VARCHAR(64) NOT NULL UNIQUE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")
    else:
        if not _col_exists("permissions","code"):
            _exec("ALTER TABLE permissions ADD COLUMN code VARCHAR(64) UNIQUE")

    if not _tbl_exists("role_permissions"):
        if _col_exists("permissions","id"):
            _exec("""CREATE TABLE role_permissions(
              role_id INT NOT NULL,
              permission_id INT NOT NULL,
              PRIMARY KEY(role_id, permission_id),
              FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE,
              FOREIGN KEY(permission_id) REFERENCES permissions(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")
        else:
            _exec("""CREATE TABLE role_permissions(
              role_id INT NOT NULL,
              permission_code VARCHAR(64) NOT NULL,
              PRIMARY KEY(role_id, permission_code),
              FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE,
              FOREIGN KEY(permission_code) REFERENCES permissions(code) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")

    _exec("""CREATE TABLE IF NOT EXISTS user_roles(
                                                      user_id INT NOT NULL,
                                                      role_id INT NOT NULL,
                                                      PRIMARY KEY(user_id,role_id),
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(role_id) REFERENCES roles(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")

    _exec("INSERT IGNORE INTO roles(name) VALUES ('admin'),('operator'),('viewer');")
    _exec("INSERT IGNORE INTO permissions(code) VALUES ('view_dashboard'),('manage_users'),('weather_protected');")

    if _col_exists("role_permissions", "permission_code"):
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_code)
                 SELECT r.id, 'view_dashboard' FROM roles r WHERE r.name IN ('admin','operator','viewer');""")
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_code)
                 SELECT r.id, 'manage_users' FROM roles r WHERE r.name='admin';""")
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_code)
                 SELECT r.id, 'weather_protected' FROM roles r WHERE r.name IN ('admin','operator');""")
    else:
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_id)
                 SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code='view_dashboard'
                 WHERE r.name IN ('admin','operator','viewer');""")
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_id)
        SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code='manage_users'
        WHERE r.name='admin';""")
        _exec("""INSERT IGNORE INTO role_permissions(role_id, permission_id)
                 SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code='weather_protected'
                 WHERE r.name IN ('admin','operator');""")



_ensure_schema()

def _hash(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def _chk(p,h): return bcrypt.checkpw(p.encode(), h.encode())

def _user_by_name(u):
    rows = _exec(
        "SELECT id, username, password_hash, is_active FROM users WHERE username=:u",
        {"u": u}
    )
    return _row2dict(rows[0]) if rows else None


def _roles_of_user(uid):
    rows=_exec("SELECT r.name FROM user_roles ur JOIN roles r ON ur.role_id=r.id WHERE ur.user_id=:i",{"i":uid})
    return [r[0] for r in rows] if rows else []

def current_user():
    return st.session_state.get("_user")

def login(username, password):
    u=_user_by_name(username)
    if not u or not u["is_active"]: return False
    if not _chk(password, u["password_hash"]): return False
    roles=_roles_of_user(u["id"])
    st.session_state["_user"]={"id":u["id"],"username":u["username"],"roles":roles}
    return True

def logout():
    st.session_state.pop("_user", None)

def _perms_of_roles(role_names):
    if not role_names:
        return set()
    has_code = _col_exists("role_permissions", "permission_code")
    ph = ",".join([f":n{i}" for i in range(len(role_names))])
    params = {f"n{i}": rn for i, rn in enumerate(role_names)}
    if has_code:
        sql = f"""
            SELECT rp.permission_code AS p
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            WHERE r.name IN ({ph})
        """
        rows = _exec(sql, params)
    else:
        sql = f"""
            SELECT p.code AS p
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE r.name IN ({ph})
        """
        rows = _exec(sql, params)
    return set([row[0] for row in rows]) if rows else set()


def require_perms(perms):
    u=current_user()
    if not u: st.switch_page("login.py")
    allowed=_perms_of_roles(u.get("roles",[]))
    if not set(perms).issubset(allowed):
        st.error("권한이 없습니다"); st.stop()

def create_user(username, password, roles=None):
    total=_exec("SELECT COUNT(*) AS c FROM users")[0][0]
    if total==0: roles=["admin"]
    if roles is None: roles=["viewer"]
    if _user_by_name(username): raise RuntimeError("이미 존재하는 아이디")
    h=_hash(password)
    _exec("INSERT INTO users(username,password_hash) VALUES(:u,:h)",{"u":username,"h":h})
    uid=_exec("SELECT id FROM users WHERE username=:u",{"u":username})[0][0]
    role_ids=[]
    for rn in roles:
        rid=_exec("SELECT id FROM roles WHERE name=:n",{"n":rn})
        if rid: role_ids.append(rid[0][0])
    rows=[{"user_id":uid,"role_id":rid} for rid in role_ids]
    if rows: _exec("INSERT IGNORE INTO user_roles(user_id,role_id) VALUES(:user_id,:role_id)", rows, many=True)
    return {"id":uid,"username":username,"roles":roles}

def delete_user_self(username, password):
    u=_user_by_name(username)
    if not u: raise RuntimeError("존재하지 않는 사용자")
    if not _chk(password, u["password_hash"]): raise RuntimeError("비밀번호 불일치")
    _exec("DELETE FROM users WHERE id=:i",{"i":u["id"]})
    logout()
    return True

def list_roles():
    r=_exec("SELECT name FROM roles ORDER BY name")
    return [x[0] for x in r] if r else []

def users_summary():
    r = _exec("""
        SELECT u.id, u.username,
               GROUP_CONCAT(r.name ORDER BY r.name SEPARATOR ',') AS roles
        FROM users u
        LEFT JOIN user_roles ur ON u.id = ur.user_id
        LEFT JOIN roles r       ON r.id = ur.role_id
        GROUP BY u.id, u.username
        ORDER BY u.id
    """)
    return [_row2dict(x) for x in r] if r else []


def _row2dict(row):
    try:
        return dict(row._mapping)
    except Exception:
        return dict(row)

def admin_users():
    rows=_exec("""
        SELECT u.id,u.username,u.is_active,
               COALESCE(GROUP_CONCAT(r.name ORDER BY r.name SEPARATOR ','),'') AS roles
        FROM users u
        LEFT JOIN user_roles ur ON u.id=ur.user_id
        LEFT JOIN roles r ON r.id=ur.role_id
        GROUP BY u.id,u.username,u.is_active
        ORDER BY u.id
    """)
    return [_row2dict(x) for x in rows] if rows else []

def admin_set_roles(username, roles):
    u=_exec("SELECT id FROM users WHERE username=:u",{"u":username})
    if not u: raise RuntimeError("존재하지 않는 사용자")
    uid=u[0][0]
    _exec("DELETE FROM user_roles WHERE user_id=:i",{"i":uid})
    if roles:
        ids=[]
        for rn in roles:
            r=_exec("SELECT id FROM roles WHERE name=:n",{"n":rn})
            if r: ids.append(r[0][0])
        rows=[{"user_id":uid,"role_id":rid} for rid in ids]
        if rows: _exec("INSERT IGNORE INTO user_roles(user_id,role_id) VALUES(:user_id,:role_id)",rows,many=True)
    return True

def admin_set_active(username, is_active):
    _exec("UPDATE users SET is_active=:a WHERE username=:u",{"a":1 if is_active else 0,"u":username})
    return True

def admin_reset_password(username, new_password):
    h=_hash(new_password)
    _exec("UPDATE users SET password_hash=:h WHERE username=:u",{"h":h,"u":username})
    return True

def admin_delete_user(username):
    _exec("DELETE FROM users WHERE username=:u",{"u":username})
    return True
