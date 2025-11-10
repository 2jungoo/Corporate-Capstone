# seed_rbac.py
import os, bcrypt, sqlalchemy as sa

DB=os.environ.get("DB_URL")
ADMIN_USER=os.environ.get("ADMIN_USER","admin")
ADMIN_PASS=os.environ.get("ADMIN_PASS","admin123!")
ADMIN_EMAIL=os.environ.get("ADMIN_EMAIL","admin@example.com")

engine=sa.create_engine(DB, pool_pre_ping=True)

def get_role_id(cx, name):
  r=cx.execute(sa.text("SELECT id FROM roles WHERE name=:n"),{"n":name}).fetchone()
  return r[0] if r else None

with engine.begin() as cx:
  u=cx.execute(sa.text("SELECT id FROM users WHERE username=:u"),{"u":ADMIN_USER}).fetchone()
  if not u:
    h=bcrypt.hashpw(ADMIN_PASS.encode(), bcrypt.gensalt()).decode()
    cx.execute(sa.text("INSERT INTO users(username,email,password_hash,is_active) VALUES(:u,:e,:h,1)"),
               {"u":ADMIN_USER,"e":ADMIN_EMAIL,"h":h})
    u=cx.execute(sa.text("SELECT id FROM users WHERE username=:u"),{"u":ADMIN_USER}).fetchone()
  uid=u[0]
  rid=get_role_id(cx,"admin")
  cx.execute(sa.text("INSERT IGNORE INTO user_roles(user_id,role_id) VALUES(:u,:r)"),{"u":uid,"r":rid})
print({"ok":True,"user":ADMIN_USER})
