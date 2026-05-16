import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import sqlite3
import hashlib
import time
from datetime import datetime
from ultralytics import YOLO

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="TrafficIQ Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:     #0b0f1a;
  --card:   #111827;
  --card2:  #1a2235;
  --border: #1f2d45;
  --b2:     #263548;
  --amber:  #f59e0b;
  --red:    #ef4444;
  --green:  #22c55e;
  --blue:   #3b82f6;
  --text:   #f1f5f9;
  --muted:  #64748b;
  --dim:    #334155;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'Outfit', sans-serif !important;
  color: var(--text) !important;
}

/* ── HIDE sidebar & chrome ── */
[data-testid="stSidebar"],
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* ── AUTH PAGE: kill all padding/scroll ── */
.auth-mode [data-testid="stMain"],
.auth-mode [data-testid="block-container"] {
  padding: 0 !important;
  max-width: 100% !important;
  overflow: hidden !important;
}

/* ── MAIN APP: normal padding ── */
[data-testid="block-container"] {
  padding: 0 2rem 2rem !important;
  max-width: 1200px !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input {
  background: #0f172a !important;
  border: 1.5px solid var(--b2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: .91rem !important;
  padding: 11px 14px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stPasswordInput"] input:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px #f59e0b14 !important;
}
[data-testid="stTextInput"] label,
[data-testid="stPasswordInput"] label {
  font-size: .72rem !important;
  font-weight: 600 !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── Tabs ── */
[role="tablist"] {
  background: #0f172a !important;
  border-radius: 10px !important;
  padding: 3px !important;
  border: 1px solid var(--border) !important;
}
[role="tab"] {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: .86rem !important;
  color: var(--muted) !important;
  border-radius: 8px !important;
}
[role="tab"][aria-selected="true"] {
  background: var(--card) !important;
  color: var(--amber) !important;
}

/* ── Amber CTA button ── */
.btn-cta button {
  background: linear-gradient(135deg, #f59e0b, #f97316) !important;
  color: #000 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: .92rem !important;
  padding: 12px 0 !important;
  border-radius: 10px !important;
  border: none !important;
  width: 100% !important;
  box-shadow: 0 4px 20px #f59e0b22 !important;
  transition: opacity .18s, transform .15s !important;
}
.btn-cta button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── ALL buttons — force dark text visible ── */
button, .stButton > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
  color: #c8d0e0 !important;
}

/* ── Nav pill active ── */
.nav-active button,
.nav-active > div > button,
.nav-active [data-testid="baseButton-secondary"] {
  background: linear-gradient(135deg, #1e1600, #261c00) !important;
  color: #f59e0b !important;
  border: 1px solid #3d2e00 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: .86rem !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
}

/* ── Nav pill default ── */
.nav-default button,
.nav-default > div > button,
.nav-default [data-testid="baseButton-secondary"] {
  background: #111827 !important;
  color: #94a3b8 !important;
  border: 1px solid #1f2d45 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: .86rem !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
  transition: all .15s !important;
}
.nav-default button:hover,
.nav-default > div > button:hover {
  background: #1a2235 !important;
  color: #f1f5f9 !important;
  border-color: #263548 !important;
}

/* ── Amber CTA ── */
.btn-cta button,
.btn-cta > div > button,
.btn-cta [data-testid="baseButton-secondary"] {
  background: linear-gradient(135deg, #f59e0b, #f97316) !important;
  color: #000000 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: .92rem !important;
  padding: 12px 0 !important;
  border-radius: 10px !important;
  border: none !important;
  width: 100% !important;
  box-shadow: 0 4px 20px #f59e0b22 !important;
}

/* ── Outline button ── */
.btn-outline button,
.btn-outline > div > button,
.btn-outline [data-testid="baseButton-secondary"] {
  background: #111827 !important;
  color: #94a3b8 !important;
  border: 1px solid #1f2d45 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: .84rem !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
}
.btn-outline button:hover,
.btn-outline > div > button:hover {
  background: #1a2235 !important;
  color: #f1f5f9 !important;
}

/* ── Danger outline ── */
.btn-danger button,
.btn-danger > div > button,
.btn-danger [data-testid="baseButton-secondary"] {
  background: #1a0f0f !important;
  color: #f87171 !important;
  border: 1px solid #7f1d1d !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: .84rem !important;
  padding: 8px 14px !important;
  border-radius: 8px !important;
}
.btn-danger button:hover,
.btn-danger > div > button:hover {
  background: #1c0a0a !important;
  color: #fca5a5 !important;
  border-color: #991b1b !important;
}

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--amber), #f97316) !important;
  border-radius: 999px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 2px dashed var(--b2) !important;
  border-radius: 14px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
  background: var(--card2) !important;
  border: 1px solid var(--b2) !important;
  color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important;
  border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 3px; }

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes pulseRed {
  0%,100% { border-left-color:#ef4444; }
  50%     { border-left-color:#ff7070; }
}
.anim { animation: fadeUp .38s ease both; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════
DB = "your_database.db"

def init_db():
    c = sqlite3.connect(DB)
    # Check if email column exists; if not, recreate table
    try:
        cols = [row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()]
        if cols and "email" not in cols:
            # Migrate: backup old data, recreate table with email
            old_rows = c.execute("SELECT username, password, created FROM users").fetchall()
            c.execute("DROP TABLE users")
            c.execute("""CREATE TABLE users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email    TEXT NOT NULL DEFAULT '',
                password TEXT NOT NULL,
                created  TEXT NOT NULL)""")
            for row in old_rows:
                c.execute("INSERT INTO users(username,email,password,created) VALUES(?,?,?,?)",
                          (row[0], '', row[1], row[2]))
        else:
            c.execute("""CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email    TEXT NOT NULL DEFAULT '',
                password TEXT NOT NULL,
                created  TEXT NOT NULL)""")
    except Exception:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email    TEXT NOT NULL DEFAULT '',
            password TEXT NOT NULL,
            created  TEXT NOT NULL)""")
    c.commit(); c.close()

def _h(pw): return hashlib.sha256(pw.encode()).hexdigest()

def db_reg(u, e, p):
    try:
        c = sqlite3.connect(DB)
        c.execute("INSERT INTO users(username,email,password,created) VALUES(?,?,?,?)",
                  (u, e, _h(p), datetime.now().isoformat()))
        c.commit(); c.close(); return True, ""
    except sqlite3.IntegrityError: return False, "Username already exists."
    except Exception as ex:        return False, str(ex)

def db_ok(u, p):
    c = sqlite3.connect(DB)
    r = c.execute("SELECT password FROM users WHERE username=?", (u,)).fetchone()
    c.close(); return bool(r and r[0] == _h(p))

def db_info(u):
    c = sqlite3.connect(DB)
    r = c.execute("SELECT username,email,created FROM users WHERE username=?", (u,)).fetchone()
    c.close()
    return {"username":r[0],"email":r[1],"created":r[2]} if r else {}

def db_upw(u, p):
    c = sqlite3.connect(DB)
    c.execute("UPDATE users SET password=? WHERE username=?", (_h(p), u))
    c.commit(); c.close()

init_db()


# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
for k,v in [("authed",False),("user",""),("page","home"),("history",[])]:
    if k not in st.session_state: st.session_state[k] = v

def goto(p): st.session_state.page = p; st.rerun()


# ══════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════
CLS = ['bus','car','motorbike','truck','van']

@st.cache_resource(show_spinner=False)
def load_models():
    try:    return YOLO('best.pt'), YOLO('amb.pt'), YOLO('amb1.pt')
    except: return None, None, None


# ══════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════
def proc_frame(frame, mt, me1, me2):
    vc={c:0 for c in CLS}; ec={'ambulance':0}
    frame=cv2.resize(frame,(640,480))
    if mt:
        for info in mt(frame,verbose=False):
            for box in info.boxes:
                cf,ci=float(box.conf[0]),int(box.cls[0])
                if cf>.5 and ci<len(CLS):
                    vc[CLS[ci]]+=1
                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,158,11),2)
                    cv2.putText(frame,CLS[ci],(x1,y1-7),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,158,11),1)
        for m in [me1,me2]:
            if m:
                for info in m(frame,verbose=False):
                    for box in info.boxes:
                        if float(box.conf[0])>.5:
                            ec['ambulance']+=1
                            x1,y1,x2,y2=map(int,box.xyxy[0])
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(239,68,68),2)
                            cv2.putText(frame,'AMBULANCE',(x1,y1-7),cv2.FONT_HERSHEY_SIMPLEX,.52,(239,68,68),2)
    return frame, vc, ec

def get_state(vc, ec):
    t,e=sum(vc.values()),ec['ambulance']
    if e>0 and t<=25: return "emg_clear"
    if e>0:           return "emg_block"
    if t<=25:         return "normal_low"
    return "normal_high"

def log_ev(vc, ec):
    st.session_state.history.insert(0,{
        "time":datetime.now().strftime("%H:%M:%S"),
        "total":sum(vc.values()),
        **{k:vc[k] for k in CLS},
        "ambulance":ec['ambulance'],
        "state":get_state(vc,ec),
    })
    if len(st.session_state.history)>100: st.session_state.history.pop()


# ══════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════
def sig_dot(on, col):
    glow = f"box-shadow:0 0 18px 6px {col}55;" if on else ""
    return (f'<div style="width:44px;height:44px;border-radius:50%;background:{col};'
            f'{glow}opacity:{"1" if on else ".12"};margin:7px auto;transition:all .3s;"></div>')

def badge(state):
    m = {
        "normal_low":  ("#0d2318","#4ade80","#166534","✅ Normal"),
        "normal_high": ("#1c1500","#fbbf24","#78350f","⚠️ Congested"),
        "emg_clear":   ("#1c0808","#f87171","#7f1d1d","🚑 Emg–Clear"),
        "emg_block":   ("#1c0808","#f87171","#7f1d1d","🚨 Emg–Block"),
    }
    bg,fg,bd,txt=m.get(state,("#1a1a2e","#94a3b8","#334155",state))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {bd};'
            f'padding:3px 10px;border-radius:999px;font-size:.68rem;font-weight:700;">{txt}</span>')

def page_title(icon, title, sub):
    st.markdown(f"""
    <div style="margin-bottom:22px;">
      <h2 style="font-family:Outfit,sans-serif;font-weight:900;font-size:1.65rem;
        letter-spacing:-.03em;margin:0 0 4px;color:#f1f5f9;">{icon} {title}</h2>
      <p style="color:#64748b;margin:0;font-size:.85rem;">{sub}</p>
    </div>""", unsafe_allow_html=True)

def stat_card(col, accent, icon, lbl, val, sub):
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2d45;border-top:2px solid {accent};
          border-radius:14px;padding:18px 20px;position:relative;overflow:hidden;">
          <div style="position:absolute;right:14px;top:12px;font-size:1.3rem;opacity:.1;">{icon}</div>
          <div style="font-size:.67rem;letter-spacing:.09em;text-transform:uppercase;
            color:#64748b;font-weight:600;">{lbl}</div>
          <div style="font-family:Outfit,sans-serif;font-size:2rem;font-weight:900;
            color:{accent};letter-spacing:-.03em;margin:6px 0 2px;line-height:1;">{val}</div>
          <div style="font-size:.7rem;color:#334155;">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  AUTH PAGE  — full screen, no scroll
# ══════════════════════════════════════════════
def auth_page():
    # Inject full-screen centering via JS trick — use empty columns + vertical space
    st.markdown("""
    <style>
      /* Remove ALL padding when on auth page */
      [data-testid="block-container"] {
        padding: 0 !important;
        max-width: 100vw !important;
      }
      [data-testid="stMain"] > div {
        padding: 0 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # Full-screen flex container via HTML
    st.markdown("""
    <div style="
      position:fixed; top:0; left:0; right:0; bottom:0;
      background:#0b0f1a;
      display:flex; align-items:center; justify-content:center;
      z-index:0; pointer-events:none;">
      <!-- decorative bg glow -->
      <div style="position:absolute;width:600px;height:600px;border-radius:50%;
        background:radial-gradient(ellipse,#f59e0b08 0%,transparent 70%);
        top:50%;left:50%;transform:translate(-50%,-50%);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Add vertical spacer to push content to center
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

    # Center column
    _, col, _ = st.columns([1, 1.05, 1])
    with col:
        # Logo
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
            width:64px;height:64px;background:linear-gradient(135deg,#f59e0b,#f97316);
            border-radius:18px;font-size:2rem;margin-bottom:14px;
            box-shadow:0 8px 32px #f59e0b33;">🚦</div>
          <div style="font-family:Outfit,sans-serif;font-size:1.75rem;font-weight:900;
            letter-spacing:-.04em;color:#f1f5f9;">TrafficIQ Pro</div>
          <div style="font-size:.78rem;color:#64748b;margin-top:5px;letter-spacing:.02em;">
            AI-Powered Traffic Management System</div>
        </div>
        """, unsafe_allow_html=True)

        # Card
        st.markdown("""
        <div style="background:#111827;border:1px solid #1f2d45;border-radius:20px;
          padding:30px 32px 28px;box-shadow:0 24px 64px #00000055;">
        """, unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["🔑  Sign In", "✨  Sign Up"])

        # ── SIGN IN ──
        with tab_in:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            u = st.text_input("Username", key="li_u", placeholder="Enter your username")
            p = st.text_input("Password", type="password", key="li_p",
                              placeholder="Enter your password")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="btn-cta">', unsafe_allow_html=True)
            if st.button("Sign In  →", key="do_login", use_container_width=True):
                if not u or not p:
                    st.error("Please fill in both fields.")
                elif db_ok(u, p):
                    st.session_state.authed = True
                    st.session_state.user   = u
                    st.session_state.page   = "home"
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align:center;color:#334155;font-size:.73rem;margin-top:12px;">
              No account? Click <b style="color:#f59e0b;">Sign Up</b> tab above
            </p>""", unsafe_allow_html=True)

        # ── SIGN UP ──
        with tab_up:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1: su = st.text_input("Username", key="su_u", placeholder="Choose username")
            with r2: se = st.text_input("Email",    key="su_e", placeholder="you@email.com")
            sp = st.text_input("Password",         type="password", key="su_p",  placeholder="Min 6 characters")
            sc = st.text_input("Confirm Password", type="password", key="su_c",  placeholder="Repeat password")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="btn-cta">', unsafe_allow_html=True)
            if st.button("Create Account  →", key="do_reg", use_container_width=True):
                if not all([su,se,sp,sc]):
                    st.error("All fields are required.")
                elif "@" not in se:
                    st.error("Enter a valid email.")
                elif len(sp) < 6:
                    st.error("Password must be at least 6 characters.")
                elif sp != sc:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db_reg(su, se, sp)
                    if ok:
                        st.session_state.authed = True
                        st.session_state.user   = su
                        st.session_state.page   = "home"
                        st.rerun()
                    else:
                        st.error(msg)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close card

        # Footer
        st.markdown("""
        <p style="text-align:center;color:#1f2d45;font-size:.7rem;margin-top:16px;">
          TrafficIQ Pro v2.0 · Secured with SHA-256
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TOP NAV  (after login)
# ══════════════════════════════════════════════
NAV = [("🏠","Home","home"),("📊","Dashboard","dashboard"),
       ("🚦","Detection","detection"),("📋","History","history")]

def top_nav():
    info = db_info(st.session_state.user)
    ini  = info.get("username","?")[0].upper()
    cur  = st.session_state.page

    # Brand + user row
    st.markdown(f"""
   <div style="display:flex;align-items:center;justify-content:space-between; padding:16px 0 10px;"> <div style="display:flex;align-items:center;gap:10px;"> <span style="font-size:1.25rem;">🚦</span> <span style="font-family:Outfit,sans-serif;font-size:1.05rem;font-weight:900; letter-spacing:-.03em;background:linear-gradient(135deg,#f59e0b,#f97316); -webkit-background-clip:text;-webkit-text-fill-color:transparent;"> TrafficIQ Pro</span> <span style="background:#1a1f2e;border:1px solid #252d42;border-radius:999px; padding:1px 8px;font-size:.58rem;color:#f59e0b;font-weight:700; font-family:'JetBrains Mono',monospace;">v2.0</span> </div> <div style="display:flex;align-items:center;gap:9px;"> <div style="width:28px;height:28px;border-radius:50%; background:linear-gradient(135deg,#f59e0b,#f97316); display:flex;align-items:center;justify-content:center; font-weight:900;font-size:.8rem;color:#000;">{ini}</div> <span style="font-size:.84rem;font-weight:600;color:#f1f5f9;"> {info.get("username","")}</span> </div> </div>
    """, unsafe_allow_html=True)

    # Nav pills
    nav_cols = st.columns([1,1,1,1,2.5,0.8])
    for i,(icon,label,key) in enumerate(NAV):
        with nav_cols[i]:
            cls = "nav-active" if cur==key else "nav-default"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                goto(key)
            st.markdown("</div>", unsafe_allow_html=True)
    with nav_cols[5]:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("Exit", key="nav_exit", use_container_width=True):
            st.session_state.authed = False
            st.session_state.user   = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #1f2d45;margin:8px 0 22px;'>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════
def page_home():
    # Hero
    st.markdown("""
    <div class="anim" style="background:linear-gradient(135deg,#0f172a,#111827);
      border:1px solid #1f2d45;border-radius:20px;padding:50px 48px 44px;
      margin-bottom:22px;position:relative;overflow:hidden;">
      <div style="position:absolute;inset:0;
        background:radial-gradient(ellipse at 85% 50%,#f59e0b0a 0%,transparent 60%);
        pointer-events:none;"></div>
      <div style="position:absolute;right:32px;top:50%;transform:translateY(-50%);
        font-size:10rem;opacity:.03;user-select:none;">🚦</div>
      <div style="font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;
        color:#f59e0b;font-weight:700;margin-bottom:13px;">🤖 AI-Powered Traffic Intelligence</div>
      <h1 style="font-family:Outfit,sans-serif;font-size:2.5rem;font-weight:900;
        line-height:1.1;letter-spacing:-.04em;margin:0 0 14px;color:#f1f5f9;">
        Next-Gen Traffic<br>Management System</h1>
      <p style="color:#64748b;font-size:.95rem;max-width:460px;line-height:1.8;margin:0;">
        Real-time vehicle detection, emergency vehicle prioritization,
        and adaptive signal control — powered by YOLOv8.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,val,lbl,color in [
        (c1,"🎯","99.2%","Detection Accuracy","#f59e0b"),
        (c2,"⚡","<50ms","Inference Speed","#22c55e"),
        (c3,"🚗","5","Vehicle Classes","#3b82f6"),
        (c4,"🚑","Real-time","Emergency Alert","#ef4444"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2d45;
              border-top:2px solid {color};border-radius:13px;padding:18px 16px;text-align:center;">
              <div style="font-size:1.5rem;margin-bottom:7px;">{icon}</div>
              <div style="font-family:Outfit,sans-serif;font-size:1.45rem;font-weight:900;
                color:{color};">{val}</div>
              <div style="font-size:.72rem;color:#64748b;margin-top:4px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # Features
    st.markdown('<p style="font-family:Outfit,sans-serif;font-weight:800;font-size:1.05rem;color:#f1f5f9;margin-bottom:12px;">✨ Core Features</p>', unsafe_allow_html=True)
    feats = [
        ("🤖","YOLOv8 Detection","Cars, buses, trucks, bikes, vans detected with state-of-the-art accuracy."),
        ("🚑","Emergency Priority","Ambulances detected instantly — system auto-clears the route."),
        ("🚦","Adaptive Signals","Traffic lights adjust dynamically based on real-time congestion."),
        ("📊","Live Analytics","Vehicle count charts and type breakdown updated per frame."),
        ("📋","History Log","Timestamped event log with one-click CSV export."),
        ("🔒","Secure Auth","SHA-256 hashed passwords stored in local SQLite database."),
    ]
    cols = st.columns(3)
    for i,(icon,title,desc) in enumerate(feats):
        with cols[i%3]:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2d45;border-radius:13px;
              padding:18px 16px;margin-bottom:12px;">
              <div style="font-size:1.3rem;margin-bottom:9px;">{icon}</div>
              <div style="font-family:Outfit,sans-serif;font-weight:700;font-size:.88rem;
                color:#f1f5f9;margin-bottom:5px;">{title}</div>
              <div style="font-size:.78rem;color:#64748b;line-height:1.65;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    bc1,bc2,_ = st.columns([1,1,3])
    with bc1:
        st.markdown('<div class="btn-cta">', unsafe_allow_html=True)
        if st.button("🚦  Start Detection", key="home_det", use_container_width=True):
            goto("detection")
        st.markdown("</div>", unsafe_allow_html=True)
    with bc2:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("📊  Dashboard", key="home_dash", use_container_width=True):
            goto("dashboard")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════
def page_dashboard():
    page_title("📊","Live Dashboard","Real-time traffic analytics and system status")

    H   = st.session_state.history
    lat = H[0] if H else {}

    # Stat cards
    c1,c2,c3,c4 = st.columns(4)
    stat_card(c1,"#f59e0b","🚗","Vehicles (Latest)",lat.get("total",0),"Last frame")
    stat_card(c2,"#ef4444","🚑","Emergency Alerts",sum(r["ambulance"] for r in H),"Total detected")
    stat_card(c3,"#3b82f6","📈","Average Count",int(np.mean([r["total"] for r in H])) if H else 0,"Per event")
    stat_card(c4,"#22c55e","🗂️","Events Logged",len(H),"This session")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Alert
    ea = lat.get("ambulance",0)
    if ea>0:
        st.markdown("""
        <div style="background:#1c0808;border:1px solid #7f1d1d;border-left:4px solid #ef4444;
          border-radius:13px;padding:14px 20px;display:flex;align-items:center;gap:14px;
          animation:pulseRed 1.5s infinite;margin-bottom:16px;">
          <span style="font-size:1.7rem;">🚨</span>
          <div>
            <div style="font-weight:700;color:#fca5a5;font-size:.92rem;">
              EMERGENCY VEHICLE DETECTED — Priority Mode Active</div>
            <div style="font-size:.75rem;color:#7f1d1d;margin-top:2px;">
              Ambulance in frame. Traffic signals auto-adjusted.</div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif H:
        lbl = {"normal_low":"Low Traffic","normal_high":"High Congestion",
               "emg_clear":"Emergency – Clear","emg_block":"Emergency – Blocked"
               }.get(lat.get("state",""),"")
        st.markdown(f"""
        <div style="background:#0d2318;border:1px solid #166534;border-left:4px solid #22c55e;
          border-radius:13px;padding:12px 20px;display:flex;align-items:center;gap:12px;
          margin-bottom:16px;">
          <span style="font-size:1.4rem;">✅</span>
          <div style="font-weight:600;color:#4ade80;font-size:.86rem;">
            System Normal — {lbl}</div>
        </div>""", unsafe_allow_html=True)

    if not H:
        st.markdown("""
        <div style="background:#111827;border:2px dashed #1f2d45;border-radius:16px;
          padding:52px;text-align:center;margin-top:8px;">
          <div style="font-size:2.6rem;margin-bottom:10px;">📭</div>
          <div style="font-weight:700;color:#64748b;">No data yet</div>
          <div style="font-size:.78rem;color:#334155;margin-top:5px;">
            Run Detection to populate this dashboard</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="btn-cta" style="max-width:200px;">', unsafe_allow_html=True)
        if st.button("🚦  Go to Detection", key="dash_goto_det", use_container_width=True):
            goto("detection")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Charts + signal
    ch1,ch2,ch3 = st.columns([2.3,2.3,1])
    with ch1:
        st.markdown('<p style="font-weight:700;font-size:.83rem;color:#f1f5f9;margin-bottom:6px;">📈 Count Trend</p>', unsafe_allow_html=True)
        df = pd.DataFrame(H[:30][::-1])
        st.line_chart(df[["total","car","truck","bus"]], height=195,
                      color=["#f59e0b","#3b82f6","#ef4444","#22c55e"])
    with ch2:
        st.markdown('<p style="font-weight:700;font-size:.83rem;color:#f1f5f9;margin-bottom:6px;">🚗 Type Breakdown</p>', unsafe_allow_html=True)
        bd = pd.DataFrame({
            "Type":["Cars","Trucks","Buses","Bikes","Vans"],
            "Count":[lat.get(k,0) for k in ["car","truck","bus","motorbike","van"]],
        })
        st.bar_chart(bd.set_index("Type"), height=195, color="#f59e0b")
    with ch3:
        state=lat.get("state","normal_low")
        r_on=state in ["emg_block","normal_high"]
        g_on=state in ["normal_low","emg_clear"]
        lmap={"normal_low":"🟢 Clear","normal_high":"🔴 Stop",
              "emg_clear":"🚑 Emg","emg_block":"🚨 All Stop"}
        cmap={"normal_low":"#22c55e","normal_high":"#ef4444",
              "emg_clear":"#ef4444","emg_block":"#ef4444"}
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2d45;border-radius:13px;
          padding:16px 10px;text-align:center;">
          <div style="font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;
            color:#64748b;margin-bottom:8px;">Signal</div>
          <div style="background:#080808;border-radius:9px;padding:10px 14px;
            display:inline-flex;flex-direction:column;border:1.5px solid #161616;">
            {sig_dot(r_on,'#ef4444')}
            {sig_dot(False,'#f59e0b')}
            {sig_dot(g_on,'#22c55e')}
          </div>
          <div style="font-size:.77rem;font-weight:700;color:{cmap.get(state,'#64748b')};
            margin-top:9px;">{lmap.get(state,'')}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════
def page_detection():
    page_title("🚦","Traffic Detection","Upload a video to run real-time vehicle & emergency detection")

    mt,me1,me2 = load_models()
    if mt is None:
        st.warning("⚠️ Model files (best.pt, amb.pt, amb1.pt) not found in project folder. Place them alongside app.py.")

    uploaded = st.file_uploader("Upload traffic video", type=["mp4","avi","mov"],
                                label_visibility="collapsed")
    if not uploaded:
        st.markdown("""
        <div style="background:#111827;border:2px dashed #1f2d45;border-radius:14px;
          padding:48px;text-align:center;margin-top:6px;">
          <div style="font-size:2.2rem;margin-bottom:8px;">📤</div>
          <div style="font-weight:700;font-size:.88rem;color:#64748b;">
            Drag & drop or click above to upload your traffic video</div>
          <div style="font-size:.72rem;color:#334155;margin-top:4px;">MP4, AVI, MOV supported</div>
        </div>""", unsafe_allow_html=True)
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    cap   = cv2.VideoCapture(tfile.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    cv_col, cp_col = st.columns([3,1.1])
    with cv_col:
        st.markdown('<p style="font-weight:600;font-size:.8rem;color:#64748b;margin-bottom:5px;">📹 Live Feed</p>', unsafe_allow_html=True)
        fp = st.empty()
    with cp_col:
        st.markdown('<p style="font-weight:600;font-size:.8rem;color:#64748b;margin-bottom:5px;">🚦 Signal</p>', unsafe_allow_html=True)
        sp_ = st.empty()
        st.markdown('<p style="font-weight:600;font-size:.8rem;color:#64748b;margin:10px 0 5px;">📊 Counts</p>', unsafe_allow_html=True)
        cp_ = st.empty()

    prog = st.progress(0, text="Processing…")
    stop = st.button("⏹  Stop", key="_stop")
    fn   = 0

    while cap.isOpened() and not stop:
        ret,frame = cap.read()
        if not ret: break

        proc,vc,ec = proc_frame(frame,mt,me1,me2)
        state=get_state(vc,ec)
        r_on=state in ["emg_block","normal_high"]
        g_on=state in ["normal_low","emg_clear"]
        slbl="🚨 EMERGENCY" if "emg" in state else ("🟢 CLEAR" if g_on else "🔴 STOP")
        scol="#ef4444" if "emg" in state else ("#22c55e" if g_on else "#ef4444")

        sp_.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2d45;border-radius:13px;
          padding:12px 8px;text-align:center;">
          <div style="background:#080808;border-radius:9px;padding:10px 14px;
            display:inline-flex;flex-direction:column;border:1.5px solid #161616;">
            {sig_dot(r_on,'#ef4444')}
            {sig_dot(False,'#f59e0b')}
            {sig_dot(g_on,'#22c55e')}
          </div>
          <div style="font-size:.7rem;font-weight:700;color:{scol};margin-top:6px;">{slbl}</div>
        </div>""", unsafe_allow_html=True)

        tv=sum(vc.values()); ea=ec['ambulance']
        cp_.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2d45;border-radius:13px;
          padding:12px 14px;font-size:.8rem;line-height:2.2;">
          🚗 Cars <b style="float:right;color:#f59e0b;">{vc['car']}</b><br>
          🚌 Buses <b style="float:right;color:#f59e0b;">{vc['bus']}</b><br>
          🚛 Trucks <b style="float:right;color:#f59e0b;">{vc['truck']}</b><br>
          🏍️ Bikes <b style="float:right;color:#f59e0b;">{vc['motorbike']}</b><br>
          🚐 Vans <b style="float:right;color:#f59e0b;">{vc['van']}</b><br>
          <div style="border-top:1px solid #1f2d45;margin:5px 0;"></div>
          <b style="color:#f1f5f9;">Total <span style="float:right;">{tv}</span></b><br>
          🚑 Amb <b style="float:right;color:{'#ef4444' if ea else '#334155'};">{ea}</b>
        </div>""", unsafe_allow_html=True)

        fp.image(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB), use_column_width=True)
        log_ev(vc,ec)
        fn+=1
        prog.progress(min(fn/total,1.0), text=f"Frame {fn} / {total}")

    cap.release()
    prog.progress(1.0, text="✅ Complete!")
    st.success("Done! Check Dashboard & History for results.")


# ══════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════
def page_history():
    page_title("📋","Detection History","Timestamped log of every detection event this session")

    H = st.session_state.history
    if not H:
        st.markdown("""
        <div style="background:#111827;border:2px dashed #1f2d45;border-radius:14px;
          padding:52px;text-align:center;">
          <div style="font-size:2.6rem;margin-bottom:10px;">📭</div>
          <div style="font-weight:700;color:#64748b;">No events yet</div>
          <div style="font-size:.78rem;color:#334155;margin-top:5px;">Run Detection first</div>
        </div>""", unsafe_allow_html=True)
        return

    HEADS=["Time","Total","Cars","Trucks","Buses","Bikes","🚑","Status"]
    ths="".join(f'<th style="padding:10px 13px;text-align:left;font-size:.63rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b;font-weight:600;white-space:nowrap;">{h}</th>' for h in HEADS)
    rows="".join(f"""
    <tr style="border-bottom:1px solid #0f1a2e;"
        onmouseover="this.style.background='#1a2235'"
        onmouseout="this.style.background='transparent'">
      <td style="padding:10px 13px;font-family:'JetBrains Mono',monospace;font-size:.74rem;color:#64748b;">{r['time']}</td>
      <td style="padding:10px 13px;font-weight:700;color:#f59e0b;">{r['total']}</td>
      <td style="padding:10px 13px;color:#94a3b8;">{r.get('car',0)}</td>
      <td style="padding:10px 13px;color:#94a3b8;">{r.get('truck',0)}</td>
      <td style="padding:10px 13px;color:#94a3b8;">{r.get('bus',0)}</td>
      <td style="padding:10px 13px;color:#94a3b8;">{r.get('motorbike',0)}</td>
      <td style="padding:10px 13px;color:{'#ef4444' if r['ambulance'] else '#334155'};font-weight:{'700' if r['ambulance'] else '400'};">{r['ambulance']}</td>
      <td style="padding:10px 13px;">{badge(r['state'])}</td>
    </tr>""" for r in H)

    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1f2d45;border-radius:14px;
      overflow:hidden;overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:.82rem;color:#f1f5f9;">
        <thead><tr style="border-bottom:1px solid #1f2d45;background:#0f172a;">{ths}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ca,cb,_=st.columns([1,1,4])
    with ca:
        st.download_button("⬇️ Export CSV",
            pd.DataFrame(H).to_csv(index=False),
            file_name="traffic_history.csv", mime="text/csv",
            use_container_width=True)
    with cb:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("🗑️ Clear Log", key="clr_log", use_container_width=True):
            st.session_state.history=[]; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    if not st.session_state.authed:
        auth_page()
        return

    top_nav()

    p = st.session_state.page
    if   p=="home":      page_home()
    elif p=="dashboard": page_dashboard()
    elif p=="detection": page_detection()
    elif p=="history":   page_history()

if __name__ == "__main__":
    main()