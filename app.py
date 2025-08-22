import os, json, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

# --------- PATHS ----------
MODEL_PATH = "models/model.joblib"
META_PATH  = "models/meta.json"
TRAIN_DEMO_CSV = "data/students_train_synth_v1.csv"   # ใช้ฝึกเดโมบนเซิร์ฟเวอร์

# --------- PAGE ----------
st.set_page_config(page_title="Student Risk Demo", layout="wide")
st.title("ระบบชี้เป้านักเรียนเสี่ยง – เดโม (MVP)")
st.markdown(
    "อัปโหลดไฟล์ **CSV** ที่มีคอลัมน์: "
    "`student_id, grade_level, gpa, failed_subjects, attendance_pct, late_count, discipline_points, midterm_avg, final_avg` "
    "(ถ้ามีคอลัมน์ `label` จะโชว์เมตริกเปรียบเทียบด้วย)"
)

# --------- HELPERS ----------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def save_meta(numeric_features, categorical_features):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "target": "label"
            },
            f, ensure_ascii=False, indent=2
        )

def train_demo_model(train_csv=TRAIN_DEMO_CSV):
    """ฝึกโมเดลเดโมบนเครื่อง/เซิร์ฟเวอร์ (ใช้ชุดสังเคราะห์ในโฟลเดอร์ data/)"""
    NUMERIC_FEATURES = ['gpa','failed_subjects','attendance_pct','late_count',
                        'discipline_points','midterm_avg','final_avg']
    CATEGORICAL_FEATURES = ['grade_level']

    df = pd.read_csv(train_csv)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['label'].astype(int)

    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', numeric_pipe, NUMERIC_FEATURES),
                             ('cat', categorical_pipe, CATEGORICAL_FEATURES)])
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    pipe = Pipeline([('prep', pre), ('clf', clf)])
    pipe.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    save_meta(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return pipe

def sort_grade(df, col="grade_level"):
    def key(v):
        m = re.search(r"(\d+)", str(v))
        return int(m.group(1)) if m else 999
    return df.sort_values(col, key=lambda s: s.map(key))

# --------- UI: FILE UPLOAD ----------
uploaded = st.file_uploader("อัปโหลดไฟล์นักเรียน (.csv)", type=["csv"])
if uploaded is None:
    st.info("โปรดอัปโหลดไฟล์ CSV เพื่อเริ่มประมวลผล")
    st.stop()

df = pd.read_csv(uploaded)
st.write("ตัวอย่างข้อมูล:")
st.dataframe(df.head(20))

# --------- PREP FEATURE LIST ----------
try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    needed_cols = meta["numeric_features"] + meta["categorical_features"]
except Exception:
    # กรณียังไม่มี meta: ใช้ลิสต์ค่าเริ่มต้น
    needed_cols = ['gpa','failed_subjects','attendance_pct','late_count',
                   'discipline_points','midterm_avg','final_avg','grade_level']

# เติมคอลัมน์ที่ขาด และแปลงฟีเจอร์ตัวเลขให้เป็นตัวเลข
for c in needed_cols:
    if c not in df.columns:
        df[c] = np.nan
numeric_guess = ['gpa','failed_subjects','attendance_pct','late_count','discipline_points','midterm_avg','final_avg']
for c in numeric_guess:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[needed_cols]  # ต้องสร้าง X ตั้งแต่ตรงนี้

# --------- LOAD / TRAIN MODEL ----------
model = load_model()
if model is None:
    st.warning("ยังไม่มีโมเดลที่ฝึกแล้ว")
    if st.button("ฝึกโมเดลเดโมตอนนี้ (ใช้ data/students_train_synth_v1.csv)"):
        model = train_demo_model()
        st.success("ฝึกโมเดลและบันทึกไว้ที่ models/model.joblib เรียบร้อย")
    else:
        st.stop()

# --------- PREDICT (HANDLE VERSION MISMATCH) ----------
try:
    proba = model.predict_proba(X)[:, 1]
except Exception as e:
    st.warning(f"โมเดลที่โหลดมาใช้ไม่ได้กับเวอร์ชันนี้: {e}")
    if st.button("ฝึกโมเดลใหม่บนเซิร์ฟเวอร์ (เดโม)"):
        model = train_demo_model()
        proba = model.predict_proba(X)[:, 1]
    else:
        st.stop()

# --------- SIDEBAR: THRESHOLD / TOP-K ----------
st.sidebar.header("ตั้งค่า")
mode = st.sidebar.radio("โหมดคัดกรอง", ["โดย Threshold", "Top K"])
if mode == "โดย Threshold":
    thr = st.sidebar.slider("Threshold แบ่งกลุ่มเสี่ยง", 0.0, 1.0, 0.5, 0.01)
else:
    K_default = min(50, len(df)) if len(df) > 0 else 1
    K = st.sidebar.number_input("จำนวนคน (K)", min_value=1, max_value=int(len(df)) if len(df)>0 else 1,
                                value=K_default, step=1)
    thr = float(pd.Series(proba).nlargest(int(K)).min()) if len(df)>0 else 1.0
    st.sidebar.caption(f"คัด Top {int(K)} คนแรก (เทียบเท่า threshold = {thr:.3f})")

def band(p):
    # HIGH = ความเสี่ยงสูงมาก (>= max(thr, 0.8))
    if p >= max(thr, 0.8): return "HIGH"
    elif p >= thr:         return "MED"
    else:                  return "LOW"

# --------- OUTPUT TABLE ----------
df_out = df.copy()
df_out["risk_score"] = proba
df_out["risk_band"]  = df_out["risk_score"].apply(band)
df_sorted = df_out.sort_values("risk_score", ascending=False)

# เมตริกเมื่อมี label
if "label" in df_sorted.columns:
    try:
        y_true = df_sorted["label"].astype(int)
        auc = roc_auc_score(y_true, df_sorted["risk_score"])
        y_pred = (df_sorted["risk_score"] >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        st.subheader("เมตริก (เมื่อมี label)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC AUC", f"{auc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1", f"{f1:.3f}")
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))
    except Exception as e:
        st.info(f"คำนวณเมตริกไม่สำเร็จ: {e}")

st.subheader("รายชื่อนักเรียนจัดอันดับตามความเสี่ยง")
st.dataframe(df_sorted)

csv = df_sorted.to_csv(index=False).encode("utf-8-sig")
st.download_button("ดาวน์โหลดผล (CSV)", data=csv, file_name="student_risk_scored.csv", mime="text/csv")

# --------- EXECUTIVE DASHBOARD ----------
st.subheader("แดชบอร์ดผู้บริหาร – สรุปตามระดับชั้น")
choice = st.selectbox("นิยามผู้มีความเสี่ยง", ["High เท่านั้น", "Medium+High"])
risk_mask = (df_out["risk_band"] == "HIGH") if choice == "High เท่านั้น" else df_out["risk_band"].isin(["HIGH","MED"])

by_grade = (
    df_out.assign(is_risk=risk_mask)
    .groupby("grade_level")
    .agg(total=("student_id","count"), at_risk=("is_risk","sum"))
    .assign(pct_at_risk=lambda d: (d["at_risk"]/d["total"]*100).round(1))
    .reset_index()
)
by_grade = sort_grade(by_grade, "grade_level")

c1, c2 = st.columns([2, 3])
with c1:
    st.caption("ตารางสรุปต่อระดับชั้น")
    st.dataframe(by_grade)
with c2:
    st.caption("จำนวนผู้มีความเสี่ยงต่อระดับชั้น")
    st.bar_chart(by_grade.set_index("grade_level")["at_risk"])

# --------- OPTIONAL DEBUG ----------
if st.sidebar.checkbox("Debug (show files)"):
    st.write("cwd:", os.getcwd())
    st.write("exists MODEL_PATH:", os.path.exists(MODEL_PATH))
    st.write("models dir:", os.listdir("models") if os.path.isdir("models") else "no models dir")
