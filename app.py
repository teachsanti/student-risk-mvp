import io, os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = 'models/model.joblib'
META_PATH = 'models/meta.json'

st.set_page_config(page_title='Student Risk Demo', layout='wide')
st.title('ระบบชี้เป้านักเรียนเสี่ยง – เดโม (MVP)')

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.markdown("""
อัปโหลดไฟล์ **CSV** ที่มีคอลัมน์:
`student_id, grade_level, gpa, failed_subjects, attendance_pct, late_count, discipline_points, midterm_avg, final_avg`
(ถ้ามีคอลัมน์ `label` จะโชว์เมตริกเปรียบเทียบด้วย)
""")

uploaded = st.file_uploader('อัปโหลดไฟล์นักเรียน (.csv)', type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('ตัวอย่างข้อมูล:')
    st.dataframe(df.head(20))

    if model is None:
        st.warning('ยังไม่มีโมเดลที่ฝึกแล้ว โปรดรัน train.py ก่อน หรือใส่ models/model.joblib')
    else:
        try:
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = None
        needed_cols = (meta['numeric_features'] + meta['categorical_features']) if meta else                       ['gpa','failed_subjects','attendance_pct','late_count','discipline_points','midterm_avg','final_avg','grade_level']
        for c in needed_cols:
            if c not in df.columns:
                df[c] = np.nan

        X = df[needed_cols]
        proba = model.predict_proba(X)[:,1]
        df_out = df.copy()
        df_out['risk_score'] = proba

        st.sidebar.header('ตั้งค่า')
        thr = st.sidebar.slider('Threshold แบ่งกลุ่มเสี่ยง', 0.0, 1.0, 0.5, 0.01)

        def tag(p):
            if p >= max(thr, 0.8):
                return 'HIGH'
            elif p >= thr:
                return 'MED'
            else:
                return 'LOW'

        df_out['risk_band'] = df_out['risk_score'].apply(tag)
        df_sorted = df_out.sort_values('risk_score', ascending=False)

        if 'label' in df_sorted.columns:
            from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
            try:
                y = df_sorted['label'].astype(int)
                auc = roc_auc_score(y, df_sorted['risk_score'])
                y_pred = (df_sorted['risk_score'] >= thr).astype(int)
                cm = confusion_matrix(y, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
                st.subheader('เมตริก (เมื่อมี label)')
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('ROC AUC', f"{auc:.3f}")
                c2.metric('Precision', f"{prec:.3f}")
                c3.metric('Recall', f"{rec:.3f}")
                c4.metric('F1', f"{f1:.3f}")
                st.write('Confusion Matrix:')
                st.write(pd.DataFrame(cm, index=['True 0','True 1'], columns=['Pred 0','Pred 1']))
            except Exception as e:
                st.info(f'คำนวณเมตริกไม่สำเร็จ: {e}')

        st.subheader('รายชื่อนักเรียนจัดอันดับตามความเสี่ยง')
        st.dataframe(df_sorted)

        csv = df_sorted.to_csv(index=False).encode('utf-8-sig')
        st.download_button('ดาวน์โหลดผล (CSV)', data=csv, file_name='student_risk_scored.csv', mime='text/csv')

        st.subheader("แดชบอร์ดผู้บริหาร – สรุปตามระดับชั้น")
        band_choice = st.selectbox("นิยามผู้มีความเสี่ยง", ["High เท่านั้น", "Medium+High"])
        if band_choice == "High เท่านั้น":
            risk_mask = df_out["risk_band"] == "HIGH"
        else:
            risk_mask = df_out["risk_band"].isin(["HIGH", "MED"])
        import re
        by_grade = (
            df_out.assign(is_risk=risk_mask)
            .groupby("grade_level")
            .agg(total=("student_id","count"), at_risk=("is_risk","sum"))
            .assign(pct_at_risk=lambda d: (d["at_risk"]/d["total"]*100).round(1))
            .reset_index()
        )
        def grade_key(val):
            m = re.search(r"(\d+)", str(val))
            return int(m.group(1)) if m else 999
        by_grade = by_grade.sort_values("grade_level", key=lambda s: s.map(grade_key))

        c1, c2 = st.columns([2, 3])
        with c1:
            st.caption("ตารางสรุปต่อระดับชั้น")
            st.dataframe(by_grade)
        with c2:
            st.caption("จำนวนผู้มีความเสี่ยงต่อระดับชั้น")
            st.bar_chart(by_grade.set_index("grade_level")["at_risk"])
else:
    st.info('โปรดอัปโหลดไฟล์ CSV เพื่อเริ่มประมวลผล')
