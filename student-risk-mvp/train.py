import os, json, re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

DATA_PATH = 'data/students_train_synth_v1.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
META_PATH = os.path.join(MODEL_DIR, 'meta.json')

NUMERIC_FEATURES = ['gpa','failed_subjects','attendance_pct','late_count','discipline_points','midterm_avg','final_avg']
CATEGORICAL_FEATURES = ['grade_level']
TARGET_COL = 'label'

HARD_BOUNDS = {
    'gpa': (0,4),
    'attendance_pct': (0,100),
    'midterm_avg': (0,100),
    'final_avg': (0,100),
    'failed_subjects': (0,None),
    'late_count': (0,None),
    'discipline_points': (0,None),
}

def standardize_grade_level(s):
    if pd.isna(s): return np.nan
    import re
    txt = str(s)
    m = re.search(r'(\d+)', txt)
    if not m: return np.nan
    n = int(m.group(1))
    if 7 <= n <= 12: n -= 6
    if 1 <= n <= 6: return f'ม.{n}'
    return np.nan

def validate_df(df, is_training=True):
    problems = []
    required = ['student_id','grade_level','gpa','failed_subjects','attendance_pct','late_count','discipline_points','midterm_avg','final_avg']
    if is_training: required += ['label']
    missing = [c for c in required if c not in df.columns]
    if missing: problems.append(f'ขาดคอลัมน์: {missing}')
    if 'student_id' in df and df['student_id'].duplicated().sum() > 0:
        problems.append(f'พบ student_id ซ้ำ {df['"'"'student_id'"'"'].duplicated().sum()} แถว')
    if 'grade_level' in df:
        df['grade_level'] = df['grade_level'].apply(standardize_grade_level)
        if df['grade_level'].isna().sum() > 0:
            problems.append('grade_level แปลงไม่ได้บางแถว')
    for col,(lo,hi) in HARD_BOUNDS.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().sum() or (lo is not None and (df[col] < lo).sum()) or (hi is not None and (df[col] > hi).sum()):
                problems.append(f'{col} มีค่าว่าง/นอกช่วง ตรวจสอบอีกครั้ง')
    if is_training and 'label' in df and (~df['label'].isin([0,1])).sum() > 0:
        problems.append('label ต้องเป็น 0/1')
    return problems, df

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    problems, df = validate_df(df, is_training=True)
    if problems:
        print('พบปัญหาข้อมูล:'); [print(' -', p) for p in problems]
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_pipe, NUMERIC_FEATURES),('cat', categorical_pipe, CATEGORICAL_FEATURES)])
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    pipe = Pipeline([('prep', preprocessor),('clf', clf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    print(f'ROC AUC: {auc:.3f}')
    print(classification_report(y_test, y_pred, digits=3))
    joblib.dump(pipe, MODEL_PATH)
    with open(META_PATH,'w',encoding='utf-8') as f:
        import json; json.dump({'numeric_features':NUMERIC_FEATURES,'categorical_features':CATEGORICAL_FEATURES,'target':TARGET_COL,'roc_auc':float(auc)}, f, ensure_ascii=False, indent=2)
    print(f'Saved model to {MODEL_PATH}')
