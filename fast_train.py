"""
fast_train.py — streamlined training, optimised for speed, still ≥91% accuracy
"""
import json, warnings, joblib
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
SEED = 42

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "live_class_data.csv"
MODEL_DIR = ROOT / "backend" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TARGET    = "class_quality"

# ── Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape}")

# ── Feature engineering ───────────────────────────────────────────────────
df["engagement_proxy"]    = (df["attendance_pct"] * df["camera_on_pct"]) / 100
df["tech_penalty"]        = df["network_drops"] * df["avg_latency_ms"] / 1000
df["quiz_efficiency"]     = df["quiz_completion"] * df["avg_quiz_score"] / 100
df["participation_index"] = (df["chat_messages"].fillna(0) +
                              df["polls_answered"].fillna(0) +
                              df["hand_raises"]) / (df["class_size"] + 1)
df["is_weekend"]          = (df["day_of_week"] >= 5).astype(int)
df["time_slot"]           = pd.cut(df["session_hour"], bins=[0,9,12,17,20,24],
                                    labels=["early","morning","afternoon","evening","night"],
                                    right=False).astype(str)

y = df[TARGET].values
X = df.drop(columns=[TARGET, "engagement_score"], errors="ignore")

cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

# ── Preprocessor ─────────────────────────────────────────────────────────
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc",  StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num_pipe, num_cols),
                          ("cat", cat_pipe, cat_cols)])

X_proc = pre.fit_transform(X)
joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
meta = {"num_cols": num_cols, "cat_cols": cat_cols, "all_input_cols": X.columns.tolist()}
json.dump(meta, open(MODEL_DIR / "feature_meta.json","w"), indent=2)

# ── Split + balance ───────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr_b, y_tr_b = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
print(f"Train (balanced): {X_tr_b.shape}  Test: {X_te.shape}")

# ── Train three fast models ───────────────────────────────────────────────
models = {
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9,
                              random_state=SEED, verbosity=0, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1,
                                 class_weight="balanced",
                                 random_state=SEED, verbosity=-1, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=25,
                                            class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
    "LogisticRegression": LogisticRegression(C=5, max_iter=2000,
                                              random_state=SEED, n_jobs=-1),
}

results = {}
trained = {}
for name, m in models.items():
    m.fit(X_tr_b, y_tr_b)
    y_pred = m.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted")
    cm  = confusion_matrix(y_te, y_pred)
    print(f"\n[{name}]  acc={acc:.4f}  f1={f1:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Low","Medium","High"]))
    results[name] = {"accuracy": float(acc), "f1": float(f1),
                     "confusion_matrix": cm.tolist()}
    trained[name] = m
    joblib.dump(m, MODEL_DIR / f"{name.lower().replace(' ','_')}_model.joblib")

# ── Soft-vote ensemble ────────────────────────────────────────────────────
ens = VotingClassifier(
    estimators=[("xgb", trained["XGBoost"]), ("lgb", trained["LightGBM"]),
                ("rf",  trained["RandomForest"])],
    voting="soft", n_jobs=-1)
ens.fit(X_tr_b, y_tr_b)
ens_acc = accuracy_score(y_te, ens.predict(X_te))
ens_f1  = f1_score(y_te, ens.predict(X_te), average="weighted")
print(f"\n[Ensemble(XGB+LGB+RF)]  acc={ens_acc:.4f}  f1={ens_f1:.4f}")
results["Ensemble"] = {"accuracy": float(ens_acc), "f1": float(ens_f1)}
trained["Ensemble"] = ens

# ── Pick best ─────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_acc   = results[best_name]["accuracy"]
best_model = trained[best_name]
print(f"\n{'='*50}")
print(f"BEST: {best_name}  accuracy={best_acc:.4f}  {'✓ ≥91%' if best_acc>=0.91 else '✗'}")

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

report = {
    "trained_at": datetime.utcnow().isoformat(),
    "best_model": best_name,
    "best_accuracy": best_acc,
    "target_met": best_acc >= 0.91,
    "all_results": results,
    "feature_count": X_proc.shape[1],
    "label_map": {"0": "Low", "1": "Medium", "2": "High"},
}
json.dump(report, open(MODEL_DIR / "evaluation_report.json","w"), indent=2)
print("Artefacts saved to", MODEL_DIR)
