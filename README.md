# 🎓 Live Class Quality Monitor

> An end-to-end AI system that predicts the quality of live online class sessions in real time using Machine Learning — built for educators to detect low-engagement sessions instantly and act on it.

---

## 📌 Overview

Live Class Quality Monitor is a **production-ready ML system** that takes in 24+ real-time class session parameters — attendance, quiz scores, camera usage, chat activity, network quality — and instantly predicts whether a session is **Good Quality** or **Needs Improvement**.

It goes beyond just a model — it's a complete pipeline from raw data to a deployed REST API with a live monitoring dashboard.

> 🏷 **Aligned with UNESCO SDG 4 — Quality Education**  
> Empowers educators with real-time AI insights to improve learning outcomes at scale.

---

## ✨ Features

- ✅ **92% Model Accuracy** — exceeds the 91% production threshold
- ⚡ **< 500ms API Response** — real-time inference
- 🤖 **4 ML Models Compared** — XGBoost, LightGBM, Random Forest, Logistic Regression
- 🔌 **FastAPI REST Backend** — single, batch, and CSV prediction endpoints
- 📊 **Live Dashboard** — no build step, open in browser directly
- 🐳 **Dockerized** — runs anywhere with one command
- ☁️ **Render Deployment** — one-click cloud deploy via `render.yaml`
- 🧪 **15 Automated Tests** — full pytest coverage
- 📋 **Prediction Logging** — tracks every inference with risk level + recommendations

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **ML & Data** | Scikit-learn, XGBoost, LightGBM, Pandas, NumPy, imbalanced-learn |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest |
| **Containerization** | Docker, Docker Compose |
| **Deployment** | Render |
| **Frontend** | HTML, CSS, JavaScript |
| **Logging** | python-json-logger |

---

## 📁 Project Structure

```
live-class-monitor/
├── backend/
│   ├── main.py                        # FastAPI app entry point
│   ├── requirements.txt               # Python dependencies
│   ├── Dockerfile                     # Docker container config
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # All API endpoints
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   ├── services/
│   │   │   └── inference.py           # Model loading + prediction logic
│   │   ├── utils/
│   │   │   └── logger.py              # Structured JSON logging
│   │   └── models/                    # Trained .joblib model files
│   │       ├── best_model.joblib
│   │       ├── preprocessor.joblib
│   │       ├── feature_meta.json
│   │       └── evaluation_report.json
│   └── tests/
│       └── test_api.py                # 15 unit + integration tests
├── frontend/
│   └── dashboard.html                 # Standalone monitoring dashboard
├── scripts/
│   ├── train_91.py                    # Main training pipeline (≥91% accuracy)
│   └── generate_dataset.py            # Dataset generation script
├── data/
│   └── live_class_data.csv            # Training dataset (8,000 rows)
├── docker-compose.yml
├── render.yaml                        # Render deployment config
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip
- Docker (optional)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/live-class-monitor.git
cd live-class-monitor
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the Model
```bash
cd ..
python scripts/train_91.py
```
> Saves `best_model.joblib`, `preprocessor.joblib`, and `evaluation_report.json` into `backend/app/models/`

### 4. Start the API
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- API live at → `http://localhost:8000`
- Interactive docs → `http://localhost:8000/docs`

### 5. Open the Dashboard
Open `frontend/dashboard.html` directly in your browser — no build step needed.

### 6. Run Tests
```bash
cd backend
python -m pytest tests/test_api.py -v
```
Expected: **15 passed ✓**

---

## 🐳 Docker

```bash
# Backend only
cd backend
docker build -t class-monitor-api .
docker run -p 8000:8000 class-monitor-api

# Full stack
docker-compose up --build
```

---

## 🔌 API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API + model status, accuracy, uptime |
| `POST` | `/predict` | Single class session prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `POST` | `/predict/csv` | Upload CSV for bulk inference |
| `GET` | `/stats` | Aggregate prediction statistics |
| `GET` | `/logs` | Recent prediction history |
| `GET` | `/model/info` | Model metadata and all results |

### Sample Request
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "class_size": 35,
    "duration_min": 60,
    "session_hour": 10,
    "day_of_week": 1,
    "subject": "Math",
    "grade_level": 9,
    "teacher_experience_years": 8,
    "attendance_pct": 82.5,
    "chat_messages": 28,
    "polls_answered": 4,
    "hand_raises": 9,
    "screen_shares": 2,
    "avg_response_sec": 4.2,
    "camera_on_pct": 65.0,
    "mic_on_pct": 30.0,
    "network_drops": 1,
    "avg_latency_ms": 65.0,
    "reconnects": 0,
    "platform": "Zoom",
    "quiz_completion": 78.0,
    "avg_quiz_score": 72.5,
    "assignment_submission_pct": 80.0,
    "prev_class_score": 73.0,
    "streak_good_sessions": 3
  }'
```

### Sample Response
```json
{
  "prediction": 1,
  "label": "GoodClass",
  "confidence": 0.8731,
  "probability_good": 0.8731,
  "probability_needs": 0.1269,
  "risk_flag": false,
  "risk_level": "LOW",
  "engagement_signals": {
    "engagement_proxy": 53.63,
    "participation_index": 1.19,
    "tech_penalty": 0.065,
    "quiz_efficiency": 56.55
  },
  "recommendations": [
    "✅ Class is performing well — keep up the engagement strategies!"
  ],
  "model_version": "LogisticRegression",
  "latency_ms": 4.2
}
```

---

## 🧠 Model Performance

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| **Logistic Regression** ⭐ | **92.0%** | 92.0% | 97.9% |
| XGBoost | 90.1% | 90.1% | 97.0% |
| Ensemble (XGB+LGB+RF) | 90.0% | 90.0% | 96.3% |
| LightGBM | 89.5% | 89.5% | 96.8% |
| Random Forest | 86.8% | 86.8% | 94.6% |

### Target Variable
Binary classification:
- `0` → **NeedsImprovement** — session engagement below threshold
- `1` → **GoodClass** — session engagement above threshold

### Key Engineered Features
| Feature | Formula |
|---|---|
| `engagement_proxy` | (attendance × camera_on) / 100 |
| `tech_penalty` | network_drops × avg_latency / 1000 |
| `quiz_efficiency` | quiz_completion × avg_quiz_score / 100 |
| `participation_index` | (chat + polls + hand_raises) / class_size |

---

## ☁️ Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-reads `render.yaml` — click **Deploy**

The `render.yaml` handles everything:
- Runtime: Python
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/v1/health`

---

## 📊 Using Your Own Dataset

Replace the synthetic data with your real CSV:

```python
# In scripts/train_91.py
df = pd.read_csv("path/to/your/real_data.csv")
TARGET = "your_target_column"
```

The pipeline auto-handles missing values, encoding, and scaling.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

## 👩‍💻 Author

**Swastika Chakraborty**  
B.Tech CSE — Institute of Engineering & Management, Kolkata  
[LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [GitHub](https://github.com/YOUR_USERNAME)
