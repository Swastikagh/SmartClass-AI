"""
Generate a realistic synthetic live-class monitoring dataset.
Replace this with your real CSV loader — the pipeline is compatible with any CSV
that contains class-level features and a target label.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)
N = 5000

def generate():
    # ── Core class metadata ──────────────────────────────────────────────────
    class_sizes   = np.random.randint(10, 80, N)
    duration_min  = np.random.choice([30, 45, 60, 90, 120], N)
    session_hour  = np.random.randint(8, 22, N)
    day_of_week   = np.random.randint(0, 7, N)          # 0=Mon … 6=Sun
    subject       = np.random.choice(["Math","Science","English",
                                       "History","Coding","Art"], N)
    grade_level   = np.random.randint(1, 13, N)
    teacher_exp   = np.random.randint(0, 31, N)         # years

    # ── Engagement signals ───────────────────────────────────────────────────
    attendance_pct = np.clip(np.random.normal(78, 15, N), 10, 100)
    chat_msgs      = np.random.poisson(25, N)
    polls_answered = np.random.poisson(3, N)
    hand_raises    = np.random.poisson(8, N)
    screen_shares  = np.random.randint(0, 10, N)
    avg_response_s = np.clip(np.random.exponential(5, N), 1, 60)
    camera_on_pct  = np.clip(np.random.normal(55, 20, N), 0, 100)
    mic_on_pct     = np.clip(np.random.normal(30, 15, N), 0, 100)

    # ── Technical / platform signals ────────────────────────────────────────
    network_drops  = np.random.poisson(2, N)
    avg_latency_ms = np.clip(np.random.exponential(80, N), 10, 500)
    reconnects     = np.random.poisson(1, N)
    platform       = np.random.choice(["Zoom","Teams","Meet","Custom"], N)

    # ── Assessment signals ───────────────────────────────────────────────────
    quiz_completion = np.clip(np.random.normal(72, 18, N), 0, 100)
    avg_quiz_score  = np.clip(np.random.normal(68, 15, N), 0, 100)
    assignment_sub  = np.clip(np.random.normal(75, 20, N), 0, 100)

    # ── Historical / rolling ─────────────────────────────────────────────────
    prev_class_score = np.clip(np.random.normal(70, 12, N), 0, 100)
    streak_good      = np.random.randint(0, 10, N)      # consecutive good sessions

    # ── Composite engagement score (ground-truth proxy) ─────────────────────
    engagement_raw = (
        0.20 * attendance_pct
        + 0.15 * (chat_msgs / 50 * 100)
        + 0.10 * (polls_answered / 5 * 100)
        + 0.10 * (hand_raises / 15 * 100)
        + 0.10 * camera_on_pct
        + 0.10 * quiz_completion
        + 0.10 * avg_quiz_score
        + 0.05 * assignment_sub
        + 0.05 * prev_class_score
        + 0.03 * (streak_good / 9 * 100)
        + 0.02 * (teacher_exp / 30 * 100)
        - 0.05 * (network_drops / 5 * 100)
        - 0.05 * (avg_latency_ms / 500 * 100)
    )
    noise = np.random.normal(0, 6, N)
    engagement_raw = np.clip(engagement_raw + noise, 0, 100)

    # ── Target: class quality label (3-class) ────────────────────────────────
    # 0 = Low  (<55)  |  1 = Medium  (55-75)  |  2 = High  (>75)
    quality = np.where(engagement_raw >= 75, 2,
               np.where(engagement_raw >= 55, 1, 0))

    df = pd.DataFrame({
        "class_size":        class_sizes,
        "duration_min":      duration_min,
        "session_hour":      session_hour,
        "day_of_week":       day_of_week,
        "subject":           subject,
        "grade_level":       grade_level,
        "teacher_experience_years": teacher_exp,
        "attendance_pct":    attendance_pct.round(2),
        "chat_messages":     chat_msgs,
        "polls_answered":    polls_answered,
        "hand_raises":       hand_raises,
        "screen_shares":     screen_shares,
        "avg_response_sec":  avg_response_s.round(2),
        "camera_on_pct":     camera_on_pct.round(2),
        "mic_on_pct":        mic_on_pct.round(2),
        "network_drops":     network_drops,
        "avg_latency_ms":    avg_latency_ms.round(1),
        "reconnects":        reconnects,
        "platform":          platform,
        "quiz_completion":   quiz_completion.round(2),
        "avg_quiz_score":    avg_quiz_score.round(2),
        "assignment_submission_pct": assignment_sub.round(2),
        "prev_class_score":  prev_class_score.round(2),
        "streak_good_sessions": streak_good,
        "engagement_score":  engagement_raw.round(2),   # continuous (bonus column)
        "class_quality":     quality,                    # TARGET  0/1/2
    })

    # Inject ~3 % missing values realistically
    for col in ["chat_messages","polls_answered","avg_quiz_score","assignment_submission_pct"]:
        mask = np.random.rand(N) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "live_class_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  ({len(df)} rows × {df.shape[1]} cols)")
    print(df["class_quality"].value_counts().sort_index()
          .rename({0:"Low",1:"Medium",2:"High"}).to_string())
