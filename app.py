import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Study AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 1. 가상 학습 데이터 생성 및 모델 학습
# -----------------------------
@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)
    n = 3000

    previous_score = rng.uniform(35, 95, n)
    study_hours = np.clip(rng.gamma(shape=2.2, scale=7, size=n), 0, 50)
    correct_rate = np.clip(
        previous_score * 0.65
        + (study_hours / 50) * 100 * 0.35
        + rng.normal(0, 12, n),
        0, 100
    )
    review_count = np.clip(rng.poisson(3, n), 0, 10)

    study_factor = (
        0.05
        + 0.35 * (study_hours / 50)
        + 0.35 * (correct_rate / 100)
        + 0.25 * (review_count / 10)
    )
    improvement = (
        (100 - previous_score) * 0.22 * study_factor
        + rng.normal(0, 2.3, n)
    )
    next_score = np.clip(previous_score + improvement, 0, 100)

    data = pd.DataFrame({
        "study_hours": study_hours,
        "previous_score": previous_score,
        "correct_rate": correct_rate,
        "review_count": review_count,
        "next_score": next_score
    })

    features = ["study_hours", "previous_score", "correct_rate", "review_count"]
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["next_score"],
        test_size=0.2, random_state=42
    )

    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        }

    selected_name = min(metrics, key=lambda name: metrics[name]["MAE"])
    return data, models, metrics, selected_name, features

data, models, metrics, selected_name, features = train_models()
model = models[selected_name]

# -----------------------------
# 2. 화면 스타일
# -----------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f5f9ff 0%, #ffffff 42%); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 24px;
        background: linear-gradient(115deg, #dcecff, #f0f6ff 60%, #e6f7ff);
        border: 1px solid #d8e8fb; margin-bottom: 1.2rem;
    }
    .hero h1 { color: #183b68; margin: 0; font-size: 2.3rem; }
    .hero p { color: #496582; margin: .45rem 0 0 0; font-size: 1rem; }
    .small-note { color: #718096; font-size: .85rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5edf7;
        padding: 14px 16px; border-radius: 16px;
    }
    .section-card {
        background: white; border: 1px solid #e5edf7;
        border-radius: 18px; padding: 18px 20px; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. 사이드바 메뉴
# -----------------------------
st.sidebar.markdown("# 📘 Study AI")
st.sidebar.caption("AI 기반 개인 맞춤형 학습 도우미")
page = st.sidebar.radio(
    "메뉴",
    ["홈 대시보드", "성적 예측", "공부 피드백", "학습 계획 추천", "모델 연구실"],
    index=0
)
st.sidebar.divider()
st.sidebar.caption("Study AI 1.0 · Synthetic Data Prototype")

# -----------------------------
# 4. 공통 입력값
# -----------------------------
with st.sidebar.expander("내 학습 정보 입력", expanded=True):
    previous_score = st.number_input("이전 시험 성적", 0.0, 100.0, 75.0, 1.0)
    study_hours = st.number_input("최근 7일 공부시간 (시간)", 0.0, 50.0, 18.0, 1.0)
    correct_rate = st.number_input("최근 문제 정답률 (%)", 0.0, 100.0, 75.0, 1.0)
    review_count = st.number_input("복습 횟수", 0, 10, 4, 1)

input_df = pd.DataFrame([{
    "study_hours": study_hours,
    "previous_score": previous_score,
    "correct_rate": correct_rate,
    "review_count": review_count
}])
predicted_score = float(np.clip(model.predict(input_df)[0], 0, 100))
change = predicted_score - previous_score

# -----------------------------
# 5. 페이지별 화면
# -----------------------------
if page == "홈 대시보드":
    st.markdown("""
    <div class="hero">
        <h1>Study AI 📘</h1>
        <p>나의 학습을 분석하고, 다음 성장을 설계하다.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("오늘의 학습 대시보드")
    st.write("왼쪽 메뉴에서 학습 정보를 입력하면 각 기능에 결과가 반영돼.")

    c1, c2, c3 = st.columns(3)
    c1.metric("예측 성적", f"{predicted_score:.1f}점", f"{change:+.1f}점")
    c2.metric("최근 7일 공부시간", f"{study_hours:.0f}시간")
    c3.metric("문제 정답률", f"{correct_rate:.0f}%")

    st.markdown("### Study AI의 핵심 기능")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="section-card"><h3>📈 성적 예측</h3><p>학습 데이터를 바탕으로 다음 시험 성적을 추정해.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section-card"><h3>🧠 공부 피드백</h3><p>공부시간, 정답률, 복습 습관에서 개선점을 찾아.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="section-card"><h3>🗓️ 학습 계획</h3><p>목표와 현재 학습 상태를 바탕으로 계획을 제안해.</p></div>', unsafe_allow_html=True)

    st.info("왼쪽 메뉴에서 '내 학습 정보 입력'을 바꾸고 각 기능을 확인해 봐.")

elif page == "성적 예측":
    st.markdown('<div class="hero"><h1>📈 성적 예측</h1><p>입력한 학습 특성을 학습된 회귀 모델에 전달해 다음 시험 점수를 추정해.</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("AI 예측 성적", f"{predicted_score:.1f}점")
        st.metric("이전 성적 대비 변화", f"{change:+.1f}점")
        st.caption(f"사용 모델: {selected_name}")
    with c2:
        st.markdown("#### 입력한 학습 정보")
        st.dataframe(pd.DataFrame({
            "항목": ["이전 성적", "최근 7일 공부시간", "문제 정답률", "복습 횟수"],
            "입력값": [f"{previous_score:.0f}점", f"{study_hours:.0f}시간", f"{correct_rate:.0f}%", f"{review_count}회"]
        }), hide_index=True, use_container_width=True)
    st.warning("이 결과는 직접 설계한 가상 데이터로 학습한 모델의 예측값이야. 실제 시험 성적을 보장하거나 실제 학생 성적 예측 정확도를 의미하지 않아.")
    st.markdown("#### 예측 결과 해석")
    if change > 2:
        st.success("가상 학습 데이터의 패턴에서는 이전 성적보다 높은 점수가 예측됐어.")
    elif change < -2:
        st.warning("가상 학습 데이터의 패턴에서는 이전 성적보다 낮은 점수가 예측됐어.")
    else:
        st.info("가상 학습 데이터의 패턴에서는 이전 성적과 비슷한 점수가 예측됐어.")

elif page == "공부 피드백":
    st.markdown('<div class="hero"><h1>🧠 공부 피드백</h1><p>현재 입력한 학습 습관을 기준으로 실천 가능한 개선점을 제안해.</p></div>', unsafe_allow_html=True)
    feedback = []
    if study_hours < 7:
        feedback.append(("공부시간", "최근 7일 공부시간이 적은 편이야. 짧더라도 규칙적인 학습 시간을 정해보자."))
    elif study_hours < 20:
        feedback.append(("공부시간", "일정한 학습 시간이 확보되어 있어. 집중이 잘 되는 시간대를 유지해 보자."))
    else:
        feedback.append(("공부시간", "학습 시간이 충분한 편이야. 시간 자체보다 학습의 질과 휴식 균형을 점검해 보자."))

    if correct_rate < 50:
        feedback.append(("정답률", "개념 이해와 기본 문제 풀이를 먼저 점검하고, 오답 원인을 분류해 보자."))
    elif correct_rate < 75:
        feedback.append(("정답률", "틀린 문제를 유형별로 정리하고 며칠 뒤 다시 풀어보자."))
    else:
        feedback.append(("정답률", "정답률이 높은 편이야. 풀이 과정을 설명하거나 응용 문제에 도전해 보자."))

    if review_count == 0:
        feedback.append(("복습", "아직 복습 기록이 없어. 학습한 날 짧게 핵심 내용을 되짚어 보자."))
    elif review_count < 4:
        feedback.append(("복습", "복습을 하고 있어. 간격을 두고 다시 확인하는 방식도 시도해 보자."))
    else:
        feedback.append(("복습", "복습을 꾸준히 하고 있어. 단순 반복뿐 아니라 스스로 설명하는 복습도 해보자."))

    for title, message in feedback:
        st.markdown(f'<div class="section-card"><h4>{title}</h4><p>{message}</p></div>', unsafe_allow_html=True)

    st.caption("피드백은 입력값에 기반한 규칙형 조언이며, 별도의 생성형 AI가 작성한 답변은 아니야.")

elif page == "학습 계획 추천":
    st.markdown('<div class="hero"><h1>🗓️ 학습 계획 추천</h1><p>현재 학습 상태와 목표를 바탕으로 예시 주간 계획을 구성해.</p></div>', unsafe_allow_html=True)
    target_score = st.slider("목표 성적", 50, 100, 90, 5)
    available_hours = st.slider("앞으로 일주일 동안 확보할 공부시간", 3, 35, 14)
    gap = max(0, target_score - previous_score)

    if gap >= 20:
        focus = "개념 정리와 기본 유형 학습"
    elif gap >= 10:
        focus = "취약 유형 보완과 오답 복습"
    elif gap > 0:
        focus = "실수 줄이기와 응용 문제 연습"
    else:
        focus = "현재 학습 수준 유지와 심화 학습"

    st.metric("목표까지의 점수 차이", f"{gap:.0f}점")
    st.write(f"**추천 학습 초점:** {focus}")

    days = ["월", "화", "수", "목", "금", "토", "일"]
    weights = [1, 1, 1, 1, 1, 1.5, 1.5]
    total_weight = sum(weights)
    daily = [round(available_hours * w / total_weight, 1) for w in weights]
    # 반올림으로 인한 합계 오차를 마지막 날에 보정
    daily[-1] = round(available_hours - sum(daily[:-1]), 1)

    plan = pd.DataFrame({
        "요일": days,
        "추천 공부시간": daily,
        "학습 내용": [
            "핵심 개념 정리", "기본 문제 풀이", "오답 분석",
            "취약 유형 보완", "주중 내용 복습", "실전 문제 풀이", "주간 점검 및 휴식"
        ]
    })
    st.dataframe(plan, hide_index=True, use_container_width=True)
    st.caption("현재 버전은 입력값에 따른 규칙 기반 계획 예시야. 실제 시간표와 과목별 시험 범위를 반영하는 기능은 다음 개발 단계에서 추가할 수 있어.")

elif page == "모델 연구실":
    st.markdown('<div class="hero"><h1>🔬 모델 연구실</h1><p>Study AI가 어떤 데이터로 학습했고, 두 모델의 성능이 어떻게 다른지 확인해.</p></div>', unsafe_allow_html=True)
    st.subheader("가상 데이터 미리보기")
    st.dataframe(data.head(12).round(1), hide_index=True, use_container_width=True)
    st.caption(f"전체 데이터: {len(data):,}개 행 · 입력 특성 4개 · 목표값 1개")

    st.subheader("모델 성능 비교")
    metric_df = pd.DataFrame([
        {"모델": name, "MAE": values["MAE"], "R²": values["R2"]}
        for name, values in metrics.items()
    ])
    st.dataframe(metric_df.round(3), hide_index=True, use_container_width=True)
    st.bar_chart(metric_df.set_index("모델")[["MAE"]])
    st.success(f"현재 테스트 데이터에서 MAE가 더 낮은 모델은 **{selected_name}**이야.")

    with st.expander("평가 지표 이해하기"):
        st.markdown("""
        - **MAE**: 예측값과 정답 사이의 절대 오차 평균. 낮을수록 좋아.
        - **R²**: 테스트 데이터의 변동을 모델이 얼마나 설명하는지 나타내는 지표.
        - **주의**: 이 평가는 생성 규칙으로 만든 가상 데이터에 대한 결과야. 실제 학생 집단에 대한 성능으로 해석하면 안 돼.
        """)

st.divider()
st.caption("Study AI 1.0 · 교육용 프로토타입 · 가상 데이터 기반")
