# -*- coding: utf-8 -*-
"""
🎓 수학 학습 진단 & YouTube 영상 추천 (Streamlit 앱)

실행 방법:
    streamlit run app.py

필요 파일(아래 7개)을 app.py 와 같은 폴더에 두거나,
앱 사이드바에서 직접 업로드하세요.
    - X_train.npy
    - y_train.npy
    - best_dkvmn_model.pth
    - math_concepts.csv
    - Mathematics Knowledge System Dataset.json
    - tag_to_concept_id.json
    - youtube_recommendations_final_844concepts.json
"""

import os
import re
import html
import tempfile

import numpy as np
import streamlit as st

import engine


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="수학 학습 진단 & 영상 추천",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 약간의 스타일 ──
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .concept-card {
        border: 1px solid #e6e6e6; border-radius: 14px;
        padding: 20px 22px; margin-bottom: 18px; background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .grade-badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        color: #fff; font-weight: 700; font-size: 0.82rem;
    }
    .meter-track {
        background: #eef0f2; border-radius: 999px; height: 12px;
        width: 100%; overflow: hidden; margin: 6px 0 4px 0;
    }
    .meter-fill { height: 100%; border-radius: 999px; }
    .vid-card {
        display: flex; gap: 12px; padding: 10px 12px; border-radius: 10px;
        background: #f8f9fb; margin-bottom: 8px; align-items: flex-start;
        text-decoration: none; border: 1px solid #eef0f2;
    }
    .vid-card:hover { background: #eef3fb; border-color: #cdd9f0; }
    .vid-rank {
        flex: 0 0 auto; background: #1a73e8; color:#fff; font-weight:700;
        width: 26px; height: 26px; border-radius: 6px; display:flex;
        align-items:center; justify-content:center; font-size:0.8rem; margin-top:2px;
    }
    .vid-meta { color:#5f6368; font-size:0.82rem; margin-top:2px; }
    .vid-title { color:#202124; font-weight:600; font-size:0.95rem; line-height:1.35; }
    .root-box {
        background:#fff6f0; border-left:4px solid #E67E22; border-radius:8px;
        padding:10px 14px; margin:10px 0; font-size:0.92rem;
    }
    .direct-box {
        background:#eef7f0; border-left:4px solid #2ECC71; border-radius:8px;
        padding:10px 14px; margin:10px 0; font-size:0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 파일 경로 관리 (로컬 폴더 자동 인식 + 업로드)
# =========================================================
SPEC = {
    "X": ("X_train.npy", "학생 풀이 기록 (문항)", ["npy"]),
    "Y": ("y_train.npy", "학생 풀이 기록 (정오답)", ["npy"]),
    "MODEL": ("best_dkvmn_model.pth", "DKVMN 학습 모델", ["pth", "pt"]),
    "CONCEPTS": ("math_concepts.csv", "개념명 목록", ["csv"]),
    "KNOWLEDGE": ("Mathematics Knowledge System Dataset.json", "지식 위계 그래프", ["json"]),
    "TAG_TO_CONCEPT": ("tag_to_concept_id.json", "태그→개념 매핑", ["json"]),
    "YOUTUBE": ("youtube_recommendations_final_844concepts.json", "YouTube 영상 매핑", ["json"]),
}


def resolve_paths():
    """로컬에 파일이 있으면 그 경로, 없으면 None. 업로드분은 세션에 저장된 임시경로 사용."""
    paths = {}
    for key, (fname, _label, _ext) in SPEC.items():
        up_key = f"uploaded_{key}"
        if up_key in st.session_state and st.session_state[up_key]:
            paths[key] = st.session_state[up_key]
        elif os.path.exists(fname):
            paths[key] = fname
        else:
            paths[key] = None
    return paths


def save_upload(key, uploaded_file):
    """업로드 파일을 임시폴더에 저장하고 경로를 세션에 보관."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    st.session_state[f"uploaded_{key}"] = tmp.name


# =========================================================
# 리소스 로딩 (캐시)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model_and_data(paths_tuple):
    """모델/데이터셋/리소스 로드. paths_tuple 은 캐시 키용 (정렬된 튜플)."""
    paths = dict(paths_tuple)

    if not engine.TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch 가 설치되어 있지 않습니다. 터미널에서 `pip install torch` 를 실행해 주세요."
        )

    engine.set_seed(42)
    device = engine.get_device()

    dataset = engine.KTDataset(paths["X"], paths["Y"])
    n_tags = int(np.max(dataset.x)) + 1

    model = engine.DKVMN(n_questions=n_tags, embed_dim=64, memory_size=50).to(device)
    import torch
    state = torch.load(paths["MODEL"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    resources = {
        "concept_name_map": engine.load_math_concepts(paths["CONCEPTS"]),
        "prereq_graph": {},
        "concept_info": {},
        "tag_to_concept": engine.load_tag_to_concept(paths["TAG_TO_CONCEPT"]),
        "youtube_map": engine.load_youtube_recommendations(paths["YOUTUBE"]),
    }
    resources["prereq_graph"], resources["concept_info"] = \
        engine.load_knowledge_graph(paths["KNOWLEDGE"])

    return model, dataset, resources, device, n_tags


# =========================================================
# 렌더링 헬퍼
# =========================================================
def render_meter(percent, color):
    return (f'<div class="meter-track"><div class="meter-fill" '
            f'style="width:{percent}%; background:{color};"></div></div>')


def render_grade_badge(grade_info):
    return (f'<span class="grade-badge" style="background:{grade_info["color"]};">'
            f'{grade_info["grade"]} · {grade_info["label"]}</span>')


def clean_title(title):
    """HTML 엔티티(&#39; 등) 정리."""
    return html.unescape(title or "")


def render_videos(videos):
    blocks = []
    for v in videos:
        rank = v.get("rank", "?")
        title = clean_title(v.get("title", ""))
        channel = v.get("channel", "")
        url = v.get("url", "#")
        duration = v.get("duration", "")
        views = v.get("view_count", 0)
        meta = f"📺 {channel}"
        if duration:
            meta += f" · ⏱️ {duration}"
        if views and views > 0:
            meta += f" · 👁️ {views:,}회"
        blocks.append(
            f'<a class="vid-card" href="{html.escape(url)}" target="_blank">'
            f'<div class="vid-rank">{rank}</div>'
            f'<div><div class="vid-title">{html.escape(title)}</div>'
            f'<div class="vid-meta">{html.escape(meta)}</div></div></a>'
        )
    return "".join(blocks)


def render_concept_card(item, index):
    g = item["grade"]
    st.markdown(f'<div class="concept-card">', unsafe_allow_html=True)

    # 헤더: 개념명 + 등급
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="font-size:1.12rem;font-weight:700;">{index}. {html.escape(item["concept_name"])}</div>'
        f'{render_grade_badge(g)}</div>',
        unsafe_allow_html=True,
    )

    # 이해도(모델 예측) + 정답률(실제) — 헷갈리지 않게 라벨 명확화
    st.markdown(
        f'<div style="color:#5f6368;font-size:0.85rem;margin-top:10px;">'
        f'이해도 <span title="AI 모델이 풀이 패턴을 분석해 예측한 종합 이해도예요." '
        f'style="cursor:help;">ⓘ</span></div>'
        f'{render_meter(g["percent"], g["color"])}'
        f'<div style="font-size:0.85rem;color:#5f6368;">'
        f'예측 이해도 <b>{g["percent"]}%</b> &nbsp;·&nbsp; '
        f'실제 정답률 {item["correct_text"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Root Cause / 직접 학습 안내
    if item["source_type"] == "root_cause" and item["root_cause"]:
        rc = item["root_cause"]
        rcg = item["root_cause_grade"]
        st.markdown(
            f'<div class="root-box">🔎 <b>이 개념이 약한 진짜 원인</b>은 '
            f'<b>「{html.escape(rc["concept_name"])}」</b>(으)로 보여요. '
            f'먼저 이 선행 개념(이해도 {rcg["percent"]}%)을 다지면 도움이 됩니다.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="direct-box">🎯 이 개념을 <b>직접</b> 복습하는 강의를 추천해요.</div>',
            unsafe_allow_html=True,
        )

    # 영상
    st.markdown('<div style="font-weight:700;margin:8px 0 6px 0;">📺 추천 영상</div>',
                unsafe_allow_html=True)
    st.markdown(render_videos(item["videos"]), unsafe_allow_html=True)

    # 상세 정보(접기) — 내부 ID 는 여기 숨김
    with st.expander("상세 정보 (선생님용)"):
        st.caption(
            f"tag_id = {item['tag_id']}  ·  concept_id = {item['concept_id']}  ·  "
            f"예측 이해도 = {item['mastery']:.4f}  ·  정답률 = {item['correct_rate']:.4f}  ·  "
            f"풀이 = {item['solve_count']}회"
        )
        if item["root_cause"]:
            st.caption(
                f"Root Cause: {item['root_cause']['concept_name']} "
                f"(concept_id={item['root_cause']['concept_id']}, "
                f"이해도={item['root_cause']['mastery']:.4f})"
            )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 사이드바 — 데이터 파일
# =========================================================
st.sidebar.title("⚙️ 데이터 설정")
st.sidebar.caption(
    "필요한 파일이 이 앱과 같은 폴더에 있으면 자동으로 인식합니다. "
    "없는 파일만 아래에서 업로드하세요."
)

paths = resolve_paths()
missing = [SPEC[k][1] for k, v in paths.items() if v is None]

with st.sidebar.expander("파일 업로드 / 상태 확인", expanded=bool(missing)):
    for key, (fname, label, ext) in SPEC.items():
        ok = paths[key] is not None
        st.markdown(f"{'✅' if ok else '⬜'} **{label}**  \n`{fname}`")
        if not ok:
            up = st.file_uploader(f"{label} 업로드", type=ext, key=f"up_{key}",
                                  label_visibility="collapsed")
            if up is not None:
                save_upload(key, up)
                st.rerun()

paths = resolve_paths()
all_ready = all(v is not None for v in paths.values())


# =========================================================
# 메인
# =========================================================
st.title("🎓 수학 학습 진단 & 영상 추천")
st.caption("학생의 풀이 기록을 분석해 약한 개념을 찾고, 그 원인이 되는 선행 개념까지 "
           "역추적해서 딱 맞는 강의를 추천해 드려요.")

if not all_ready:
    st.warning("왼쪽 사이드바에서 필요한 데이터 파일을 모두 준비해 주세요.")
    st.info("준비되지 않은 파일: " + ", ".join(missing))
    st.stop()

# 리소스 로드
try:
    paths_tuple = tuple(sorted(paths.items()))
    model, dataset, resources, device, n_tags = load_model_and_data(paths_tuple)
except Exception as e:
    st.error(f"데이터를 불러오는 중 문제가 발생했습니다.\n\n{e}")
    st.stop()

n_students = len(dataset)
n_mapped = len(resources["youtube_map"])

c1, c2, c3 = st.columns(3)
c1.metric("전체 학생 수", f"{n_students:,}명")
c2.metric("전체 개념 수", f"{n_mapped:,}개")
c3.metric("분석 모델", "DKVMN")

st.divider()

# 학생 선택
col_a, col_b = st.columns([3, 1])
with col_a:
    student_idx = st.number_input(
        f"학생 번호 (0 ~ {n_students - 1})",
        min_value=0, max_value=n_students - 1, value=0, step=1,
        help="선생님/관리자가 학생 ID에 해당하는 번호를 입력해 진단합니다.",
    )
    st.caption("💡 시연용 번호: 우수 학습자는 69번, 보충이 많이 필요한 학습자는 33번")
with col_b:
    st.write("")
    st.write("")
    run = st.button("진단 시작", type="primary", use_container_width=True)

if not run:
    st.info("학생 번호를 입력하고 **진단 시작**을 눌러주세요.")
    st.stop()

# 진단 실행
with st.spinner("학생의 풀이 기록을 분석하는 중..."):
    q_valid, r_valid, p_valid = engine.run_inference(model, dataset, int(student_idx), device)
    if len(q_valid) == 0:
        st.warning("이 학생은 분석할 풀이 기록이 없어요. 다른 학생 번호를 선택해 주세요.")
        st.stop()
    out = engine.diagnose(q_valid, r_valid, p_valid, resources,
                          weak_threshold=0.3, top_k_videos=2)

# ── 요약 헤더 ──
st.markdown(f"### 📋 {int(student_idx)}번 학생 진단 결과")
og = out["overall_grade"]
sc1, sc2, sc3 = st.columns(3)
sc1.metric("푼 문제 수", f"{out['total_records']}문제")
sc2.metric("학습한 개념 수", f"{out['unique_matched_tags']}개")
sc3.metric("전반 이해도", f"{og['percent']}% ({og['grade']})")

with st.expander("💡 '예측 이해도'와 '실제 정답률'이 다른 이유"):
    st.markdown(
        "**예측 이해도**는 AI 모델이 풀이 패턴·선행 개념·문제 난이도까지 종합해 "
        "추정한 실제 실력이에요. **실제 정답률**은 푼 문제 중 맞춘 비율이고요.\n\n"
        "예를 들어 5문제를 다 맞혔어도(정답률 100%), 표본이 적거나 선행 개념이 "
        "약하면 AI는 아직 완벽히 마스터했다고 보긴 이르다고 판단해 이해도를 "
        "더 낮게 줄 수 있어요. 학습 진단에는 **예측 이해도**가 더 정확합니다."
    )

n_show = len(out["weak_items"])

# ── 성취도 좋은 학생: 취약 개념 없음 ──
if out["n_weak_all"] == 0:
    st.success("🎉 훌륭해요! 기준치 이하로 약한 개념이 하나도 없습니다. 아주 잘하고 있어요.")
    if out["strong_items"]:
        st.markdown("#### 💪 특히 잘하는 개념")
        st.caption("괄호 안: AI 예측 이해도 · 실제 정답률")
        for s in out["strong_items"][:6]:
            sg = s["grade"]
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'{render_grade_badge(sg)} &nbsp;<b>{html.escape(s["concept_name"])}</b> '
                f'<span style="color:#5f6368;font-size:0.85rem;">'
                f'— 이해도 {sg["percent"]}% · {s["correct_text"]}</span>'
                f'</div>', unsafe_allow_html=True)

    # 더 다지면 좋은 개념 — 매핑+영상 있는 것만, 영상 카드 포함
    if out["refine_items"]:
        st.markdown("#### 🔧 그래도 더 다지면 좋은 개념")
        st.caption("이미 잘하고 있지만 한 번 더 짚어두면 좋은 강의를 추천드려요.")
        for item in out["refine_items"]:
            gg = item["grade"]
            st.markdown(f'<div class="concept-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div style="font-size:1.05rem;font-weight:700;">{html.escape(item["concept_name"])}</div>'
                f'{render_grade_badge(gg)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="color:#5f6368;font-size:0.85rem;margin-top:10px;">'
                f'이해도 <span title="AI 모델이 풀이 패턴을 분석해 예측한 종합 이해도예요." '
                f'style="cursor:help;">ⓘ</span></div>'
                f'{render_meter(gg["percent"], gg["color"])}'
                f'<div style="font-size:0.85rem;color:#5f6368;">'
                f'예측 이해도 <b>{gg["percent"]}%</b> &nbsp;·&nbsp; '
                f'실제 정답률 {item["correct_text"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="font-weight:700;margin:10px 0 6px 0;">📺 추천 영상</div>',
                        unsafe_allow_html=True)
            st.markdown(render_videos(item["videos"]), unsafe_allow_html=True)
            with st.expander("상세 정보 (선생님용)"):
                st.caption(
                    f"tag_id = {item['tag_id']}  ·  concept_id = {item['concept_id']}  ·  "
                    f"예측 이해도 = {item['mastery']:.4f}  ·  정답률 = {item['correct_rate']:.4f}  ·  "
                    f"풀이 = {item['solve_count']}회"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── 취약은 있으나 영상이 매핑된 게 없는 경우 ──
if n_show == 0:
    st.info(
        f"약한 개념이 {out['n_weak_all']}개 발견됐지만, 아직 추천 영상이 매핑되지 않은 개념들이에요. "
        "영상 매핑이 점차 확대되고 있으니 곧 추천이 가능해집니다."
    )
    st.markdown("#### 보충이 필요한 개념")
    weak_all = out["result_df"][
        out["result_df"]["avg_mastery"] <= out["weak_threshold"]
    ].sort_values("avg_mastery")
    for r in weak_all.itertuples(index=False):
        gg = engine.to_grade(r.avg_mastery)
        st.markdown(
            f'- {render_grade_badge(gg)} &nbsp;**{html.escape(r.concept_name)}** '
            f'<span style="color:#5f6368;">{gg["percent"]}%</span>',
            unsafe_allow_html=True)
    st.stop()

# ── 일반: 취약 개념 + 영상 추천 ──
st.markdown(
    f"#### 📚 보충이 필요한 개념 {n_show}개"
)
st.caption("이해도가 낮은 순서로 보여드려요. 약한 개념의 '진짜 원인'이 되는 선행 개념이 "
           "있으면 그 강의를 먼저 추천합니다.")

for i, item in enumerate(out["weak_items"], start=1):
    render_concept_card(item, i)

# ── 강점도 함께(동기부여) ──
if out["strong_items"]:
    with st.expander(f"💪 잘하고 있는 개념도 보기 ({len(out['strong_items'])}개)"):
        st.caption("이해도는 AI 예측, 정답률은 실제 풀이 결과예요.")
        for s in out["strong_items"][:8]:
            sg = s["grade"]
            st.markdown(
                f'{render_grade_badge(sg)} &nbsp;<b>{html.escape(s["concept_name"])}</b> '
                f'<span style="color:#5f6368;font-size:0.85rem;">'
                f'— 이해도 {sg["percent"]}% · {s["correct_text"]}</span>',
                unsafe_allow_html=True)

st.divider()
st.caption(
    f"추천된 개념 {n_show}개 · 선행 개념(Root Cause) 기반 추천 {out['n_root_cause']}개 · "
    f"직접 추천 {out['n_direct']}개"
)
