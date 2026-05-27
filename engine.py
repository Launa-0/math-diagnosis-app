# -*- coding: utf-8 -*-
"""
지식 위계 기반 역추적 진단 엔진 (Streamlit 백엔드)
- 원본 노트북(DKVMN + 지식위계 Root Cause + YouTube 추천)의 로직을 그대로 옮기되,
  Streamlit에서 재사용하기 좋게 함수 단위로 정리했습니다.
- UI(streamlit) 코드는 app.py 에 있습니다. 이 파일은 순수 계산/데이터 로직만 담습니다.
"""

import os
import json
import random

import numpy as np
import pandas as pd

# torch 는 모델 추론에만 필요. 등급화/필터링 등 순수 로직은 torch 없이도 동작하도록 분리.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    Dataset = object  # 더미


# =========================================================
# 0. 기본 설정
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_AVAILABLE:
        return torch.device("cpu")
    return "cpu"


# 기본 파일명 (원본 노트북과 동일). app.py 에서 경로를 덮어쓸 수 있음.
DEFAULT_FILES = {
    "X": "X_train.npy",
    "Y": "y_train.npy",
    "MODEL": "best_dkvmn_model.pth",
    "CONCEPTS": "math_concepts.csv",
    "KNOWLEDGE": "Mathematics Knowledge System Dataset.json",
    "TAG_TO_CONCEPT": "tag_to_concept_id.json",
    "YOUTUBE": "youtube_recommendations_final_844concepts.json",
}


# =========================================================
# 1. 데이터셋 클래스
# =========================================================
class KTDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if TORCH_AVAILABLE:
            return {
                "q": torch.LongTensor(self.x[idx]),
                "r": torch.LongTensor(self.y[idx]),
            }
        return {"q": self.x[idx], "r": self.y[idx]}


# =========================================================
# 2. DKVMN 모델 (원본과 동일)
# =========================================================
if TORCH_AVAILABLE:

    class DKVMN(nn.Module):
        def __init__(self, n_questions, embed_dim=64, memory_size=50):
            super(DKVMN, self).__init__()
            self.n_questions = n_questions
            self.embed_dim = embed_dim
            self.memory_size = memory_size

            self.key_memory = nn.Parameter(torch.randn(memory_size, embed_dim))
            self.init_value_memory = nn.Parameter(torch.randn(memory_size, embed_dim))

            self.q_embed = nn.Embedding(n_questions + 1, embed_dim, padding_idx=0)
            self.qa_embed = nn.Embedding(2 * n_questions + 1, embed_dim, padding_idx=0)

            self.f_linear = nn.Linear(embed_dim * 2, embed_dim)
            self.p_linear = nn.Linear(embed_dim, 1)

            self.erase_linear = nn.Linear(embed_dim, embed_dim)
            self.add_linear = nn.Linear(embed_dim, embed_dim)

        def forward(self, q_seq, r_seq):
            batch_size, seq_len = q_seq.size()
            value_memory = self.init_value_memory.unsqueeze(0).repeat(batch_size, 1, 1)

            preds = []
            attentions = []

            for t in range(seq_len):
                q_t = q_seq[:, t]
                r_t = r_seq[:, t]

                q_emb = self.q_embed(q_t)
                correlation = F.softmax(
                    torch.matmul(q_emb, self.key_memory.t()), dim=-1
                )
                read_content = torch.bmm(
                    correlation.unsqueeze(1), value_memory
                ).squeeze(1)
                attentions.append(correlation)

                combined = torch.cat([read_content, q_emb], dim=-1)
                f_t = torch.tanh(self.f_linear(combined))
                p_t = torch.sigmoid(self.p_linear(f_t)).squeeze(-1)
                preds.append(p_t)

                qa_t = q_t + r_t * self.n_questions
                qa_t = qa_t.clone()
                qa_t[q_t == 0] = 0
                qa_emb = self.qa_embed(qa_t)

                erase = torch.sigmoid(self.erase_linear(qa_emb)).unsqueeze(1)
                add = torch.tanh(self.add_linear(qa_emb)).unsqueeze(1)
                cw = correlation.unsqueeze(-1)
                value_memory = value_memory * (1 - cw * erase) + (cw * add)

            preds = torch.stack(preds, dim=1)
            attentions = torch.stack(attentions, dim=1)
            return preds, attentions
else:  # pragma: no cover
    class DKVMN:  # 더미 (torch 없는 환경 검증용)
        def __init__(self, *a, **k):
            pass


# =========================================================
# 3. 지식체계 / 개념명 / 매핑 로드
# =========================================================
def load_math_concepts(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "id" not in df.columns or "name" not in df.columns:
        raise ValueError("math_concepts.csv에는 id, name 컬럼이 있어야 합니다.")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    return dict(zip(df["id"], df["name"]))


def load_knowledge_graph(path):
    """fromConcept -> toConcept : toConcept를 fromConcept의 선행 개념으로 해석"""
    if not os.path.exists(path):
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prereq_graph = {}
    concept_info = {}
    for _, item in data.items():
        from_c = item.get("fromConcept", {})
        to_c = item.get("toConcept", {})
        if "id" not in from_c or "id" not in to_c:
            continue
        from_id = int(from_c["id"])
        to_id = int(to_c["id"])
        concept_info[from_id] = from_c.get("name", f"개념 {from_id}")
        concept_info[to_id] = to_c.get("name", f"개념 {to_id}")
        prereq_graph.setdefault(from_id, []).append(to_id)
    return prereq_graph, concept_info


def load_tag_to_concept(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def load_youtube_recommendations(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        youtube_dict = json.load(f)
    return {int(k): v for k, v in youtube_dict.items()}


def get_concept_name(tag_id, concept_name_map, concept_info, tag_to_concept):
    """tag_id -> concept_id 변환 후 이름 조회. Returns (name, is_matched, concept_id)"""
    tag_id = int(tag_id)
    concept_id = tag_to_concept.get(tag_id)
    if concept_id is None:
        return f"지식태그 {tag_id}", False, None
    if concept_id in concept_name_map:
        return concept_name_map[concept_id], True, concept_id
    if concept_id in concept_info:
        return concept_info[concept_id], True, concept_id
    return f"지식태그 {tag_id}", False, concept_id


# =========================================================
# 4. 선행 개념 탐색 / Root Cause
# =========================================================
def trace_prerequisites(concept_id, prereq_graph, concept_name_map,
                        concept_info, tag_to_concept, max_depth=2):
    concept_id = int(concept_id)
    results = []
    visited = set()

    def dfs(current_id, depth):
        if depth > max_depth or current_id not in prereq_graph:
            return
        for pre_id in prereq_graph[current_id]:
            if pre_id in visited:
                continue
            visited.add(pre_id)
            pre_name = concept_name_map.get(
                pre_id, concept_info.get(pre_id, f"개념 {pre_id}")
            )
            is_matched = (pre_id in concept_name_map) or (pre_id in concept_info)
            results.append({
                "concept_id": int(pre_id),
                "concept_name": pre_name,
                "depth": depth,
                "is_matched": is_matched,
            })
            dfs(pre_id, depth + 1)

    dfs(concept_id, 1)
    return results


def build_mastery_map(q_valid, p_valid, tag_to_concept):
    mastery_map = {}
    for tag_id in sorted(set(q_valid)):
        tag_id_int = int(tag_id)
        concept_id = tag_to_concept.get(tag_id_int)
        if concept_id is None:
            continue
        idx = q_valid == tag_id
        mastery_map[concept_id] = float(np.mean(p_valid[idx]))
    return mastery_map


def find_root_cause(target_concept_id, mastery_map, prereq_graph,
                    concept_name_map, concept_info, tag_to_concept):
    prereqs = trace_prerequisites(
        target_concept_id, prereq_graph, concept_name_map,
        concept_info, tag_to_concept, max_depth=2,
    )
    candidates = []
    for p in prereqs:
        pre_id = int(p["concept_id"])
        candidates.append({
            "concept_id": pre_id,
            "concept_name": p["concept_name"],
            "depth": p["depth"],
            "mastery": mastery_map.get(pre_id, None),
        })
    valid = [c for c in candidates if c["mastery"] is not None]
    if not valid:
        return None, candidates
    return sorted(valid, key=lambda x: x["mastery"])[0], candidates


# =========================================================
# 5. 등급화 (일반 사용자용)
# =========================================================
# 이해도(0~1)를 5단계 등급으로. 경계값은 취약 기준 0.3을 포함하도록 설계.
GRADE_BANDS = [
    (0.0, 0.20, "최하", "기초부터 다시", "#E74C3C"),
    (0.20, 0.30, "하", "보충 필요", "#E67E22"),
    (0.30, 0.50, "중", "조금 더", "#F1C40F"),
    (0.50, 0.75, "상", "잘하고 있어요", "#2ECC71"),
    (0.75, 1.01, "최상", "완벽해요", "#27AE60"),
]


def to_grade(mastery):
    """이해도(0~1) -> dict(grade, label, color, percent)"""
    m = float(mastery)
    for lo, hi, grade, label, color in GRADE_BANDS:
        if lo <= m < hi:
            return {"grade": grade, "label": label, "color": color,
                    "percent": round(m * 100)}
    # 안전망
    return {"grade": "중", "label": "조금 더", "color": "#F1C40F",
            "percent": round(m * 100)}


def correct_rate_text(rate, count):
    """정답률+풀이수를 자연어로. 예: '6문제 중 2문제 정답 (33%)'"""
    n_correct = round(rate * count)
    return f"{count}문제 중 {n_correct}문제 정답 ({round(rate*100)}%)"


# =========================================================
# 6. 진단 (DataFrame 반환) — UI 비의존
# =========================================================
def run_inference(model, dataset, student_idx, device):
    """모델 추론. (q_valid, r_valid, p_valid) 반환."""
    model.eval()
    sample = dataset[student_idx]
    q = sample["q"].unsqueeze(0).to(device)
    r = sample["r"].unsqueeze(0).to(device)
    with torch.no_grad():
        preds, _ = model(q, r)
    q_np = q.squeeze(0).cpu().numpy()
    r_np = r.squeeze(0).cpu().numpy()
    p_np = preds.squeeze(0).cpu().numpy()
    valid = q_np != 0
    return q_np[valid], r_np[valid], p_np[valid]


def build_result_df(q_valid, r_valid, p_valid,
                    concept_name_map, concept_info, tag_to_concept):
    rows = []
    for tag_id in sorted(set(q_valid)):
        idx = q_valid == tag_id
        name, is_matched, concept_id = get_concept_name(
            tag_id, concept_name_map, concept_info, tag_to_concept
        )
        rows.append({
            "tag_id": int(tag_id),
            "concept_id": concept_id,
            "concept_name": name,
            "is_matched": is_matched,
            "avg_mastery": float(np.mean(p_valid[idx])),
            "actual_correct_rate": float(np.mean(r_valid[idx])),
            "solve_count": int(np.sum(idx)),
        })
    return pd.DataFrame(rows)


def diagnose(q_valid, r_valid, p_valid, resources, weak_threshold=0.3,
             top_k_videos=2):
    """
    전체 진단 결과를 dict 로 반환 (Streamlit 이 그대로 렌더링).
    resources: dict with concept_name_map, concept_info, tag_to_concept,
               prereq_graph, youtube_map
    """
    cnm = resources["concept_name_map"]
    cinfo = resources["concept_info"]
    t2c = resources["tag_to_concept"]
    prereq = resources["prereq_graph"]
    ymap = resources["youtube_map"]

    result_df = build_result_df(q_valid, r_valid, p_valid, cnm, cinfo, t2c)
    mastery_map = build_mastery_map(q_valid, p_valid, t2c)

    def get_videos(concept_id):
        if concept_id is None or concept_id not in ymap:
            return []
        return [v for v in ymap.get(concept_id, [])
                if v.get("rank", 999) <= 2][:top_k_videos]

    # ── 취약 개념 (영상이 매핑된 것만 노출) ──
    weak_df = result_df[result_df["avg_mastery"] <= weak_threshold] \
        .sort_values("avg_mastery").reset_index(drop=True)

    weak_items = []
    for row in weak_df.itertuples(index=False):
        root_cause = None
        if row.is_matched and row.concept_id is not None:
            root_cause, _ = find_root_cause(
                int(row.concept_id), mastery_map, prereq, cnm, cinfo, t2c
            )

        videos, source, source_type = [], None, None
        if root_cause is not None:
            videos = get_videos(root_cause["concept_id"])
            if videos:
                source = root_cause["concept_name"]
                source_type = "root_cause"
        if not videos and row.concept_id is not None:
            videos = get_videos(int(row.concept_id))
            if videos:
                source = row.concept_name
                source_type = "direct"

        if not videos:  # 매핑된(영상있는) 태그만 출력 — 요구사항
            continue

        weak_items.append({
            "concept_name": row.concept_name,
            "tag_id": row.tag_id,
            "concept_id": row.concept_id,
            "mastery": row.avg_mastery,
            "grade": to_grade(row.avg_mastery),
            "correct_rate": row.actual_correct_rate,
            "solve_count": row.solve_count,
            "correct_text": correct_rate_text(row.actual_correct_rate, row.solve_count),
            "root_cause": root_cause,
            "root_cause_grade": to_grade(root_cause["mastery"]) if root_cause else None,
            "videos": videos,
            "source": source,
            "source_type": source_type,
        })

    # 매핑된(이름과 concept_id가 확인된) 개념만 화면에 노출
    matched_df = result_df[result_df["is_matched"] & result_df["concept_id"].notna()]

    def _item_with_videos(r):
        cid = int(r.concept_id) if r.concept_id is not None else None
        videos = get_videos(cid) if cid is not None else []
        return {
            "concept_name": r.concept_name,
            "tag_id": int(r.tag_id),
            "concept_id": cid,
            "mastery": float(r.avg_mastery),
            "grade": to_grade(r.avg_mastery),
            "correct_rate": float(r.actual_correct_rate),
            "solve_count": int(r.solve_count),
            "correct_text": correct_rate_text(r.actual_correct_rate, r.solve_count),
            "videos": videos,
        }

    # ── 강점 개념 (잘하는 것, 매핑된 것만) ──
    strong_df = matched_df[matched_df["avg_mastery"] >= 0.5] \
        .sort_values("avg_mastery", ascending=False)
    strong_items = [_item_with_videos(r) for r in strong_df.itertuples(index=False)]

    # ── '더 다지면 좋은 개념' (우수 학습자용, 매핑+영상 있는 것 우선) ──
    # 취약 임계(0.3)는 넘지만 상대적으로 약한 개념 중에서 영상이 있는 것만 우선 노출.
    refine_df = matched_df[matched_df["avg_mastery"] > weak_threshold] \
        .sort_values("avg_mastery")
    refine_items_all = [_item_with_videos(r) for r in refine_df.itertuples(index=False)]
    refine_items = [it for it in refine_items_all if it["videos"]][:5]

    # ── 전반 요약 ──
    overall = float(np.mean(p_valid)) if len(p_valid) else 0.0
    n_weak_all = int((result_df["avg_mastery"] <= weak_threshold).sum())

    return {
        "result_df": result_df,
        "total_records": int(len(q_valid)),
        "unique_tags": int(len(result_df)),             # 풀이한 전체 고유 태그
        "unique_matched_tags": int(len(matched_df)),    # 매핑된 것만
        "weak_threshold": weak_threshold,
        "n_weak_all": n_weak_all,                       # 취약 전체(영상 유무 무관)
        "weak_items": weak_items,                       # 화면에 노출할 취약 개념(영상 있음)
        "strong_items": strong_items,                   # 강점(매핑된 것만, 영상정보 포함)
        "refine_items": refine_items,                   # 더 다지면 좋은 개념(영상 있음)
        "overall_mastery": overall,
        "overall_grade": to_grade(overall),
        "n_mapped_concepts": len(ymap),
        "n_root_cause": sum(1 for w in weak_items if w["source_type"] == "root_cause"),
        "n_direct": sum(1 for w in weak_items if w["source_type"] == "direct"),
    }
