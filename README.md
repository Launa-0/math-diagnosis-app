# 🎓 수학 학습 진단 & YouTube 영상 추천 (Streamlit 앱)

학생의 풀이 기록을 DKVMN 모델로 분석해 **약한 개념**을 찾고,
지식 위계를 **역추적**해 그 약점의 *진짜 원인(Root Cause)* 이 되는 선행 개념까지 짚어
딱 맞는 **YouTube 강의**를 추천해 주는 앱입니다.

원본 주피터 노트북을 일반 사용자(학생·학부모)가 쓰기 쉽게 다듬은 버전이에요.

---

## 📁 폴더에 같이 둬야 하는 파일 (7개)

| 파일 이름 | 설명 |
|---|---|
| `X_train.npy` | 학생 풀이 기록 (문항) |
| `y_train.npy` | 학생 풀이 기록 (정오답) |
| `best_dkvmn_model.pth` | 학습된 DKVMN 모델 |
| `math_concepts.csv` | 개념 이름 목록 (`id`, `name` 컬럼) |
| `Mathematics Knowledge System Dataset.json` | 지식 위계 그래프 |
| `tag_to_concept_id.json` | 태그 → 개념 매핑 |
| `youtube_recommendations_final_844concepts.json` | YouTube 영상 매핑 (844개 개념) |

> 파일이 `app.py` 와 같은 폴더에 있으면 **자동 인식**됩니다.
> 일부가 없으면 앱 화면 **왼쪽 사이드바에서 직접 업로드**할 수도 있어요.

폴더 구성 예시:
```
my_project/
├── app.py
├── engine.py
├── requirements.txt
├── X_train.npy
├── y_train.npy
├── best_dkvmn_model.pth
├── math_concepts.csv
├── Mathematics Knowledge System Dataset.json
├── tag_to_concept_id.json
└── youtube_recommendations_final_844concepts.json
```

---

## 🚀 실행 방법

### 1. 파이썬 확인
터미널에서:
```bash
python --version      # 또는  python3 --version
```
Python 3.8 ~ 3.11 권장.

### 2. (선택) 가상환경
```bash
python -m venv streamlit_env
# Windows
streamlit_env\Scripts\activate
# macOS / Linux
source streamlit_env/bin/activate
```

### 3. 라이브러리 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> `torch` 설치가 느리거나 막히면 https://pytorch.org 에서 본인 OS에 맞는 설치 명령을 복사해 쓰세요.
> 예) CPU 전용: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 4. 실행
```bash
python -m streamlit run app.py
```
브라우저가 자동으로 열리고 `http://localhost:8501` 에서 앱이 뜹니다.
(안 열리면 주소창에 직접 입력)

종료는 터미널에서 `Ctrl + C`.

---

## 🖥️ 화면에서 하는 일

1. 왼쪽에서 데이터 파일이 모두 ✅ 인지 확인 (없으면 업로드)
2. **학생 번호**를 입력하고 **[진단 시작]** 클릭
3. 결과 확인:
   - 전반 이해도 / 푼 문제 수 / 학습 개념 수
   - **보충이 필요한 개념** — 이해도 막대그래프 + 등급(최하·하·중·상·최상) + 추천 영상
   - 약점의 **진짜 원인(Root Cause)** 선행 개념 안내
   - **잘하고 있는 개념**(접기)
   - 내부 ID(tag_id 등)는 *"상세 정보 (선생님용)"* 접기 안에만 표시

성취도가 좋아 약한 개념이 하나도 없으면 **축하 화면**과 함께
강점 개념·조금 더 다지면 좋은 개념을 보여줍니다.

---

## ❓ 자주 막히는 부분

- **`streamlit: command not found`** → `pip install streamlit` 다시 실행, 또는 `python -m streamlit run app.py`
- **`PyTorch 가 설치되어 있지 않습니다`** → `pip install torch`
- **파일을 못 찾음** → 파일 이름이 위 표와 정확히 같은지 확인 (띄어쓰기·대소문자 포함)
- **Jupyter 안에서 설치** → `%pip install streamlit`
