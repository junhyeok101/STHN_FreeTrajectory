# STHN_FreeTrajectory

### 요약
GT에 의존한 고정 경로 매칭의 한계를 넘어, **완전한 자유 경로에서도 동작하는 STHN 기반 매칭 모델을 구현**한 프로젝트. A project that overcomes the GT-dependent, fixed-path limitations of the original **STHN by enabling fully free-trajectory matching**.

---
### 문제 상황
기존 STHN 방식은 query의 GT 좌표를 기반으로 database를 미리 크롭해두고, 각 query와 매칭할 후보를 고정된 방식으로 설정한 뒤 출발한다.
이 때문에 query가 반드시 database 내부에 존재하는 환경에서만 매칭을 수행하게 되며, 사실상 정답 위치를 부분적으로 알고 있는 상태에서의 로컬 정밀 매칭에 가까웠다. (자세한 설명은 STHN_TraExtension 참고)

결과적으로 기존 접근은 고정된 경로에서의 좌표 세밀화 수준에 머물렀고, evaluation과 시각화 역시 GT 정보를 전제한 제한된 상황에서만 가능했다.
즉, 이러한 구조에서는 진정한 의미의 자유 경로 탐색이나 일반화된 매칭 능력을 확인하기 어렵다는 한계가 있었다.

이에 이 프로젝트는 GT에 의존한 고정 경로 매칭의 한계를 넘어, 완전한 자유 경로에서도 동작하는 STHN 기반 매칭 모델을 구현한 프로젝트로 구현 방식은 아래와 같다. 

---
### 데이터 로딩 방식 수정
기존의 
