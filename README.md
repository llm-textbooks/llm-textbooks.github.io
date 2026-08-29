# LLM Textbooks

LLM의 서빙·훈련·에이전트 메커니즘을 코드, 수학, 상태, 하드웨어의 연결로 읽는 공개 교과서 사이트입니다.

- 1권: **CUDA로 내려가며 읽는 LLM 추론 서버** — 78장
- 2권: **파인튜닝 메커니즘** — 본문·실습·플레이북 53문서
- 3권: **멀티 에이전트 메커니즘** — 준비 중

사이트: <https://llm-textbooks.github.io>

## 로컬 빌드

Node.js 24 이상을 사용합니다.

```bash
npm ci
npm test
npx serve _site
```

`npm test`는 다음 작업을 한 번에 수행합니다.

1. `books.yaml`과 각 권의 `book.yaml`에서 권·부·장 구조를 읽습니다.
2. Markdown을 장별 HTML로 변환합니다.
3. 표, 코드, KaTeX 수식, Mermaid 도식, 내부·외부 링크를 독서 UI에 맞게 구성합니다.
4. Pagefind 한국어 검색 색인을 생성합니다.
5. 모든 내부 링크, heading ID, 표 wrapper, 필수 자산과 권별 문서 수를 검사합니다.
6. 논문·코드·모델 카드·공식 문서 링크를 분류하고 mutable GitHub branch 링크를 차단합니다.
7. 2권의 모든 본문 장에 실행 지도, 상태표, Golden Run 인계와 반증 절차가 있는지 편집 품질 게이트로 검사합니다.

각 장 끝의 **이 장의 원전 바로가기** 패널은 본문에 인용된 링크를 논문, 코드, 모델·데이터, 공식 문서로 자동 분류합니다. 코드 근거는 가능하면 `repository + 40자리 commit + file + #Lx-Ly`, 논문은 arXiv/DOI/공식 출판 페이지, 모델은 revision이 고정된 모델 카드를 사용합니다.

생성 결과는 `_site/`에 만들어지며 Git에는 포함되지 않습니다. `main` 브랜치에 반영되면 GitHub Actions가 같은 검사를 통과한 정적 artifact만 GitHub Pages에 배포합니다.

## 콘텐츠 구조

```text
books.yaml                     # 시리즈 전체 등록부
content/
  volume-1/
    book.yaml                  # 1권의 부·장 spine
    chapters/
  volume-2/
    book.yaml                  # 2권의 부·장 spine
    chapters/
    labs/
    playbooks/
public/downloads/              # EPUB
site-src/                      # 독서 UI의 CSS와 JavaScript
scripts/build.mjs              # 정적 사이트 생성기
scripts/verify.mjs             # fail-closed 산출물 검사
scripts/audit-volume2-editorial.mjs # 2권 실행 중심 편집 감사
```

URL은 제목이 아니라 파일 slug를 기반으로 생성합니다. 제목을 고쳐도 장 주소가 바뀌지 않습니다. 각 절의 anchor에는 장 slug와 문서 안 순번을 함께 넣어 같은 제목이 반복되어도 충돌하지 않게 했습니다.

## 3권 추가 방법

3권 원고가 준비되면 다음 순서로 등록합니다.

1. `content/volume-3/book.yaml`에 권 정보와 `parts`, `chapters`를 선언합니다.
2. 원고를 `content/volume-3/chapters/` 등에 둡니다.
3. `books.yaml`의 `volume-3` 항목에 `manifest`, `content_root`, EPUB 경로를 추가하고 상태를 `published`로 바꿉니다.
4. `npm test`로 전체 세 권의 링크·검색·도식·표·코드 렌더링을 검증합니다.

별도 템플릿을 복제할 필요 없이 동일한 탐색, 검색, EPUB, 저자, SEO 구조가 자동으로 생성됩니다.

## 저자

[AUTHOR.md](AUTHOR.md)의 정보를 단일 원본으로 사용합니다.

- Jioh L. Jung
- <jung@jioh.net> / <ziozzang@skku.edu>
- <http://www.linkedin.com/in/ziozzang>

본문의 인용·소스·논문 링크는 각 장에 연결되어 있습니다. 저작권과 출처 정책은 각 권의 manifest 및 해당 장에 기록된 정보를 따릅니다.
