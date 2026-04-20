# Yongjun's Blog

[Hugo](https://github.com/gohugoio/hugo)로 빌드하고, [nostyleplease](https://github.com/hanwenguo/hugo-theme-nostyleplease) 테마를 커스터마이징하여 사용하는 개인 기술 블로그.

배포 URL: https://elwhyjay.github.io

---

## 디렉토리 구조

```
myblog/
├── archetypes/          # 새 포스트 생성 시 사용되는 템플릿
│   └── default.md
├── config/
│   └── _default/
│       ├── config.toml  # Hugo 메인 설정 (baseURL, 테마, GA 등)
│       ├── menu.toml    # 사이드바 메뉴 구성
│       └── params.toml  # 추가 파라미터 (검색, 코드 라인 수 등)
├── content/
│   └── post/            # 블로그 포스트 (.md 파일)
├── static/
│   └── img/             # 이미지 파일 (favicon, 포스트 이미지 등)
├── layouts/             # 커스텀 레이아웃 오버라이드 (현재 비어있음)
├── themes/
│   └── nostyleplease/   # 테마 (Git submodule)
├── data/                # 데이터 파일 (현재 비어있음)
├── i18n/                # 다국어 지원 (현재 비어있음)
├── public/              # 빌드 결과물 (.gitignore 대상)
└── resources/           # Hugo 캐시 (.gitignore 대상)
```

---

## 빠른 시작

### 사전 요구사항

- [Hugo Extended](https://gohugo.io/installation/) (v0.41.0 이상, SCSS 컴파일을 위해 Extended 버전 필요)

### 다른 기기에서 세팅하기

테마가 Git submodule이므로 클론 시 함께 가져와야 함.

```bash
# 최초 클론 (submodule 포함)
git clone --recurse-submodules https://github.com/elwhyjay/myblog.git
cd myblog

# 이미 클론했는데 themes/nostyleplease가 비어있다면
git submodule update --init --recursive
```

작업 전환 흐름:

```bash
# 기존 기기: 작성 중인 내용 모두 푸쉬
git add -A && git commit -m "sync" && git push

# 새 기기: 최신 pull + submodule 업데이트
git pull
git submodule update --init --recursive
```

> **테마 수정 후 push 하려면** (새 기기 최초 1회만):
> `.gitmodules`의 테마 URL은 HTTPS지만 GitHub가 HTTPS push에 토큰을 요구함. 매번 remote를 SSH로 고치는 대신 글로벌 설정 한 줄로 해결:
> ```bash
> git config --global url."git@github.com:".insteadOf "https://github.com/"
> ```
> 이후 SSH 키가 등록된 상태라면 HTTPS URL로 clone/pull 하면서도 push는 자동으로 SSH로 나감.

### 로컬 서버 실행

```bash
# 드래프트 포함하여 로컬에서 미리보기 (public/ 미변경)
hugo server -D --renderToMemory

# 드래프트 제외 (배포용 확인)
hugo server --renderToMemory
```

기본적으로 `http://localhost:1313` 에서 확인 가능. 파일 수정 시 자동 리로드됨.
`--renderToMemory`를 붙여야 `public/`에 쓰지 않음.

### 사이트 빌드

```bash
hugo       # draft 제외
hugo -D    # draft 포함
```

`public/` 디렉토리에 정적 사이트가 생성됨.

---

## 포스트 작성

### 새 포스트 생성

```bash
hugo new post/포스트-제목.md
```

`archetypes/default.md` 템플릿을 기반으로 `content/post/포스트-제목.md` 파일이 생성됨. 파일명의 `-`는 제목에서 공백으로 변환됨.

### Front Matter

포스트 상단에 TOML 형식의 메타데이터를 작성:

```toml
+++
title = '포스트 제목'
date = 2025-03-05T15:22:43+09:00
draft = true          # true면 로컬(-D 플래그)에서만 보임, false면 배포 시에도 보임
tags = ['태그1', '태그2']
mathjax = true        # 수식 렌더링이 필요한 경우
toc = true            # 목차(Table of Contents) 표시
tocBorder = true      # 목차에 테두리 추가
+++
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `title` | O | 포스트 제목 |
| `date` | O | 작성일시 (YYYY-MM-DDTHH:MM:SS+09:00) |
| `draft` | O | `true`: 로컬 미리보기 전용 / `false`: 배포 시 공개 |
| `tags` | - | 태그 배열. 카테고리 메뉴에서 필터링에 사용됨 |
| `mathjax` | - | `true`로 설정하면 MathJax 수식 렌더링 활성화 |
| `toc` | - | `true`로 설정하면 포스트 상단에 목차 표시 |
| `tocBorder` | - | `true`로 설정하면 목차에 테두리 스타일 적용 |
| `custom_js` | - | 포스트에서 사용할 커스텀 JS 파일 목록 |

### 포스트 공개하기

`draft = true` → `draft = false`로 변경하면 빌드 시 사이트에 포함됨.

---

## Shortcodes

테마에서 제공하는 커스텀 shortcode 3가지:

### 수식 (MathJax)

Front matter에 `mathjax = true` 설정 필요.

```markdown
인라인 수식: {{</* texi "E = mc^2" */>}}

디스플레이 수식:
{{</* texd "\\sum_{i=1}^{n} x_i = x_1 + x_2 + \\cdots + x_n" */>}}
```

### 이미지 나란히 배치

이미지와 캡션을 짝으로 전달. 이미지 수에 따라 자동으로 너비 분배됨.

```markdown
{{</* image-row "/img/image1.png" "캡션1" "/img/image2.png" "캡션2" */>}}
```

---

## 설정 파일 가이드

### config.toml (메인 설정)

`config/_default/config.toml`

```toml
baseURL = 'https://elwhyjay.github.io'   # 배포 URL
languageCode = 'ko-kr'                    # 언어 설정
title = "Yongjun's blog"                  # 사이트 제목
theme = 'nostyleplease'                   # 사용 테마
googleAnalytics = "G-YXXYQPDYLX"         # Google Analytics ID

[taxonomies]
    categories = 'categories'
    tag = 'tags'
    remind = 'reminds'                    # 커스텀 택소노미

[Params]
    favicon = "img/favicon.ico"
    disable_comments = false              # Utterances 댓글 활성화
```

### menu.toml (메뉴 구성)

`config/_default/menu.toml` — 홈페이지 사이드바 메뉴를 정의.

- `weight` 값이 작을수록 위에 표시됨
- `parent`를 지정하면 하위 메뉴로 들어감

```toml
# 최상위 메뉴
[[main]]
  name = "Home"
  url = ""
  weight = -110

# 하위 메뉴가 있는 항목
[[main]]
  name = "Links"
  identifier = "Links"
  weight = -108

[[main]]
  parent = "Links"
  name = "Github"
  url = "https://github.com/elwhyjay"
```

### params.toml (추가 파라미터)

`config/_default/params.toml`

```toml
enableSearch = true      # 검색 기능 활성화
codeLineNumbers = true   # 코드 블록 줄 번호
codeMaxLines = 15        # 코드 블록 최대 표시 줄 수
```

---

## 테마

### 개요

**nostyleplease** — CSS 최소화를 지향하는 미니멀 Hugo 테마.

- Git submodule로 관리: `themes/nostyleplease/`
- 원본: https://github.com/hanwenguo/hugo-theme-nostyleplease
- 다크/라이트 모드 자동 전환 + 수동 토글 버튼 (우측 상단)
- MathJax 수식 지원
- Utterances 댓글 시스템 (repo: `elwhyjay/blog-comment`)

### 테마 설정 (테마 내 config.toml)

`themes/nostyleplease/config.toml`에서 변경 가능한 주요 설정:

| 설정 | 현재 값 | 설명 |
|------|---------|------|
| `appearance` | `"auto"` | 테마 모드: `auto`, `light`, `dark` |
| `back_home_text` | `".."` | 포스트에서 홈으로 돌아가는 링크 텍스트 |
| `date_format` | `"2006-01-02"` | 날짜 표시 형식 (Go 날짜 포맷) |
| `isListGroupByDate` | `false` | 포스트 목록을 연도별로 그룹화 |
| highlight style | `"rrt"` | 코드 하이라이팅 색상 테마 |
| ToC levels | `2 ~ 3` | 목차에 포함할 헤딩 레벨 (h2 ~ h3) |

### 커스터마이징

테마를 직접 수정하지 않고 오버라이드하려면, 테마의 `layouts/` 내 파일과 동일한 경로로 프로젝트 루트의 `layouts/`에 파일을 생성하면 됨. Hugo는 프로젝트 루트의 레이아웃을 우선 적용.

```
# 예: single 포스트 레이아웃을 오버라이드
myblog/layouts/_default/single.html  ← 이 파일이 우선 적용됨
themes/nostyleplease/layouts/_default/single.html
```

---

## 이미지/정적 파일

`static/` 디렉토리의 파일은 빌드 시 사이트 루트에 복사됨.

```
static/img/favicon.ico    → https://elwhyjay.github.io/img/favicon.ico
static/img/example.png    → https://elwhyjay.github.io/img/example.png
```

포스트에서 이미지 사용:

```markdown
![설명](/img/example.png)
```

---

## 댓글 시스템

[Utterances](https://utteranc.es/)를 사용하여 GitHub Issues 기반 댓글 제공.

- 댓글 저장소: `elwhyjay/blog-comment`
- 이슈 매핑: 포스트 제목 기준
- 테마: `github-dark`
- 비활성화: `config.toml`에서 `disable_comments = true`

---

## 배포 (GitHub Pages)

`public/`은 별도 git repo (`elwhyjay.github.io`)로 관리됨.

```bash
# 1. 로컬 미리보기 (public/ 건드리지 않음)
hugo server -D --renderToMemory

# 2. 배포용 빌드 (draft 제외하려면 -D 빼기)
hugo -D

# 3. public/ 배포
cd public && git add -A && git commit -m "deploy" && git push && cd ..

# 4. 소스 커밋
git add -A && git commit -m "update" && git push
```

> `--renderToMemory` 없이 `hugo server -D`를 쓰면 public/에 draft 포함 빌드가 써지므로 주의.

---

## 자주 쓰는 명령어

| 명령어 | 설명 |
|--------|------|
| `hugo server -D --renderToMemory` | 드래프트 포함 로컬 미리보기 (public/ 미변경) |
| `hugo server --renderToMemory` | 배포용 로컬 미리보기 (public/ 미변경) |
| `hugo` | 정적 사이트 빌드 - draft 제외 (`public/`에 출력) |
| `hugo -D` | 정적 사이트 빌드 - draft 포함 (`public/`에 출력) |
| `hugo new post/제목.md` | 새 포스트 생성 |
| `git submodule update --init` | 테마 submodule 초기화 (클론 후 최초 1회) |
| `git submodule update --remote` | 테마를 최신 버전으로 업데이트 |

---

## 택소노미 (분류 체계)

포스트를 분류하는 3가지 택소노미가 설정되어 있음:

- **tags**: 포스트 태그 (주로 사용)
- **categories**: 카테고리
- **reminds**: 커스텀 택소노미

포스트 front matter에서 사용:

```toml
tags = ['CUDA', 'GPU']
categories = ['Development']
```

태그별 포스트 목록: `https://elwhyjay.github.io/tags/태그명/`
