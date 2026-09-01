import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { Marked } from "marked";
import markedKatex from "marked-katex-extension";
import hljs from "highlight.js";
import YAML from "yaml";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "_site");
const REPO = "https://github.com/llm-textbooks/llm-textbooks.github.io";

const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
const yaml = (p) => YAML.parse(read(p));
const esc = (s = "") => String(s).replace(/[&<>\"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const stripTags = (s = "") => String(s).replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
const mkdir = (p) => fs.mkdirSync(p, { recursive: true });
const write = (rel, data) => { const p = path.join(OUT, rel); mkdir(path.dirname(p)); fs.writeFileSync(p, data); };
const copy = (src, dest) => { mkdir(path.dirname(path.join(OUT, dest))); fs.copyFileSync(path.join(ROOT, src), path.join(OUT, dest)); };
const copyTree = (src, dest) => {
  const from = path.join(ROOT, src);
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    // Python verifiers may leave bytecode caches beside the lab fixtures.
    // They are local execution artifacts, not deployable lab content.
    if (entry.name === "__pycache__") continue;
    const childSrc = path.posix.join(src, entry.name);
    const childDest = path.posix.join(dest, entry.name);
    if (entry.isDirectory()) copyTree(childSrc, childDest);
    else copy(childSrc, childDest);
  }
};

class StableSlugger {
  constructor(prefix) { this.prefix = prefix; this.seen = new Map(); this.order = 0; }
  slug(text) {
    this.order += 1;
    const base = stripTags(text).normalize("NFKC").toLowerCase()
      .replace(/[`'"“”‘’()[\]{}]/g, "")
      .replace(/[^\p{Letter}\p{Number}._-]+/gu, "-").replace(/^-+|-+$/g, "") || "section";
    const n = (this.seen.get(base) || 0) + 1;
    this.seen.set(base, n);
    return `${this.prefix}-s${String(this.order).padStart(3, "0")}-${base}${n > 1 ? `-${n}` : ""}`;
  }
}

const siteConfig = yaml("books.yaml");
const published = [];
const planned = [];
const routeBySource = new Map();
const fragmentBySource = new Map();

for (const entry of siteConfig.books) {
  if (!new Set(["published", "draft"]).has(entry.status)) { planned.push(entry); continue; }
  const manifest = yaml(entry.manifest);
  const docs = [];
  const seen = new Set();
  const parts = (manifest.parts || []).map((part, partIndex) => {
    const partDocs = [];
    for (const declared of part.chapters || []) {
      const normalized = declared.replace(/^next\//, "");
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      const source = `${entry.content_root}/${normalized}`;
      if (!fs.existsSync(path.join(ROOT, source))) throw new Error(`missing content: ${source}`);
      const kind = normalized.split("/")[0];
      const slug = path.basename(normalized, ".md");
      const route = `/books/${entry.id}/${kind}/${slug}/`;
      const doc = { source, logical: normalized, kind, slug, route, partIndex, order: docs.length + 1 };
      docs.push(doc); partDocs.push(doc); routeBySource.set(source, route);
    }
    return { ...part, index: partIndex + 1, docs: partDocs, route: `/books/${entry.id}/part-${partIndex + 1}/` };
  });
  published.push({ ...entry, manifest, docs, parts, route: `/books/${entry.id}/` });
}

function resolveMarkdownLink(book, doc, href) {
  if (!href || /^(https?:|mailto:|tel:|#)/.test(href)) return href;
  const [target, fragment = ""] = href.split("#", 2);
  if (!target.endsWith(".md")) return href;
  const logicalTarget = path.posix.normalize(path.posix.join(path.posix.dirname(doc.logical), target));
  const source = `${book.content_root}/${logicalTarget}`;
  const route = routeBySource.get(source);
  if (!route) return href;
  const requested = fragment ? decodeURIComponent(fragment) : "";
  const sourceMap = fragmentBySource.get(source);
  const tokenSubsequence = (needle, haystack) => {
    let cursor = 0;
    for (const token of needle.split("-").filter(Boolean)) {
      const found = haystack.indexOf(token, cursor);
      if (found < 0) return false;
      cursor = found + token.length;
    }
    return true;
  };
  const mapped = requested && sourceMap
    ? sourceMap.get(requested)
      || [...sourceMap.entries()].find(([key]) => key.startsWith(requested))?.[1]
      || [...sourceMap.entries()].find(([key]) => tokenSubsequence(requested, key))?.[1]
    : null;
  return route + (mapped ? `#${mapped}` : "");
}

function linkClass(href) {
  if (!href?.startsWith("http")) return "internal-link";
  if (/github\.com/.test(href)) return "external-link source-link";
  if (/arxiv\.org|doi\.org|aclanthology\.org|openreview\.net/.test(href)) return "external-link paper-link";
  if (/docs\.|developer\.|pytorch\.org|nvidia\.com/.test(href)) return "external-link docs-link";
  return "external-link";
}

function renderMarkdown(book, doc, markdown) {
  const headings = [];
  const references = [];
  const referenceUrls = new Set();
  const slugger = new StableSlugger(`${book.id}-${doc.slug}`);
  const renderer = {
    heading(token) {
      const text = this.parser.parseInline(token.tokens);
      const id = slugger.slug(stripTags(text));
      headings.push({ depth: token.depth, id, text: stripTags(text) });
      return `<h${token.depth} id="${esc(id)}">${text}<a class="heading-anchor" href="#${esc(id)}" aria-label="이 절의 주소 복사">#</a></h${token.depth}>\n`;
    },
    link(token) {
      const href = resolveMarkdownLink(book, doc, token.href);
      const body = this.parser.parseInline(token.tokens);
      const external = /^https?:/.test(href || "");
      if (external && !referenceUrls.has(href)) {
        referenceUrls.add(href);
        references.push({ href, label: stripTags(body) || href, kind: referenceKind(href) });
      }
      return `<a href="${esc(href)}" class="${linkClass(href)}"${token.title ? ` title="${esc(token.title)}"` : ""}${external ? ' rel="noopener noreferrer external"' : ""}>${body}${external ? '<span class="external-mark" aria-hidden="true">↗</span>' : ""}</a>`;
    },
    code(token) {
      const lang = (token.lang || "text").trim().split(/\s+/)[0].toLowerCase();
      if (lang === "mermaid") {
        return `<figure class="diagram" data-diagram><pre class="mermaid">${esc(token.text)}</pre><figcaption>구조와 상태 흐름을 나타낸 도식</figcaption></figure>`;
      }
      if (lang === "math") return `<div class="math-display">${esc(token.text)}</div>`;
      const alias = lang === "sh" ? "bash" : lang === "py" ? "python" : lang;
      let highlighted;
      try { highlighted = hljs.getLanguage(alias) ? hljs.highlight(token.text, { language: alias }).value : esc(token.text); }
      catch { highlighted = esc(token.text); }
      return `<div class="code-block"><div class="code-toolbar"><span>${esc(lang || "text")}</span><button type="button" class="copy-code">복사</button></div><pre><code class="hljs language-${esc(alias)}">${highlighted}</code></pre></div>`;
    }
  };
  const marked = new Marked({ gfm: true, breaks: false, renderer });
  marked.use(markedKatex({ throwOnError: false, strict: "ignore", nonStandard: true, output: "htmlAndMathml" }));
  const prepared = markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `\n$$\n${body.trim()}\n$$\n`)
    .replace(/\\\((.+?)\\\)/gs, (_, body) => `$${body}$`);
  let html = marked.parse(prepared);
  html = html.replace(/<table>/g, '<div class="table-scroll" tabindex="0" role="region" aria-label="가로로 스크롤할 수 있는 표"><table>')
    .replace(/<\/table>/g, "</table></div>");
  return { html, headings, references };
}

function referenceKind(href) {
  if (/arxiv\.org|doi\.org|aclanthology\.org|openreview\.net|proceedings\.mlr\.press/.test(href)) return "논문";
  if (/github\.com|gitlab\.com|codeberg\.org/.test(href)) return "코드";
  if (/huggingface\.co/.test(href)) return "모델·데이터";
  if (/docs\.|developer\.|pytorch\.org|nvidia\.com|kubernetes\.io/.test(href)) return "공식 문서";
  return "관련 원문";
}

function referencePanel(references) {
  if (!references?.length) return "";
  const order = ["논문", "코드", "모델·데이터", "공식 문서", "관련 원문"];
  const grouped = new Map(order.map(kind => [kind, []]));
  for (const ref of references) grouped.get(ref.kind).push(ref);
  const groups = order.filter(kind => grouped.get(kind).length).map(kind =>
    `<section><h3>${kind} <span>${grouped.get(kind).length}</span></h3><ul>${grouped.get(kind).map(ref => `<li><a href="${esc(ref.href)}" rel="noopener noreferrer external">${esc(ref.label)}<span class="external-mark" aria-hidden="true">↗</span></a><small>${esc(new URL(ref.href).hostname)}</small></li>`).join("")}</ul></section>`
  ).join("");
  return `<details class="origin-references" data-pagefind-ignore><summary>이 장의 원전 바로가기 <span>${references.length}</span></summary><div>${groups}</div></details>`;
}

function shell({ title, description, body, canonical = "/", book = null, doc = null, toc = [] }) {
  const fullTitle = title === siteConfig.site.title ? title : `${title} · ${siteConfig.site.title}`;
  const nav = book ? chapterNav(book, doc) : "";
  const tocHtml = toc.filter(h => h.depth >= 2 && h.depth <= 3).map(h => `<a class="toc-d${h.depth}" href="#${esc(h.id)}">${esc(h.text)}</a>`).join("");
  return `<!doctype html>
<html lang="ko-KR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(fullTitle)}</title><meta name="description" content="${esc(description || siteConfig.site.description)}">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="canonical" href="${siteConfig.site.url}${canonical}"><meta property="og:title" content="${esc(fullTitle)}"><meta property="og:description" content="${esc(description || siteConfig.site.description)}"><meta property="og:type" content="${doc ? "article" : "website"}"><meta property="og:url" content="${siteConfig.site.url}${canonical}">
<link rel="stylesheet" href="/assets/site.css"><link rel="stylesheet" href="/assets/highlight.css"><link rel="stylesheet" href="/assets/katex.min.css"><link rel="stylesheet" href="/pagefind/pagefind-ui.css">
<script defer src="/assets/mermaid.min.js"></script><script defer src="/assets/site.js"></script></head>
<body${book ? ` style="--book-accent:${esc(book.accent)}"` : ""}><a class="skip-link" href="#main">본문으로 건너뛰기</a>
<header class="site-header"><a class="brand" href="/">LLM <strong>Textbooks</strong></a><nav aria-label="주 메뉴"><a href="/books/">전권</a><a href="/search/">검색</a><a href="/author/">저자</a><a href="${REPO}">GitHub</a></nav>${book ? '<button class="nav-toggle" aria-label="장 목차 열기" aria-expanded="false">목차</button>' : ""}</header>
${book ? `<div class="reader-layout"><aside class="chapter-nav" aria-label="책 목차">${nav}</aside>` : ""}
<main id="main"${doc ? ' data-pagefind-body' : ""}>${body}</main>
${book && doc ? `<aside class="page-toc" aria-label="현재 장의 절"><strong>이 장에서</strong>${tocHtml}</aside>` : ""}${book ? "</div>" : ""}
<footer><p>© Jioh L. Jung · <a href="/author/">저자 소개</a> · <a href="${REPO}">원문과 개정 이력</a></p></footer></body></html>`;
}

function chapterNav(book, current) {
  return `<a class="book-nav-title" href="${book.route}"><span>${book.number}권</span>${esc(book.manifest.title)}</a>` + book.parts.map(part =>
    `<details${current?.partIndex === part.index - 1 ? " open" : ""}><summary>${esc(part.title)}</summary>${part.docs.map(d => `<a href="${d.route}"${current?.route === d.route ? ' aria-current="page"' : ""}><span>${String(d.order).padStart(2,"0")}</span>${esc(d.title || d.slug)}</a>`).join("")}</details>`
  ).join("");
}

function partCard(book, part) {
  return `<section class="part-card"><p class="eyebrow">${book.number}권 · ${part.index}부</p><h2><a href="${part.route}">${esc(part.title)}</a></h2><p>${esc(part.promise || part.question || "")}</p><ol>${part.docs.map(d => `<li><a href="${d.route}">${esc(d.title || d.slug)}</a></li>`).join("")}</ol></section>`;
}

// First pass renders documents and captures titles/headings.
for (const book of published) {
  for (const doc of book.docs) {
    const md = read(doc.source);
    const rendered = renderMarkdown(book, doc, md);
    doc.rendered = rendered;
    doc.title = rendered.headings.find(h => h.depth === 1)?.text || doc.slug;
    doc.description = stripTags(rendered.html).slice(0, 180);
    const sourceMap = new Map();
    for (const heading of rendered.headings) {
      const githubSlug = heading.text.normalize("NFKC").toLowerCase()
        .replace(/<[^>]*>/g, "").replace(/[\p{Punctuation}\p{Symbol}]/gu, "")
        .replace(/\s+/g, "-").replace(/^-+|-+$/g, "");
      if (githubSlug && !sourceMap.has(githubSlug)) sourceMap.set(githubSlug, heading.id);
    }
    fragmentBySource.set(doc.source, sourceMap);
  }
}

// Re-render now that every target document's source fragment map is known.
for (const book of published) {
  for (const doc of book.docs) doc.rendered = renderMarkdown(book, doc, read(doc.source));
}

for (const book of published) {
  for (let i = 0; i < book.docs.length; i++) {
    const doc = book.docs[i];
    const prev = book.docs[i - 1], next = book.docs[i + 1];
    const sourceUrl = `${REPO}/blob/main/${doc.source}`;
    const body = `<article class="chapter" data-pagefind-filter="volume:${book.number}권" data-pagefind-meta="title:${esc(doc.title)}"><div class="chapter-meta"><a href="${book.route}">${book.number}권</a><span>${esc(book.parts[doc.partIndex].title)}</span><span>${doc.kind === "chapters" ? `${doc.order}장` : doc.kind}</span></div>${doc.rendered.html}${referencePanel(doc.rendered.references)}<div class="chapter-source"><a href="${sourceUrl}">이 장의 Markdown 원문과 개정 이력 보기 ↗</a></div><nav class="pager" aria-label="이전과 다음 장">${prev ? `<a rel="prev" href="${prev.route}"><small>이전</small>${esc(prev.title)}</a>` : "<span></span>"}${next ? `<a rel="next" href="${next.route}"><small>다음</small>${esc(next.title)}</a>` : ""}</nav></article>`;
    write(path.posix.join(doc.route, "index.html").slice(1), shell({ title: doc.title, description: doc.description, body, canonical: doc.route, book, doc, toc: doc.rendered.headings }));
  }
  const downloadAction = book.download ? `<a class="button" href="${book.download}">EPUB 내려받기</a>` : "";
  const publicationState = book.status === "draft" ? "품질 검증 중인 공개 프리뷰" : "계속 개정되는 공개판";
  const bookBody = `<section class="book-hero"><p class="eyebrow">LLM 시스템 메커니즘 · ${book.number}권</p><h1>${esc(book.manifest.title)}</h1><p class="lead">${esc(book.manifest.subtitle || book.manifest.description)}</p><div class="actions"><a class="button primary" href="${book.docs[0]?.route}">처음부터 읽기</a>${downloadAction}</div><dl><div><dt>구성</dt><dd>${book.parts.length}부 · ${book.docs.length}개 문서</dd></div><div><dt>언어</dt><dd>한국어</dd></div><div><dt>상태</dt><dd>${publicationState}</dd></div></dl></section><div class="parts-grid">${book.parts.map(p => partCard(book,p)).join("")}</div>`;
  write(`books/${book.id}/index.html`, shell({ title: book.manifest.title, description: book.manifest.description, body: bookBody, canonical: book.route, book }));
  for (const part of book.parts) {
    const body = `<article class="part-opener"><p class="eyebrow">${book.number}권 · ${part.index}부</p><h1>${esc(part.title)}</h1><p class="lead">${esc(part.promise || "")}</p><div class="contract-grid"><section><h2>핵심 질문</h2><p>${esc(part.question || "")}</p></section><section><h2>선행 지식</h2><ul>${(part.prerequisites || []).map(x=>`<li>${esc(x)}</li>`).join("")}</ul></section><section><h2>이 부를 마치면</h2><p>${esc(part.exit_artifact || "")}</p></section><section><h2>다음 연결</h2><p>${esc(part.next_handoff || "")}</p></section></div><ol class="chapter-list">${part.docs.map(d=>`<li><a href="${d.route}"><span>${String(d.order).padStart(2,"0")}</span><strong>${esc(d.title)}</strong></a></li>`).join("")}</ol></article>`;
    write(`books/${book.id}/part-${part.index}/index.html`, shell({ title: part.title, description: part.promise, body, canonical: part.route, book }));
  }
}

const bookCards = siteConfig.books.map(entry => {
  const book = published.find(b => b.id === entry.id);
  return `<article class="series-card ${entry.status}" style="--book-accent:${esc(entry.accent)}"><p>${entry.number}권</p><h2>${esc(book?.manifest.title || entry.title)}</h2><h3>${esc(entry.short_title)}</h3><p>${esc(book?.manifest.description || entry.subtitle || "집필 준비 중")}</p>${book ? `<a href="${book.route}">온라인으로 읽기 <span>→</span></a>` : '<span class="status">준비 중</span>'}</article>`;
}).join("");
const home = `<section class="home-hero"><p class="eyebrow">LLM 시스템 메커니즘</p><h1>표면의 API에서<br><em>작동 원리</em>까지.</h1><p class="lead">서빙, 훈련, 에이전트를 따로 외우지 않고 실제 코드·수학·상태·하드웨어의 연결로 읽습니다.</p><div class="actions"><a class="button primary" href="/books/volume-1/">1권 읽기</a><a class="button" href="/books/volume-2/">2권 읽기</a><a class="button" href="/books/volume-3/">3권 프리뷰</a><a class="button" href="/search/">전체 검색</a></div></section><section class="series"><div class="section-heading"><p>THE SERIES</p><h2>세 권으로 이어지는 하나의 지도</h2></div><div class="series-grid">${bookCards}</div></section><section class="principles"><h2>필요할 때 바로 원문까지</h2><div><article><strong>01</strong><h3>함수와 상태</h3><p>설정 이름에서 멈추지 않고 실제 consumer와 상태 전이를 추적합니다.</p></article><article><strong>02</strong><h3>논문과 구현</h3><p>논문의 전제와 고정된 코드 좌표를 같은 문맥에서 연결합니다.</p></article><article><strong>03</strong><h3>실패와 검증</h3><p>정답만 나열하지 않고 최초 불일치와 반증 절차를 남깁니다.</p></article></div></section>`;
write("index.html", shell({ title: siteConfig.site.title, description: siteConfig.site.description, body: home, canonical: "/" }));
write("books/index.html", shell({ title: "전권", body: `<section class="listing"><p class="eyebrow">BOOKS</p><h1>LLM 시스템 메커니즘 전권</h1><div class="series-grid">${bookCards}</div></section>`, canonical: "/books/" }));

const authorMd = read("AUTHOR.md");
const dummyBook = { id:"author", content_root:"", parts:[] };
const dummyDoc = { slug:"author", logical:"AUTHOR.md" };
const author = renderMarkdown(dummyBook, dummyDoc, authorMd);
write("author/index.html", shell({ title: "저자", description: "Jioh L. Jung 저자 소개", body: `<article class="author-page">${author.html}</article>`, canonical: "/author/", toc: author.headings }));
write("search/index.html", shell({ title: "전체 검색", body: `<section class="search-page"><p class="eyebrow">SEARCH</p><h1>세 권을 한 번에 검색하기</h1><p>한국어 개념, 함수명, CUDA 심벌, 논문 제목과 AgentRun 상태를 검색할 수 있습니다.</p><div id="search" role="search"></div><script>window.addEventListener('DOMContentLoaded',()=>{if(window.PagefindUI)new PagefindUI({element:'#search',showSubResults:true,showImages:false,translations:{placeholder:'예: PagedAttention, Muon, AgentRun, 체크포인트 복구'}})});</script><script src="/pagefind/pagefind-ui.js"></script></section>`, canonical: "/search/" }));
write("404.html", shell({ title: "페이지를 찾을 수 없습니다", body: `<section class="not-found"><h1>404</h1><p>주소가 바뀌었거나 아직 출판되지 않은 페이지입니다.</p><a class="button primary" href="/">홈으로</a></section>`, canonical: "/404.html" }));

const urls = ["/", "/books/", "/author/", "/search/", ...published.flatMap(b => [b.route, ...b.parts.map(p=>p.route), ...b.docs.map(d=>d.route)])];
write("sitemap.xml", `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls.map(u=>`<url><loc>${siteConfig.site.url}${u}</loc></url>`).join("")}</urlset>`);
write("robots.txt", `User-agent: *\nAllow: /\nSitemap: ${siteConfig.site.url}/sitemap.xml\n`);
write("site-manifest.json", JSON.stringify({ generatedFrom: siteConfig.books.map(b=>b.id), books: published.map(b=>({id:b.id, parts:b.parts.length, documents:b.docs.length})), routes: urls.length }, null, 2));

for (const rel of ["public/downloads/volume-1-cuda-llm-serving-ko.epub", "public/downloads/volume-2-finetuning-mechanisms-ko.epub"]) copy(rel, rel.replace(/^public\//, ""));
// Volume 3 is rebuilt from its manuscript during editorial work. Copy the
// canonical dist artifact directly so the deployed download cannot lag behind
// the EPUB that was just validated.
copy("content/volume-3/dist/llm-multi-agent-mechanisms-ko-draft.epub", "downloads/volume-3-multi-agent-mechanisms-ko-draft.epub");
copyTree("labs/volume-3", "labs/volume-3");
copy("labs/volume-3/runtime-cancellation-observability/README.md", "labs/volume-3/runtime-cancellation-observability/README.txt");
copy("public/fonts/NotoSansKR-VF.ttf", "assets/fonts/NotoSansKR-VF.ttf");
copy("public/fonts/OFL.txt", "assets/fonts/OFL.txt");
copy("node_modules/mermaid/dist/mermaid.min.js", "assets/mermaid.min.js");
copy("node_modules/katex/dist/katex.min.css", "assets/katex.min.css");
for (const font of fs.readdirSync(path.join(ROOT,"node_modules/katex/dist/fonts"))) copy(`node_modules/katex/dist/fonts/${font}`, `assets/fonts/${font}`);
copy("node_modules/highlight.js/styles/github-dark.min.css", "assets/highlight.css");
copy("site-src/site.css", "assets/site.css");
copy("site-src/site.js", "assets/site.js");
copy("site-src/favicon.svg", "assets/favicon.svg");
copy(".nojekyll", ".nojekyll");

console.log(`built ${published.length} books, ${published.reduce((n,b)=>n+b.docs.length,0)} documents, ${urls.length} routes`);
