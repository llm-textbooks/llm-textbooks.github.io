import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";

const ROOT = path.resolve(import.meta.dirname, "..");
const bookPath = path.join(ROOT, "content/volume-3/book.yaml");
const book = YAML.parse(fs.readFileSync(bookPath, "utf8"));
const failures = [];
const rows = [];
const forbidden = [
  [/\bontology\b/iu, "reader-facing-ontology-term"],
  [/온톨로지/u, "reader-facing-ontology-term"],
  [/\/home\/ziozzang/u, "local-absolute-path"],
  [/llmis-(?:core|code|agent|prov|sw):/u, "internal-curie"],
  [/물론입니다|요청하신 내용을 정리하면|도움이 되셨길 바랍니다|추가 질문이 있으시면/u, "chatbot-frame-residue"],
  [/되어진다|되어지는|보여질 수/u, "double-passive"],
  [/시사하는 바가 크다|주목할 만하다|결론적으로|정리하자면|요약하면/u, "signature-conclusion-phrase"],
  [/단순한? .{0,30}(?:를|을) 넘어/u, "scope-escalation-cliche"],
];

for (const relative of book.chapters) {
  const file = path.join(ROOT, "content/volume-3", relative);
  if (!fs.existsSync(file)) {
    failures.push({ file: relative, reason: "missing" });
    continue;
  }
  const text = fs.readFileSync(file, "utf8");
  const metrics = {
    file: relative,
    characters: [...text].length,
    h2: (text.match(/^## /gmu) || []).length,
    mermaid: (text.match(/```mermaid/g) || []).length,
    tables: (text.match(/^\|.*\|$/gmu) || []).length,
    codeBlocks: Math.floor((text.match(/^```/gmu) || []).length / 2),
    links: (text.match(/https?:\/\//g) || []).length,
    checklists: (text.match(/^- \[[ xX]\]/gmu) || []).length,
    antithesis: (text.match(/(?:아니라|것은 아니다|것이 아니다)/gu) || []).length,
    splitFocus: (text.match(/(?:핵심은|문제는|관건은|중요한 것은)/gu) || []).length,
    conclusionMarkers: (text.match(/(?:결국|따라서|그러므로|이를 통해)/gu) || []).length,
  };
  rows.push(metrics);
  const minimumCharacters = relative.startsWith("chapters/00-") ? 6000 : 9000;
  if (metrics.characters < minimumCharacters) failures.push({ file: relative, reason: "too-short", value: metrics.characters, minimum: minimumCharacters });
  if (metrics.h2 < 5) failures.push({ file: relative, reason: "insufficient-section-structure", value: metrics.h2 });
  if (metrics.mermaid < 1) failures.push({ file: relative, reason: "missing-mermaid" });
  if (metrics.tables < 2) failures.push({ file: relative, reason: "missing-comparison-table" });
  if (metrics.codeBlocks < 1) failures.push({ file: relative, reason: "missing-code-or-lab" });
  if (metrics.links < 3) failures.push({ file: relative, reason: "insufficient-primary-links", value: metrics.links });
  if (!/(체크리스트|점검표|checklist)/iu.test(text)) failures.push({ file: relative, reason: "missing-checklist-heading" });
  if (!/(비보장|보장|한계|범위 밖|증명하지|뜻하지 않|아니다|unsupported)/iu.test(text)) failures.push({ file: relative, reason: "missing-non-guarantee-boundary" });
  for (const [pattern, reason] of forbidden) {
    if (pattern.test(text)) failures.push({ file: relative, reason });
  }
}

const numbered = book.chapters.filter((relative) => /^chapters\/(?:0[1-9]|[1-3][0-9]|4[01])-/.test(relative));
if (book.chapters.length !== 43) failures.push({ file: "book.yaml", reason: "unexpected-document-count", value: book.chapters.length, expected: 43 });
if (numbered.length !== 41) failures.push({ file: "book.yaml", reason: "unexpected-numbered-chapter-count", value: numbered.length, expected: 41 });

const report = { schemaVersion: 1, chapters: rows.length, failures: failures.length, rows, findings: failures };
fs.writeFileSync(path.join(ROOT, "_site/volume3-preview-editorial-report.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ chapters: rows.length, failures: failures.length }, null, 2));
if (failures.length) {
  console.error(JSON.stringify(failures, null, 2));
  process.exit(1);
}
