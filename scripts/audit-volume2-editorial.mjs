import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const dir = path.join(ROOT, "content/volume-2/chapters");
const files = fs.readdirSync(dir).filter(name => /^\d\d-.+\.md$/.test(name) && !name.startsWith("00-")).sort();
const rows = [];

for (const name of files) {
  const source = fs.readFileSync(path.join(dir, name), "utf8");
  const count = re => [...source.matchAll(re)].length;
  rows.push({
    chapter: name,
    mermaid: count(/^```mermaid\s*$/gm),
    tables: count(/^\|.+\|\s*$/gm),
    immutableCodeLinks: count(/https:\/\/github\.com\/[^\s)]+\/blob\/[0-9a-f]{40}\/[^\s)]+#L\d+(?:-L\d+)?/g),
    equations: count(/^\$\$|^\\\[/gm),
    goldenRun: /GR-001|GoldenRunID/.test(source),
    falsifier: /반증|최초 불일치|failure injection|실패 주입/i.test(source),
  });
}

const failures = [];
for (const row of rows) {
  if (row.mermaid < 1) failures.push(`${row.chapter}: Mermaid 실행 지도가 없음`);
  if (row.tables < 3) failures.push(`${row.chapter}: 상태·shape 표가 3개 미만`);
  if (!row.goldenRun) failures.push(`${row.chapter}: Golden Run 연결이 없음`);
  if (!row.falsifier) failures.push(`${row.chapter}: 반증/최초 불일치 절차가 없음`);
}

const report = { schemaVersion: 1, chapters: rows.length, failures: failures.length, rows, findings: failures };
fs.mkdirSync(path.join(ROOT, "_site"), { recursive: true });
fs.writeFileSync(path.join(ROOT, "_site/volume2-editorial-report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ chapters: rows.length, failures: failures.length }, null, 2));
if (process.argv.includes("--strict") && failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
