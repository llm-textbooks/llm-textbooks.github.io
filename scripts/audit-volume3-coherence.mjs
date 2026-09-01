import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";

const ROOT = path.resolve(import.meta.dirname, "..");
const contentRoot = path.join(ROOT, "content/volume-3");
const book = YAML.parse(fs.readFileSync(path.join(contentRoot, "book.yaml"), "utf8"));
const failures = [];
const rows = [];
const paragraphs = new Map();

function normalizeParagraph(text) {
  return text
    .replace(/[`*_\[\]()]/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

for (const relative of book.chapters) {
  const text = fs.readFileSync(path.join(contentRoot, relative), "utf8");
  const lines = text.split(/\r?\n/u);
  let inFence = false;
  let fenceCount = 0;
  let executableFences = 0;
  let illustrativeFences = 0;
  let paragraph = [];
  const flush = () => {
    if (!paragraph.length) return;
    const normalized = normalizeParagraph(paragraph.join(" "));
    paragraph = [];
    if (normalized.length < 140) return;
    const owners = paragraphs.get(normalized) || [];
    owners.push(relative);
    paragraphs.set(normalized, owners);
  };

  for (const line of lines) {
    if (line.startsWith("```")) {
      flush();
      inFence = !inFence;
      fenceCount += 1;
      if (inFence) {
        const language = line.slice(3).trim().toLowerCase();
        if (["bash", "sh", "python", "javascript", "typescript", "json"].includes(language)) executableFences += 1;
        else illustrativeFences += 1;
      }
      continue;
    }
    if (inFence || /^(?:#|\||-|\*|>)/u.test(line)) {
      flush();
      continue;
    }
    if (line.trim()) paragraph.push(line.trim());
    else flush();
  }
  flush();
  if (inFence || fenceCount % 2 !== 0) failures.push({ file: relative, reason: "unbalanced-code-fence" });

  const chapterRefs = [...text.matchAll(/(?<!권)(?<!부)(?<!-)\b(\d{1,2})장/gu)].map((match) => Number(match[1]));
  const invalidRefs = chapterRefs.filter((number) => number < 1 || number > 45);
  if (invalidRefs.length) failures.push({ file: relative, reason: "invalid-chapter-reference", values: invalidRefs });
  rows.push({ file: relative, chapterRefs: chapterRefs.length, executableFences, illustrativeFences });
}

for (const [paragraph, owners] of paragraphs.entries()) {
  const uniqueOwners = [...new Set(owners)];
  if (uniqueOwners.length > 1) {
    failures.push({ files: uniqueOwners, reason: "exact-cross-chapter-paragraph-duplicate", preview: paragraph.slice(0, 180) });
  }
}

const report = { schemaVersion: 1, chapters: rows.length, failures: failures.length, rows, findings: failures };
fs.writeFileSync(path.join(ROOT, "_site/volume3-coherence-report.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ chapters: rows.length, failures: failures.length }, null, 2));
if (failures.length) {
  console.error(JSON.stringify(failures, null, 2));
  process.exit(1);
}
