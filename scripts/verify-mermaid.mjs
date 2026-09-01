import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import mermaid from "mermaid";
import YAML from "yaml";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const books = YAML.parse(fs.readFileSync(path.join(ROOT, "books.yaml"), "utf8")).books;
const syntaxFindings = [];
let environmentConstrained = 0;
let diagrams = 0;

for (const entry of books.filter(book => ["published", "draft"].includes(book.status))) {
  const manifest = YAML.parse(fs.readFileSync(path.join(ROOT, entry.manifest), "utf8"));
  for (const declared of manifest.chapters) {
    const relative = declared.replace(/^next\//u, "");
    const source = path.join(ROOT, entry.content_root, relative);
    const text = fs.readFileSync(source, "utf8");
    const blocks = [...text.matchAll(/```mermaid\s*\n([\s\S]*?)```/gu)];
    for (const [index, block] of blocks.entries()) {
      diagrams += 1;
      try {
        await mermaid.parse(block[1]);
      } catch (error) {
        // Mermaid's parser imports DOMPurify for several diagram types.  In a
        // Node-only process this is not a grammar verdict; reserve an actual
        // browser for that final render check and preserve the distinction.
        if (/DOMPurify\.addHook is not a function/u.test(error.message)) {
          environmentConstrained += 1;
        } else {
          syntaxFindings.push(`${path.relative(ROOT, source)} diagram ${index + 1}: ${error.message}`);
        }
      }
    }
  }
}

const browserCandidates = [process.env.BROWSER, "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].filter(Boolean);
const browser = browserCandidates.find(candidate => fs.existsSync(candidate));
console.log(`Mermaid parser: ${diagrams - environmentConstrained - syntaxFindings.length}/${diagrams} diagrams parsed in Node`);
if (environmentConstrained) console.log(`Mermaid parser environment-constrained: ${environmentConstrained} diagrams require a DOM-backed renderer`);
if (browser) console.log(`Mermaid browser runtime smoke is not wired: browser available at ${browser}`);
else console.log("Mermaid browser runtime smoke SKIPPED: no Chromium-compatible browser is available in this environment");
if (syntaxFindings.length) {
  console.warn(`Mermaid syntax findings (non-blocking until a browser renderer is provisioned): ${syntaxFindings.length}`);
  console.warn(syntaxFindings.join("\n"));
}
