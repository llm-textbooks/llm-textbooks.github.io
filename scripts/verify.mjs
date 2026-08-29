import fs from "node:fs";
import path from "node:path";
import { load } from "cheerio";

const ROOT = path.resolve(import.meta.dirname, "..");
const SITE = path.join(ROOT, "_site");
const htmlFiles = [];
const walk = dir => {
  for (const item of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, item.name);
    if (item.isDirectory()) walk(p); else if (p.endsWith(".html")) htmlFiles.push(p);
  }
};
walk(SITE);

const errors = [];
const stats = { html: htmlFiles.length, links: 0, external: 0, tables: 0, code: 0, mermaid: 0, headings: 0, images: 0 };
const existsForUrl = href => {
  const clean = decodeURI(href.split(/[?#]/)[0]);
  if (!clean || clean === "/") return fs.existsSync(path.join(SITE, "index.html"));
  const rel = clean.replace(/^\//, "");
  return fs.existsSync(path.join(SITE, rel)) || fs.existsSync(path.join(SITE, rel, "index.html"));
};
const idCache = new Map();
const idsForUrl = href => {
  const clean = decodeURI(href.split(/[?#]/)[0]);
  const rel = clean.replace(/^\//, "");
  const target = clean === "/" ? path.join(SITE,"index.html") : fs.existsSync(path.join(SITE,rel)) && path.join(SITE,rel).endsWith(".html") ? path.join(SITE,rel) : path.join(SITE,rel,"index.html");
  if (!fs.existsSync(target)) return new Set();
  if (!idCache.has(target)) { const q=load(fs.readFileSync(target,"utf8")); idCache.set(target,new Set(q("[id]").map((_,e)=>q(e).attr("id")).get())); }
  return idCache.get(target);
};

for (const file of htmlFiles) {
  const rel = path.relative(SITE, file);
  const $ = load(fs.readFileSync(file, "utf8"));
  if ($("html").attr("lang") !== "ko-KR") errors.push(`${rel}: missing lang=ko-KR`);
  if ($("main#main").length !== 1) errors.push(`${rel}: missing unique main`);
  if (!$('meta[name="description"]').attr("content")) errors.push(`${rel}: missing description`);
  const ids = new Set();
  $("[id]").each((_, el) => { const id = $(el).attr("id"); if (ids.has(id)) errors.push(`${rel}: duplicate id ${id}`); ids.add(id); });
  stats.headings += $("h1,h2,h3,h4,h5,h6").length;
  stats.tables += $("table").length;
  stats.code += $(".code-block").length;
  stats.mermaid += $("pre.mermaid").length;
  stats.images += $("img,svg").length;
  $("table").each((_, el) => { if (!$(el).parent().hasClass("table-scroll")) errors.push(`${rel}: table without scroll wrapper`); });
  $("img").each((_, el) => { if (!$(el).attr("alt")) errors.push(`${rel}: image without alt`); });
  $("a[href]").each((_, el) => {
    stats.links += 1;
    const href = $(el).attr("href");
    if (/^(https?:|mailto:|tel:)/.test(href)) { stats.external += 1; return; }
    if (href.startsWith("#")) { if (href.length > 1 && !ids.has(decodeURIComponent(href.slice(1)))) errors.push(`${rel}: broken local fragment ${href}`); return; }
    if (href.includes(".md")) errors.push(`${rel}: leaked markdown href ${href}`);
    if (href.startsWith("/") && !existsForUrl(href)) errors.push(`${rel}: broken internal href ${href}`);
    if (href.startsWith("/") && href.includes("#") && existsForUrl(href)) {
      const fragment = decodeURIComponent(href.split("#")[1]);
      if (fragment && !idsForUrl(href).has(fragment)) errors.push(`${rel}: broken cross-page fragment ${href}`);
    }
  });
}

for (const asset of ["assets/site.css","assets/site.js","assets/favicon.svg","assets/mermaid.min.js","assets/katex.min.css","pagefind/pagefind.js","pagefind/pagefind-ui.js","downloads/volume-1-cuda-llm-serving-ko.epub","downloads/volume-2-finetuning-mechanisms-ko.epub","sitemap.xml","robots.txt","site-manifest.json","source-link-report.json"]) {
  if (!fs.existsSync(path.join(SITE, asset))) errors.push(`missing artifact ${asset}`);
}

const manifest = JSON.parse(fs.readFileSync(path.join(SITE,"site-manifest.json"),"utf8"));
const sourceReport = JSON.parse(fs.readFileSync(path.join(SITE,"source-link-report.json"),"utf8"));
if (sourceReport.counts.fatal !== 0) errors.push(`source-link audit has ${sourceReport.counts.fatal} fatal finding(s)`);
const counts = Object.fromEntries(manifest.books.map(b => [b.id,b.documents]));
if (counts["volume-1"] !== 78) errors.push(`volume-1 expected 78 documents, got ${counts["volume-1"]}`);
if (counts["volume-2"] !== 53) errors.push(`volume-2 expected 53 documents, got ${counts["volume-2"]}`);
if (stats.mermaid !== 106) errors.push(`expected 106 Mermaid diagrams, got ${stats.mermaid}`);
if (stats.tables < 340) errors.push(`expected at least 340 rendered tables, got ${stats.tables}`);
if (stats.external < 1500) errors.push(`expected at least 1500 external links, got ${stats.external}`);

console.log(JSON.stringify({ ...stats, books: counts, errors: errors.length }, null, 2));
if (errors.length) {
  console.error(errors.slice(0, 100).join("\n"));
  process.exit(1);
}
console.log("site verification PASS");
