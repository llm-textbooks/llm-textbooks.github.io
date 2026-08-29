import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const files = [];
const walk = dir => {
  for (const item of fs.readdirSync(dir,{withFileTypes:true})) {
    const p=path.join(dir,item.name);
    if(item.isDirectory()) walk(p); else if(p.endsWith(".md")) files.push(p);
  }
};
walk(path.join(ROOT,"content"));

const records=[];
const plainArxiv=[];
const coordinateCandidates=[];
const fatal=[];
const linkRe=/\[[^\]]*\]\((https?:\/\/[^)\s]+)(?:\s+"[^"]*")?\)|<(https?:\/\/[^>\s]+)>/g;
const arxivIdRe=/(?<![\w.])(?:arXiv[: ]*)?([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?/gi;
const coordRe=/[A-Za-z0-9_./-]+\.(?:py|pyi|cpp|cc|c|cu|cuh|h|hpp|rs|go|java|js|ts):(?:L)?[0-9]+(?:[-–:](?:L)?[0-9]+)?/g;

for(const file of files){
  const rel=path.relative(ROOT,file).replaceAll(path.sep,"/");
  let fenced=false;
  for(const [i,line] of fs.readFileSync(file,"utf8").split(/\r?\n/).entries()){
    if(/^\s*```/.test(line)){fenced=!fenced;continue;}
    if(fenced) continue;
    const urls=[];
    for(const m of line.matchAll(linkRe)) urls.push(m[1]||m[2]);
    for(const url of urls){
      let kind="other",stability="context",status="pass";
      if(/github\.com/.test(url)){
        kind="code";
        if(/\/(blob|tree)\/(main|master|HEAD)(\/|$)/i.test(url)){stability="mutable";status="fail";fatal.push({file:rel,line:i+1,url,reason:"mutable-github-ref"});}
        else if(/\/blob\/[0-9a-f]{40}\//i.test(url)&&/#L[0-9]+(?:-L[0-9]+)?$/.test(url)){stability="exact-code";}
        else if(/\/(blob|tree)\/[0-9a-f]{40}\//i.test(url)){stability="commit-pinned";}
        else if(/\/(issues|pull|commit)\//.test(url)){stability="event-pinned";}
        else stability="repository-context";
      } else if(/arxiv\.org/.test(url)){kind="paper";stability=/\/abs\/[0-9]{4}\.[0-9]{4,5}v[0-9]+/.test(url)?"version-pinned":"work-pinned";}
      else if(/doi\.org/.test(url)){kind="paper";stability="doi";}
      else if(/huggingface\.co/.test(url)){kind="model-or-data";stability=/\/(blob|tree)\/[0-9a-f]{40}\//.test(url)?"commit-pinned":"mutable-card";}
      else if(/docs\.|developer\.|nvidia\.com|pytorch\.org/.test(url)){kind="official-docs";stability="documentation";}
      records.push({file:rel,line:i+1,url,kind,stability,status});
    }
    for(const m of line.matchAll(arxivIdRe)){
      const id=m[1],version=m[2]||"";
      if(!urls.some(url=>url.includes(`arxiv.org/abs/${id}`)||url.includes(`arxiv.org/pdf/${id}`))) plainArxiv.push({file:rel,line:i+1,id:id+version});
    }
    const coords=[...line.matchAll(coordRe)].map(m=>m[0]);
    if(coords.length&&!urls.some(url=>/github\.com\/.+\/blob\/[0-9a-f]{40}\//.test(url))) coordinateCandidates.push({file:rel,line:i+1,coordinates:coords});
  }
}

const counts={
  files:files.length, links:records.length, uniqueLinks:new Set(records.map(r=>r.url)).size,
  exactCode:records.filter(r=>r.stability==="exact-code").length,
  commitPinned:records.filter(r=>r.stability==="commit-pinned").length,
  papers:records.filter(r=>r.kind==="paper").length,
  modelOrData:records.filter(r=>r.kind==="model-or-data").length,
  officialDocs:records.filter(r=>r.kind==="official-docs").length,
  plainArxiv:plainArxiv.length, coordinateCandidates:coordinateCandidates.length, fatal:fatal.length
};
const report={schemaVersion:1,counts,fatal,plainArxiv,coordinateCandidates,records};
fs.mkdirSync(path.join(ROOT,"_site"),{recursive:true});
fs.writeFileSync(path.join(ROOT,"_site/source-link-report.json"),JSON.stringify(report,null,2));
console.log(`source links: ${JSON.stringify(counts)}`);
if(fatal.length){console.error(JSON.stringify(fatal,null,2));process.exit(1);}
