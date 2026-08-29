(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.querySelector(".chapter-nav");
  toggle?.addEventListener("click", () => {
    const open = root.classList.toggle("nav-open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") { root.classList.remove("nav-open"); toggle?.setAttribute("aria-expanded", "false"); }
  });

  document.querySelectorAll(".copy-code").forEach(button => button.addEventListener("click", async () => {
    const code = button.closest(".code-block")?.querySelector("code")?.innerText || "";
    try {
      await navigator.clipboard.writeText(code);
      const old = button.textContent; button.textContent = "복사됨";
      setTimeout(() => { button.textContent = old; }, 1400);
    } catch { button.textContent = "복사 실패"; }
  }));

  document.querySelectorAll(".heading-anchor").forEach(anchor => anchor.addEventListener("click", async () => {
    if (!navigator.clipboard) return;
    try { await navigator.clipboard.writeText(anchor.href); } catch {}
  }));

  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "neutral",
      fontFamily: "Pretendard, Inter, system-ui, sans-serif",
      flowchart: { htmlLabels: false, useMaxWidth: true }
    });
    window.mermaid.run({ nodes: document.querySelectorAll(".mermaid"), suppressErrors: true }).catch(error => {
      console.warn("Mermaid rendering failed", error);
      document.querySelectorAll("[data-diagram]").forEach(el => el.classList.add("diagram-error"));
    });
  }
})();
