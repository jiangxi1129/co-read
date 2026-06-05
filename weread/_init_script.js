// Injected before any page JS via context.add_init_script.
// Watches DOM for #preRenderContent — weread renders chapter text there,
// rasterizes to canvas, then removes it. We clone innerHTML on first sight.
(function () {
  if (window.__weread_init_done) return;
  window.__weread_init_done = true;
  window.__weread_captured = [];

  function tryCapture(el) {
    if (!el || !el.innerHTML) return;
    window.__weread_captured.push({ ts: Date.now(), html: el.innerHTML });
  }

  // 1. Catch the moment the element is added
  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const node of m.addedNodes) {
        if (!(node instanceof Element)) continue;
        if (node.id === 'preRenderContent') {
          tryCapture(node);
        } else if (node.querySelector) {
          const inner = node.querySelector('#preRenderContent');
          if (inner) tryCapture(inner);
        }
      }
      // 2. Also catch character-data / subtree updates of an existing node
      if (m.type === 'childList' && m.target && m.target.id === 'preRenderContent') {
        tryCapture(m.target);
      }
    }
  });

  function start() {
    obs.observe(document.documentElement || document, {
      childList: true,
      subtree: true,
      characterData: false,
    });
    // 3. Poll fallback (in case observer misses a fast insert+remove)
    const poll = setInterval(() => {
      const el = document.getElementById('preRenderContent');
      if (el) tryCapture(el);
    }, 200);
    setTimeout(() => clearInterval(poll), 30000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();
