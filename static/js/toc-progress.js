(() => {
  const nav = document.querySelector('.toc-disclosure nav');
  if (!nav) return;
  const disclosure = nav.closest('details');
  const entries = Array.from(nav.querySelectorAll('a[href^="#"]')).flatMap(link => {
    let heading;
    try { heading = document.getElementById(decodeURIComponent(link.hash.slice(1))); }
    catch { return []; }
    return heading ? [{ link, heading }] : [];
  });
  if (!entries.length) return;

  const marker = document.createElement('span');
  marker.className = 'toc-position';
  marker.setAttribute('aria-hidden', 'true');
  nav.append(marker);
  let active;
  let pending = false;

  function update() {
    pending = false;
    let current = entries[0];
    const readingLine = Math.min(160, window.innerHeight * 0.25);
    for (const entry of entries) {
      if (entry.heading.getBoundingClientRect().top <= readingLine) current = entry;
    }
    // Short final sections may never reach the reading line.
    if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      current = entries[entries.length - 1];
    }
    if (active !== current) {
      active?.link.removeAttribute('aria-current');
      current.link.setAttribute('aria-current', 'location');
      active = current;
    }
    if (!disclosure.open) return;
    const box = current.link.getBoundingClientRect();
    marker.style.height = `${box.height}px`;
    marker.style.transform = `translateY(${box.top - nav.getBoundingClientRect().top}px)`;
  }

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(update);
  }
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  window.addEventListener('hashchange', schedule);
  window.addEventListener('load', schedule);
  disclosure.addEventListener('toggle', schedule);
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(schedule);
    observer.observe(nav);
    const content = document.querySelector('.post-content');
    if (content) observer.observe(content);
  }
  document.fonts?.ready.then(schedule);
  update();
})();
