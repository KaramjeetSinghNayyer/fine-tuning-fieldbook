(() => {
  const key = 'fieldbook-theme';
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  let saved = null;
  try { saved = localStorage.getItem(key); } catch (_) {}
  let chosen = saved === 'dark' || saved === 'light';
  const root = document.documentElement;
  function apply(dark) {
    root.dataset.theme = dark ? 'dark' : 'light';
    const button = document.getElementById('theme-toggle');
    if (button) {
      button.textContent = dark ? 'Light mode' : 'Dark mode';
      button.setAttribute('aria-pressed', String(dark));
      button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    }
  }
  apply(chosen ? saved === 'dark' : media.matches);
  document.addEventListener('DOMContentLoaded', () => {
    apply(root.dataset.theme === 'dark');
    document.getElementById('theme-toggle').addEventListener('click', () => {
      const dark = root.dataset.theme !== 'dark';
      chosen = true;
      apply(dark);
      try { localStorage.setItem(key, dark ? 'dark' : 'light'); } catch (_) {}
    });
  });
  media.addEventListener('change', event => { if (!chosen) apply(event.matches); });
})();
