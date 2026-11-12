# Browser QA

2026-11-09, local file preview with existing Chromium headless shell1243. No system libraries or project dependencies were installed. Missing runtime libraries were extracted from Ubuntu Jammy debs into `/tmp/crop-daily-browser-libs` only. Browser processes were closed after testing.

Final run exposes only DejaVu as system fallback. Korean rendering therefore uses the HTML's embedded `ReportKorean` WOFF; both `document.fonts.check('16px ReportKorean')` and font-face loaded state passed.

- Desktop1440×1000 and mobile390×844: no document horizontal overflow, including all details expanded.
- All17 experiment cards present; family filter, text search, reset work.
- All4 protocol selectors update captions, table rows and chart bars consistently (10/10/9/9).
- Both embedded PNGs decode successfully. No JavaScript console errors.
- Before-print expands all details; after-print exactly restores their prior states.
- Mobile header navigation intentionally scrolls horizontally; offscreen navigation anchors do not create document overflow.

Artifacts: `desktop_1440.png`, `mobile_390.png`, `desktop_expanded.png`, `mobile_expanded.png`, `browser_qa.json`. The JSON contains the tested HTML SHA256. `browser_qa.py` preserves the raw-CDP checks; rerunning needs Chromium's local runtime libraries available and a browser listening on localhost9229.
