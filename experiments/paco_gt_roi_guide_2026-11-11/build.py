"""Build a single-file offline guide with verified data, SVG interactions, embedded Korean font."""
import base64
import json
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
FONT = Path('/home/irteam/.cache/gradient-memory-fonts/NanumGothic-Regular.ttf')


def main():
    ASSETS.mkdir(exist_ok=True)
    template = (OUT / 'report.html').read_text()
    script = (OUT / 'report.js').read_text()
    data = json.loads((OUT / 'data.json').read_text())
    data_text = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    text = template + script + data_text
    opts = subset.Options()
    opts.flavor = 'woff'
    font = TTFont(FONT)
    coverage = font.getBestCmap()
    needed = {ord(c) for c in text if '\uac00' <= c <= '\ud7a3'}
    assert needed <= set(coverage), sorted(needed - set(coverage))
    sub = subset.Subsetter(options=opts)
    sub.populate(text=text)
    sub.subset(font)
    font.flavor = 'woff'
    font.save(ASSETS / 'GuideKorean.woff')
    encoded = base64.b64encode((ASSETS / 'GuideKorean.woff').read_bytes()).decode()
    css = '@font-face{font-family:GuideKorean;src:url(data:font/woff;base64,' + encoded + ') format("woff");font-style:normal;font-weight:400;font-display:swap}'
    license_text = (ASSETS / 'FONT_LICENSE.txt').read_text()
    html = template.replace('/*__FONT__*/', css).replace('/*__DATA__*/', data_text).replace('/*__SCRIPT__*/', script).replace('/*__LICENSE__*/', license_text)
    (OUT / 'index.html').write_text(html)
    print(json.dumps(dict(html_bytes=len(html.encode()),embedded_images=len(data['images']),hangul_characters=len(needed),font_bytes=(ASSETS/'GuideKorean.woff').stat().st_size)))


if __name__ == '__main__':
    main()
