#!/usr/bin/env python3
"""
packagist.py - renderiza o card de downloads do Packagist como SVG. So stdlib.

    python scripts/packagist.py --vendor gsferro --out assets

Escreve <out>/card-packagist-{dark,light}.svg.

Os numeros vem da API publica do Packagist a cada execucao: a lista de pacotes
do vendor em uma chamada, e o stats.json leve de cada pacote (total, mensal,
diario). Nada de badge de terceiro que pode sair do ar e levar a secao junto.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "packagist.py (github.com/gsferro/gsferro)"}
API = "https://packagist.org"

THEMES = {
    "dark": {
        "bg": "#0d1117", "border": "#30363d", "title": "#FF4D3D",
        "text": "#c9d1d9", "muted": "#8b949e", "value": "#e6edf3",
        "leader": "#262c36", "accent": "#F59E0B",
    },
    "light": {
        "bg": "#ffffff", "border": "#d0d7de", "title": "#C7301F",
        "text": "#1f2328", "muted": "#57606a", "value": "#1f2328",
        "leader": "#e1e6eb", "accent": "#B45309",
    },
}

FONT = "ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"


def br(value: int) -> str:
    """Formata milhar no padrao brasileiro: 2408 -> 2.408."""
    return f"{value:,}".replace(",", ".")


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def text_width(s: str, size: float) -> float:
    return len(s) * size * 0.53


def get(path: str):
    req = urllib.request.Request(API + path, headers=dict(UA))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def collect(vendor: str) -> tuple[list[dict], dict]:
    """Retorna (pacotes ordenados por total, agregados)."""
    names = get(f"/packages/list.json?vendor={vendor}")["packageNames"]
    packages: list[dict] = []
    for name in names:
        try:
            stats = get(f"/packages/{name}/stats.json")["downloads"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as exc:
            # Um pacote fora do ar nao pode derrubar o card inteiro.
            print(f"  !! {name}: {exc}", file=sys.stderr)
            continue
        packages.append({
            "name": name.split("/", 1)[-1],
            "total": stats.get("total", 0),
            "monthly": stats.get("monthly", 0),
            "daily": stats.get("daily", 0),
        })
    packages.sort(key=lambda p: p["total"], reverse=True)
    totals = {
        "packages": len(packages),
        "total": sum(p["total"] for p in packages),
        "monthly": sum(p["monthly"] for p in packages),
        "daily": sum(p["daily"] for p in packages),
    }
    return packages, totals


def leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1 + 6:
        return ""
    xs = [x1 + i * 5 for i in range(int((x2 - x1) / 5))]
    return "".join(f"M{x:.0f} {y:.0f}h1.6" for x in xs)


def render(vendor: str, packages: list[dict], totals: dict, theme: str,
           top: int) -> str:
    c = THEMES[theme]
    pad, W = 22, 480
    shown = packages[:top]
    # A altura mede ate a baseline da ultima linha, nao ate uma linha alem
    # dela: `len(shown) * 21` deixava 21 px de faixa morta no rodape.
    H = pad + 52 + 17 + 26 + (len(shown) - 1) * 21 + pad

    out = [
        f'<text x="{pad}" y="{pad + 14}" font-size="15" font-weight="700" '
        f'fill="{c["title"]}">{esc(vendor)}</text>',
        f'<text x="{W - pad}" y="{pad + 14}" font-size="11" text-anchor="end" '
        f'fill="{c["muted"]}">packagist</text>',
        f'<line x1="{pad}" y1="{pad + 26}" x2="{W - pad}" y2="{pad + 26}" '
        f'stroke="{c["border"]}"/>',
    ]

    tiles = [
        (br(totals["total"]), "downloads"),
        (br(totals["packages"]), "pacotes"),
        (br(totals["monthly"]), "no mês"),
    ]
    tw = (W - 2 * pad) / 3
    for i, (value, label) in enumerate(tiles):
        cx = pad + i * tw
        out.append(
            f'<text x="{cx:.0f}" y="{pad + 52}" font-size="23" font-weight="700" '
            f'fill="{c["value"]}">{esc(value)}</text>'
        )
        out.append(
            f'<text x="{cx:.0f}" y="{pad + 69}" font-size="10.5" '
            f'fill="{c["muted"]}">{esc(label)}</text>'
        )

    y = pad + 52 + 17 + 26
    for pkg in shown:
        count = br(pkg["total"])
        name_w = text_width(pkg["name"], 11.5)
        count_w = text_width(count, 11.5)
        out.append(
            f'<text x="{pad}" y="{y:.0f}" font-size="11.5" fill="{c["text"]}">'
            f'{esc(pkg["name"])}</text>'
        )
        out.append(
            f'<path d="{leader(pad + name_w + 8, W - pad - count_w - 8, y - 4)}" '
            f'stroke="{c["leader"]}" stroke-width="1" fill="none"/>'
        )
        out.append(
            f'<text x="{W - pad}" y="{y:.0f}" font-size="11.5" text-anchor="end" '
            f'fill="{c["accent"]}" font-weight="600">{esc(count)}</text>'
        )
        y += 21

    body = "".join(out)
    label = f"Downloads dos pacotes {vendor} no Packagist"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="{esc(label)}" '
        f'font-family="{FONT}">'
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" '
        f'fill="{c["bg"]}" stroke="{c["border"]}"/>'
        f"{body}</svg>"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vendor", default="gsferro")
    p.add_argument("--out", type=Path, default=Path("assets"))
    p.add_argument("--top", type=int, default=6,
                   help="quantos pacotes listar abaixo dos totais")
    args = p.parse_args(argv)

    packages, totals = collect(args.vendor)
    if not packages:
        print("nenhum pacote lido, card nao foi escrito", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for theme in ("dark", "light"):
        dest = args.out / f"card-packagist-{theme}.svg"
        dest.write_text(
            render(args.vendor, packages, totals, theme, args.top), encoding="utf-8"
        )
    print(
        f"wrote card-packagist-*.svg  ({totals['packages']} pacotes, "
        f"{br(totals['total'])} downloads, {br(totals['monthly'])} no mes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
