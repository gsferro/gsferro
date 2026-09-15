#!/usr/bin/env python3
"""Gera os banners animados do perfil (dark e light).

Rode a partir da raiz do repositorio:
    python scripts/banner/generate.py

O retrato vem de assets/source/avatar.png e passa por um dithering
Floyd-Steinberg de 1 bit. Os pontos resultantes viajam, por transporte
otimo, ate formarem as silhuetas do PHP, do Laravel e do Filament.
"""

from __future__ import annotations

import html

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/avatar.png"
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"
DATA = Path(__file__).resolve().parent / "data"

W, H = 1180, 610
LOOP_SECONDS = 14.2
INTRO_SECONDS = 3.2
TRAVELLER_COUNT = 900
SEED = 314159

# Recorte cabeca + ombros dentro do avatar 460x460, na proporcao do quadro
# VISUAL.MAP (300x340). Ajuste aqui se trocar a foto de origem.
CROP = (100, 22, 410, 374)

ROWS = [
    ("Subject", "Guilherme Ferro"),
    ("Role", "Desenvolvedor PHP / Laravel"),
    ("Origin", "Brasil"),
    ("Focus", "Pacotes open source easy"),
    ("Status", "Building + Shipping + Simplifying"),
    ("ToolChain", "PhpStorm / Docker / Git"),
    ("Core.Lang", "PHP / JavaScript / SQL"),
    ("Core.Framework", "Laravel / Filament / Livewire"),
    ("Core.Frontend", "Blade / Bootstrap / Alpine.js"),
    ("Core.Database", "MySQL / Postgres / SQL Server"),
    ("Core.Infra", "Docker / Composer / GitHub Actions"),
    ("Grid.Packagist", "packagist.org/packages/gsferro"),
    ("Grid.LinkedIn", "/in/guilherme-ferro"),
    ("Grid.GitHub", "gsferro"),
    ("Grid.Mail", "gsferroti+github@gmail.com"),
]

THEMES = {
    "dark": {
        "bg": "#0A0E14",
        "panel": "#0D1117",
        "panel2": "#11161F",
        "line": "#262C36",
        "muted": "#8B949E",
        "text": "#E6EDF3",
        "portrait": "#FF4D3D",
        "chrome": "#F59E0B",
        "accent": "#3FB950",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#F0F3F6",
        "line": "#D0D7DE",
        "muted": "#57606A",
        "text": "#1F2328",
        "portrait": "#C7301F",
        "chrome": "#B45309",
        "accent": "#1A7F37",
        "shadow": "#AAB7C4",
    },
}

LOGO_ORDER = ("php", "laravel", "filament")


def fit_silhouette(image: Image.Image, size: int) -> Image.Image:
    """Recorta no conteudo e recentra a silhueta num quadrado de `size`."""
    box = image.getchannel("A").getbbox()
    if box is None:
        return image
    cropped = image.crop(box)
    scale = min(size * 0.92 / cropped.width, size * 0.92 / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
        resized,
    )
    return canvas


def rasterize_svg(source: Path, size: int) -> Image.Image:
    """Rasteriza um SVG de marca em uma silhueta preta de `size` x `size`.

    Usa svgelements (Python puro) em vez do cairosvg, que exigiria a libcairo2
    instalada no sistema e nao existe pronta no Windows. Cada subpath vira um
    poligono e entra por XOR, ou seja, regra par-impar: e assim que os furos
    dos icones do simple-icons (o miolo do "e", a contra-forma do gear) se
    abrem sem precisar interpretar a direcao de cada contorno.
    """
    from svgelements import SVG, Path as SvgPath, Shape

    svg = SVG.parse(str(source), width=size, height=size)
    filled = np.zeros((size, size), dtype=bool)

    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        path = SvgPath(element)
        path.reify()
        for subpath in path.as_subpaths():
            sub = SvgPath(subpath)
            length = sub.length(error=1e-3)
            if length <= 0:
                continue
            # Um ponto a cada ~1,2 px mantem as curvas lisas nesta escala.
            steps = max(24, min(2000, int(length / 1.2)))
            points = []
            for i in range(steps + 1):
                p = sub.point(i / steps)
                points.append((float(p.x), float(p.y)))
            if len(points) < 3:
                continue
            layer = Image.new("1", (size, size), 0)
            ImageDraw.Draw(layer).polygon(points, fill=1)
            filled ^= np.asarray(layer, dtype=bool)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 3] = filled * 255
    return Image.fromarray(rgba, "RGBA")


def load_logos() -> dict[str, Image.Image]:
    """Rasteriza os SVGs oficiais (simple-icons) em silhuetas de 400px."""
    size = 400
    logos: dict[str, Image.Image] = {}
    for name in LOGO_ORDER:
        src = LOGOS / f"{name}.svg"
        if not src.exists():
            raise SystemExit(f"Logo ausente: {src}")
        image = rasterize_svg(src, size)
        # PHP e Filament sao marcas largas e baixas; normalizar a bounding box
        # evita que uma delas apareca minuscula ao lado das outras.
        logos[name] = fit_silhouette(image, size)
    return logos


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Difusao Floyd-Steinberg serpentina de 1 bit; True = pixel aceso."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def vignette(width: int, height: int, params: tuple) -> np.ndarray:
    """Mascara eliptica suave: 1 sobre o rosto, 0 nas bordas.

    O avatar e um JPEG sem canal alpha e com fundo noturno movimentado. Sem
    esta mascara o fundo vira ruido pontilhado e engole o rosto.
    """
    fw, fh, center_y, power, softness = params
    ys, xs = np.mgrid[0:height, 0:width]
    nx = (xs - width / 2) / (width * fw)
    ny = (ys - height * center_y) / (height * fh)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip((1.05 - radius) / softness, 0.0, 1.0) ** power


def coldness(image: Image.Image) -> np.ndarray:
    """1 onde o pixel e frio e saturado, 0 na pele e no cabelo.

    A foto foi tirada a noite em frente aos Arcos da Lapa: os arcos estao
    iluminados de verde e ha luzes de predio logo atras da cabeca. Pele e
    cabelo ficam entre 0 e 50 graus de matiz; esse fundo, entre 70 e 300.
    """
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1] / 255.0
    cold_hue = np.clip((hue - 60.0) / 25.0, 0.0, 1.0) * np.clip(
        (320.0 - hue) / 25.0, 0.0, 1.0
    )
    return cold_hue * np.clip((sat - 0.12) / 0.18, 0.0, 1.0)


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Coordenadas x/y do banner amostradas numa grade de dither 300x340."""
    source = Image.open(SOURCE).convert("RGB")
    crop = source.crop(CROP).resize((300, 340), Image.Resampling.LANCZOS)

    # Os dois temas selecionam pixels opostos, entao pedem tratamentos opostos
    # do fundo. No escuro o ponto marca o que e claro, e as luzes dos arcos
    # viram riscos colados na orelha: preciso de vinheta apertada e da
    # supressao por matiz. No claro o ponto marca o que e escuro, o fundo
    # iluminado ja some sozinho, e suprimi-lo so criaria manchas geometricas
    # onde antes havia degrade.
    if theme == "dark":
        mask = vignette(300, 340, (0.42, 0.52, 0.45, 1.8, 0.40))
        mask = mask * (1.0 - 0.85 * coldness(crop))
        select_lit = True
    else:
        mask = vignette(300, 340, (0.50, 0.54, 0.46, 1.5, 0.45))
        select_lit = False

    lum = np.asarray(ImageOps.grayscale(crop), dtype=np.float32)
    if select_lit:
        # Fundo puxado para o preto: nenhum ponto aceso sobra fora do rosto.
        merged = lum * mask
    else:
        # Fundo puxado para o branco: nenhum ponto escuro sobra fora do rosto.
        merged = lum * mask + 255.0 * (1.0 - mask)

    prepared = Image.fromarray(np.uint8(np.clip(merged, 0, 255)), "L")
    # Equaliza so contra o sujeito, para que pele iluminada e barba escura nao
    # esmaguem os meios-tons; depois reforca o contraste local dos tracos.
    subject = Image.fromarray(np.uint8((mask > 0.25) * 255), "L")
    if theme == "dark":
        prepared = ImageOps.equalize(prepared, mask=subject)
    else:
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(
        ImageFilter.UnsharpMask(radius=2, percent=175, threshold=1)
    )

    bits = floyd_steinberg(np.asarray(prepared))
    active = bits if select_lit else ~bits
    active &= mask > 0.08

    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Centrado no recorte do VISUAL.MAP (x 49..439, y 124..538).
    points = np.column_stack((94 + xs, 161 + ys)).astype(np.float32)
    if len(points) > 18000:
        points = points[rng.choice(len(points), 18000, replace=False)]
    return points


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Amostra uma silhueta no espaco de coordenadas do quadro do retrato."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    # O logo ocupa um quadrado centrado de 270x270 dentro do VISUAL.MAP.
    return np.column_stack(
        (109 + xs[chosen] * 0.675, 196 + ys[chosen] * 0.675)
    ).astype(np.float32)


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ordena os pontos de destino por atribuicao de custo minimo."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    """Uma casa decimal, o suficiente para coordenada em pixel."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def frac(value: float) -> str:
    """Fracao de 0 a 1 com quatro casas, para keyTimes.

    Com a precisao de `num()` os limites de fase viravam 0.2/0.3/0.4/... e a
    cadencia saia irreconhecivel: o retrato ficava 2,84 s no ar em vez de 3,0,
    o PHP 1,42 s em vez de 2,0 e o Laravel 2,84 s em vez de 2,0. keyTimes sao
    normalizados, entao uma casa decimal e grossa demais.
    """
    return f"{value:.4f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Agrega pontos horizontais adjacentes em trechos compactos de path."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    """Largura monoespacada estavel, usada no textLength e nas linhas guia."""
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_points: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    php = transport(source, logo_points["php"][:n])
    laravel = transport(php, logo_points["laravel"][:n])
    filament = transport(laravel, logo_points["filament"][:n])

    # Limites de fase explicitos e irregulares: 3.0 de retrato, 2.0 por logo
    # e quatro transicoes de 1.3 = 14.2 segundos.
    times = [0, 3.0, 4.3, 6.3, 7.6, 9.6, 10.9, 12.9, 14.2]
    key_times = ";".join(frac(v / LOOP_SECONDS) for v in times)
    # Devolver cada viajante a coordenada exata de partida mantem o fim do loop
    # invisivel. Todo morph entre logos usa transporte otimo.
    frames = [source, source, php, php, laravel, laravel, filament, filament, source]
    opacity_values = "0;0;1;1;1;1;1;1;0"

    mono = "ui-monospace,SFMono-Regular,Consolas,monospace"

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Perfil ao vivo do Guilherme Ferro</title>',
        '<desc id="desc">Terminal animado com um retrato pontilhado que se '
        "transforma nas silhuetas do PHP, do Laravel e do Filament.</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/>'
        f'<feFlood flood-color="{t["chrome"]}" flood-opacity=".35"/>'
        '<feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip">'
        '<rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        f'font-family="{mono}" font-size="13" '
        'letter-spacing=".4">profile.sh --live</text>',
        # Quadro visual da esquerda.
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="{mono}" '
        'font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        f'font-family="{mono}" font-size="11">300x340 / 1-BIT</text>',
        '<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        # A camada do loop ja aparece em t=0, para que o primeiro quadro
        # estatico (thumbnail, preview) tambem mostre o rosto. A copia da
        # introducao cintila por cima e passa o bastao aos 3.2s.
        '<g opacity="1">',
    ]

    # Deriva densa do retrato: 94 faixas ruidosas rumo ao centroide do PHP.
    php_centroid = php.mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 4, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (php_centroid - centroid) * 0.18 + noise[band]
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{key_times}" '
            f'values="0 0;0 0;{num(delta[0])} {num(delta[1])};'
            f'{num(delta[0])} {num(delta[1])};0 0;0 0;0 0;0 0;0 0"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" keyTimes="{key_times}" '
            'values=".94;.94;0;0;0;0;0;0;.94"/></path>'
        )

    # Viajantes por transporte otimo, como quadradinhos de path (nunca glifos).
    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'begin="{INTRO_SECONDS}s" dur="{LOOP_SECONDS}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" '
            f'dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{opacity_values}"/></path>'
        )
    parts.append("</g>")

    # Introducao unica e espalhada: sessenta grupos aleatorios intercalados.
    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" '
            'values="1;0" fill="freeze"/>'
            "</path>"
        )
    parts.extend(
        [
            "</g>",
            # Telemetria discreta do quadro.
            f'<text x="58" y="551" fill="{t["muted"]}" font-family="{mono}" '
            f'font-size="10">PTS {len(portrait):05d} / FS-SERPENTINE</text>',
            # Painel de informacoes da direita.
            f'<rect x="474" y="88" width="672" height="472" rx="6" '
            f'fill="{t["panel2"]}" stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" font-family="{mono}" '
            'font-size="13" font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
            # Selo LIVE e pilula do usuario.
            '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A">'
            '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" '
            'repeatCount="indefinite"/></circle></g>',
            f'<text x="927" y="111" fill="#FF4D5A" font-family="{mono}" '
            'font-size="12" font-weight="700">LIVE</text>',
            f'<rect x="982" y="94" width="146" height="24" rx="12" '
            f'fill="{t["chrome"]}" opacity=".16" stroke="{t["chrome"]}"/>',
            f'<text x="1055" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            f'font-family="{mono}" font-size="14" font-weight="700">@gsferro</text>',
        ]
    )

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend(
            [
                f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" '
                f'font-family="{mono}" font-size="14">{html.escape(label)}</text>',
                f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
                f'fill="none" stroke="{t["line"]}" stroke-width="1" '
                'shape-rendering="crispEdges"/>',
                f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" '
                f'fill="{t["text"]}" font-family="{mono}" font-size="14" '
                f'textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(value)}</text>",
            ]
        )
        row_y += 23

    parts.extend(
        [
            f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
            f'<text x="491" y="548" fill="{t["accent"]}" font-family="{mono}" '
            'font-size="11">ALL SYSTEMS NOMINAL</text>',
            f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" '
            f'font-family="{mono}" font-size="11">UTC-3 / BRAZIL NODE</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Retrato de origem ausente: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logos = load_logos()

    # Guarda os pontos de dither por tema como dado de origem reproduzivel.
    portraits: dict[str, np.ndarray] = {}
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        portraits[theme] = points
        np.save(DATA / f"portrait-{theme}.npy", points)

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in logos.items()
        }
        for name, points in sampled.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render_svg(theme, portraits[theme], sampled, rng)
        output = ASSETS / f"banner-{theme}.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), "
            f"{len(portraits[theme]):,} pontos de retrato, "
            f"{TRAVELLER_COUNT} viajantes"
        )


if __name__ == "__main__":
    main()
