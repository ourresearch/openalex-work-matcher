"""The README's two benchmark charts (single theme-neutral SVGs). Numbers: `python3 benchmarks/score.py nights` (8 judged nights,
the title + author rule on the same records with the same labels). Run from the repository root."""
import os

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
INK, AXIS = "#707a84", "#8b949e"
OLD, NEW = "#888780", "#2a78d6"
CHARTS = [
    ("precision", "Right when it merges", "Share of merges that join the right work",
     [("Title + author rule", 78.6, OLD), ("New matcher", 99.1, NEW)]),
    ("found", "Duplicates it finds", "Share of records with a match that it merges",
     [("Title + author rule", 74.3, OLD), ("New matcher", 92.2, NEW)]),
]


def bar(x, top, w, base, r=4):
    return (f"M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + w - r:.1f} "
            f"Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{base:.1f} Z")


def svg(title, sub, bars):
    W, H, left, right, base, top = 440, 330, 44, 432, 296, 84
    y = lambda v: base - (base - top) * v / 100
    alt = f"{title}, 8 judged nights of incoming records: " + ", ".join(f"{n} {v:.1f}%" for n, v, _ in bars)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f"font-family='{FONT}' style=\"font-variant-numeric: tabular-nums\">", f"<title>{alt}</title>",
           f'<text x="0" y="16" fill="{INK}" font-size="18" font-weight="650" dominant-baseline="middle">{title}</text>',
           f'<text x="0" y="40" fill="{INK}" font-size="14" dominant-baseline="middle">{sub}</text>']
    for v in (0, 25, 50, 75, 100):
        op = 0.8 if v == 0 else 0.35
        out.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{right}" y2="{y(v):.1f}" stroke="{AXIS}" stroke-width="1" stroke-opacity="{op}"/>')
        out.append(f'<text x="36" y="{y(v):.1f}" fill="{INK}" font-size="12" text-anchor="end" dominant-baseline="middle">{v}%</text>')
    slot = (right - left) / len(bars)
    for i, (name, v, color) in enumerate(bars):
        x = left + slot * i + (slot - 72) / 2
        new = color == NEW
        out.append(f'<path d="{bar(x, y(v), 72, base)}" fill="{color}"/>')
        out.append(f'<text x="{x + 36:.1f}" y="{y(v) - 14:.1f}" fill="{INK}" font-size="{20 if new else 17}" font-weight="{700 if new else 500}" '
                   f'text-anchor="middle" dominant-baseline="middle">{v:.1f}%</text>')
        out.append(f'<text x="{x + 36:.1f}" y="314" fill="{INK}" font-size="14" font-weight="{600 if new else 400}" '
                   f'text-anchor="middle" dominant-baseline="middle">{name}</text>')
    return "\n".join(out + ["</svg>"]) + "\n"


os.makedirs("docs/img", exist_ok=True)
for name, title, sub, bars in CHARTS:
    open(f"docs/img/{name}.svg", "w").write(svg(title, sub, bars))
    print(f"docs/img/{name}.svg")
