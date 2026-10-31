"""Draw today's piece, seeded by the UTC date.

A given day always draws the same thing.  Stdlib only.
Usage: python3 scripts/draw.py [YYYY-MM-DD] [out.svg]

Each piece is a ground and one to three lenses.  The ground is one drawing
system over one noise field; each lens cuts a hole in it and shows the other
system over a finer field, like a loupe laid on a map.  The systems:

  streamlines  evenly spaced lines traced along the flow
  contours     iso-lines of the field, read as a survey map

The whole thing draws itself in when the image loads.  Each stroke carries
its own measured length (pathLength="1" is not reliably honoured in an
<img>), and a renderer that never runs the animation shows the finished
drawing rather than a blank card: see the timing note in draw().
"""

import datetime as dt
import hashlib
import math
import os
import random
import sys

W, H = 1000, 520
MARGIN = 36
CAPTION = 28  # band at the bottom for the caption
EPOCH = dt.date(2020, 1, 1)  # No. 0001
DRAW_TIME = 3.4  # seconds until the last stroke starts

PALETTES = [
    # background, ink, accent
    ("#f3efe6", "#1f1d1a", "#e4572e"),
    ("#0e1116", "#d7dde6", "#f2c14e"),
    ("#0b1d2a", "#9fc3d6", "#ff6b4a"),
    ("#e9ecdf", "#2f3b2a", "#c2410c"),
    ("#ece6da", "#3a3530", "#2563eb"),
    ("#111111", "#e8e2d6", "#ff3b30"),
    ("#2a1d17", "#e8cdb5", "#7dd3fc"),
]

BOX = (MARGIN, MARGIN, W - MARGIN, H - MARGIN - CAPTION)


class Noise:
    """Improved Perlin noise, 2D."""

    def __init__(self, rng):
        p = list(range(256))
        rng.shuffle(p)
        self.p = p + p

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _grad(h, x, y):
        h &= 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

    def __call__(self, x, y):
        xi, yi = int(math.floor(x)) & 255, int(math.floor(y)) & 255
        xf, yf = x - math.floor(x), y - math.floor(y)
        u, v = self._fade(xf), self._fade(yf)
        p = self.p
        aa, ab = p[p[xi] + yi], p[p[xi] + yi + 1]
        ba, bb = p[p[xi + 1] + yi], p[p[xi + 1] + yi + 1]
        x1 = self._grad(aa, xf, yf) + u * (self._grad(ba, xf - 1, yf) - self._grad(aa, xf, yf))
        x2 = self._grad(ab, xf, yf - 1) + u * (self._grad(bb, xf - 1, yf - 1) - self._grad(ab, xf, yf - 1))
        return x1 + v * (x2 - x1)

    def fbm(self, x, y):
        """Three octaves.  The third is deliberately faint; turn it up and
        the streamlines stop tracing and break into stubble."""
        return (self(x, y)
                + 0.50 * self(x * 2.03 + 17, y * 2.03 + 31)
                + 0.12 * self(x * 4.11 + 53, y * 4.11 + 71))


def make_field(rng, fine=1.0):
    """A scalar field (for contours) and the angle field that follows its
    flow (for streamlines).  `fine` scales the detail, for the lenses."""
    noise = Noise(rng)
    kind = rng.choice(["drift", "vortex", "tide", "shear"])
    scale = rng.uniform(0.0021, 0.0040) * fine
    turn = rng.uniform(1.5, 2.6)

    # away from a vortex the eddies stay weak; strong ones curl the lines
    # tight enough that they terminate almost immediately.
    n_eddies = rng.randint(1, 3) if kind == "vortex" else rng.randint(0, 2)
    eddies = [(rng.uniform(0.15, 0.85) * W, rng.uniform(0.18, 0.82) * H,
               rng.choice([-1, 1]),
               rng.uniform(6000, 11000) if kind == "vortex" else rng.uniform(1800, 4200))
              for _ in range(n_eddies)]
    bands = rng.uniform(1.2, 2.8)
    band_amp = rng.uniform(0.18, 0.45) if kind == "shear" else 0.0
    base = rng.uniform(-0.25, 0.25)

    def swirl(x, y):
        vx = vy = 0.0
        for ex, ey, spin, strength in eddies:
            dx, dy = x - ex, y - ey
            r2 = dx * dx + dy * dy + 900
            r = math.sqrt(r2)
            w = strength / r2
            vx += -dy * spin * w / r
            vy += dx * spin * w / r
        return vx, vy

    def height(x, y):
        h = noise.fbm(x * scale, y * scale)
        if band_amp:
            h += math.sin(y / H * bands * math.tau) * band_amp
        for ex, ey, spin, strength in eddies:
            h += spin * strength / 9000 * math.exp(-((x - ex) ** 2 + (y - ey) ** 2) / 40000)
        return h

    def angle(x, y):
        n = noise.fbm(x * scale, y * scale)
        vx, vy = swirl(x, y)
        if kind == "drift":
            a = n * turn * math.pi
        elif kind == "vortex":
            a = n * turn * math.pi * 0.6
            return math.atan2(vy + 0.45 * math.sin(a), vx + 0.45 * math.cos(a))
        elif kind == "tide":
            a = base + noise.fbm(x * scale, y * scale * 1.8) * 1.05
        else:  # shear: stacked bands that pull against each other
            a = n * turn * math.pi * 0.7 + math.sin(y / H * bands * math.tau) * band_amp * math.pi
        return math.atan2(math.sin(a) + vy, math.cos(a) + vx)

    return angle, height


def streamlines(angle, rng, dsep, box, allowed):
    """Jobard-Lefer style: no two lines closer than dsep, stop at dtest."""
    x0, y0, x1, y1 = box
    dtest = dsep * 0.55
    step = min(2.6, dsep * 0.45)
    cell = dsep
    grid = {}

    def near(x, y, d, own=None):
        cx, cy = int(x // cell), int(y // cell)
        d2 = d * d
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for px, py, lid in grid.get((gx, gy), ()):
                    if lid != own and (px - x) ** 2 + (py - y) ** 2 < d2:
                        return True
        return False

    def ok(x, y):
        return x0 < x < x1 and y0 < y < y1 and allowed(x, y)

    def trace(sx, sy, sign, lid):
        pts = []
        x, y = sx, sy
        for _ in range(1400):
            a = angle(x, y)
            # midpoint step, keeps curves smooth
            mx, my = x + math.cos(a) * step * 0.5 * sign, y + math.sin(a) * step * 0.5 * sign
            a = angle(mx, my)
            x, y = x + math.cos(a) * step * sign, y + math.sin(a) * step * sign
            if not ok(x, y) or near(x, y, dtest, lid):
                break
            # avoid tight self loops
            if len(pts) > 12 and any((px - x) ** 2 + (py - y) ** 2 < dtest * dtest
                                     for px, py in pts[:-10]):
                break
            pts.append((x, y))
        return pts

    area = (x1 - x0) * (y1 - y0)
    seeds = [(rng.uniform(x0, x1), rng.uniform(y0, y1)) for _ in range(int(area / 90))]
    lines, queue = [], []
    while seeds or queue:
        sx, sy = queue.pop() if queue else seeds.pop()
        if not ok(sx, sy) or near(sx, sy, dsep):
            continue
        lid = len(lines)
        pts = trace(sx, sy, -1, lid)[::-1] + [(sx, sy)] + trace(sx, sy, 1, lid)
        if len(pts) < 8:
            continue
        for px, py in pts:
            grid.setdefault((int(px // cell), int(py // cell)), []).append((px, py, lid))
        lines.append(pts)
        # new seeds either side of this line, so the field fills in waves
        for i in range(0, len(pts) - 1, 5):
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            dx, dy = bx - ax, by - ay
            n = math.hypot(dx, dy) or 1
            queue.append((ax - dy / n * dsep, ay + dx / n * dsep))
            queue.append((ax + dy / n * dsep, ay - dx / n * dsep))
    return lines


# marching squares: case -> the pairs of cell edges the iso-line crosses.
# edges are 0 top, 1 right, 2 bottom, 3 left.
_MS = {
    0: (), 1: ((2, 3),), 2: ((1, 2),), 3: ((1, 3),), 4: ((0, 1),),
    5: ((0, 3), (1, 2)), 6: ((0, 2),), 7: ((0, 3),), 8: ((0, 3),),
    9: ((0, 2),), 10: ((0, 1), (2, 3)), 11: ((0, 1),), 12: ((1, 3),),
    13: ((1, 2),), 14: ((2, 3),), 15: (),
}


def contours(height, n_levels, box, allowed, cell=4.8):
    """Iso-lines of the scalar field, chained into polylines, and cut
    wherever they leave the allowed region."""
    x0, y0, x1, y1 = box
    nx, ny = max(8, int((x1 - x0) / cell)), max(8, int((y1 - y0) / cell))
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    val = [[height(x0 + i * dx, y0 + j * dy) for j in range(ny + 1)] for i in range(nx + 1)]
    lo = min(min(col) for col in val)
    hi = max(max(col) for col in val)
    if hi - lo < 1e-6:
        return []
    levels = [lo + (hi - lo) * (k + 1) / (n_levels + 1) for k in range(n_levels)]

    lines = []
    for level in levels:
        segs = []
        for i in range(nx):
            for j in range(ny):
                a, b = val[i][j], val[i + 1][j]
                c, d = val[i + 1][j + 1], val[i][j + 1]
                pairs = _MS[((a > level) << 3) | ((b > level) << 2) | ((c > level) << 1) | (d > level)]
                if not pairs:
                    continue

                def point(edge):
                    if edge == 0:
                        return (x0 + (i + (level - a) / (b - a)) * dx, y0 + j * dy)
                    if edge == 1:
                        return (x0 + (i + 1) * dx, y0 + (j + (level - b) / (c - b)) * dy)
                    if edge == 2:
                        return (x0 + (i + (level - d) / (c - d)) * dx, y0 + (j + 1) * dy)
                    return (x0 + i * dx, y0 + (j + (level - a) / (d - a)) * dy)

                for e1, e2 in pairs:
                    segs.append((point(e1), point(e2)))

        # chain segments end to end
        key = lambda p: (round(p[0], 2), round(p[1], 2))
        ends = {}
        for s in segs:
            ends.setdefault(key(s[0]), []).append(s)
            ends.setdefault(key(s[1]), []).append(s)
        used = set()
        for seg in segs:
            if id(seg) in used:
                continue
            used.add(id(seg))
            chain = [seg[0], seg[1]]
            for at_end in (True, False):
                while True:
                    tip = key(chain[-1] if at_end else chain[0])
                    nxt = next((s for s in ends.get(tip, ()) if id(s) not in used), None)
                    if nxt is None:
                        break
                    used.add(id(nxt))
                    far = nxt[1] if key(nxt[0]) == tip else nxt[0]
                    if at_end:
                        chain.append(far)
                    else:
                        chain.insert(0, far)
            # cut the chain where it leaves the region
            run = []
            for p in chain + [None]:
                if p is not None and allowed(*p):
                    run.append(p)
                    continue
                if len(run) >= 6:
                    lines.append(run)
                run = []
    return lines


def place_lenses(rng):
    """One to three circles that do not overlap each other or the edge."""
    x0, y0, x1, y1 = BOX
    lenses = []
    want = rng.choice([1, 1, 2, 2, 3])
    for _ in range(400):
        if len(lenses) == want:
            break
        r = rng.uniform(58, 150) if not lenses else rng.uniform(40, 110)
        cx, cy = rng.uniform(x0 + r + 12, x1 - r - 12), rng.uniform(y0 + r + 12, y1 - r - 12)
        if all(math.hypot(cx - ox, cy - oy) > r + orr + 24 for ox, oy, orr in lenses):
            lenses.append((cx, cy, r))
    return lenses


def system(name, field, rng, box, allowed, density):
    angle, height = field
    if name == "contours":
        return contours(height, int(rng.randint(22, 40) * density), box, allowed)
    return streamlines(angle, rng, rng.uniform(4.6, 6.4) / density, box, allowed)


def simplify(pts, eps=0.3):
    if len(pts) < 3:
        return pts
    (ax, ay), (bx, by) = pts[0], pts[-1]
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy) or 1e-9
    worst, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = abs(dy * px - dx * py + bx * ay - by * ax) / n
        if d > worst:
            worst, idx = d, i
    if worst < eps:
        return [pts[0], pts[-1]]
    return simplify(pts[: idx + 1], eps)[:-1] + simplify(pts[idx:], eps)


def _num(v):
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def path_d(pts):
    """Path data plus its drawn length, which the animation runs on."""
    pts = simplify(pts)
    x, y = pts[0]
    out = [f"M{x:.1f} {y:.1f}"]
    px, py = round(x, 1), round(y, 1)
    total = 0.0
    for x, y in pts[1:]:
        x, y = round(x, 1), round(y, 1)
        total += math.hypot(x - px, y - py)
        out.append(f"l{_num(x - px)} {_num(y - py)}")
        px, py = x, y
    # round up so the dash always covers the whole stroke
    return "".join(out), math.ceil(total) + 1


def draw(day):
    digest = hashlib.sha256(day.isoformat().encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    edition = (day - EPOCH).days + 1
    bg, ink, accent = rng.choice(PALETTES)

    ground_sys = rng.choice(["streamlines", "contours"])
    lens_sys = "contours" if ground_sys == "streamlines" else "streamlines"
    lenses = place_lenses(rng)

    def outside(x, y):
        return all(math.hypot(x - cx, y - cy) > r + 5 for cx, cy, r in lenses)

    # (points, delay, stroke attributes)
    strokes = []
    fx, fy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H
    far = math.hypot(W, H)

    ground = system(ground_sys, make_field(rng), rng, BOX, outside, 1.0)
    ranked = sorted(range(len(ground)), key=lambda i: -len(ground[i]))
    marked = set(ranked[: rng.randint(1, 2)])
    for i, pts in enumerate(ground):
        mx, my = pts[len(pts) // 2]
        t = DRAW_TIME * 0.75 * math.hypot(mx - fx, my - fy) / far * 1.6 + rng.uniform(0, 0.25)
        if i in marked:
            strokes.append((pts, DRAW_TIME, f'stroke="{accent}" stroke-width="2.4"'))
        else:
            strokes.append((pts, min(t, DRAW_TIME * 0.8),
                            f'stroke-opacity="{rng.uniform(0.4, 0.95):.2f}" '
                            f'stroke-width="{rng.uniform(0.55, 1.4):.2f}"'))

    rings = []
    for k, (cx, cy, r) in enumerate(lenses):
        box = (cx - r, cy - r, cx + r, cy + r)
        inside = lambda x, y, cx=cx, cy=cy, r=r: math.hypot(x - cx, y - cy) < r - 3
        start = DRAW_TIME * 0.35 + k * 0.3
        rings.append((cx, cy, r, start))
        for pts in system(lens_sys, make_field(rng, fine=2.2), rng, box, inside, 1.35):
            mx, my = pts[len(pts) // 2]
            t = start + 0.4 + 1.3 * math.hypot(mx - cx, my - cy) / r
            strokes.append((pts, t, f'stroke-opacity="{rng.uniform(0.5, 0.95):.2f}" '
                                    f'stroke-width="{rng.uniform(0.5, 1.1):.2f}"'))

    # Timing.  Every stroke shares one timeline that starts 1ms in, with no
    # fill mode, and holds itself hidden inside its keyframes until its turn.
    # A browser that never advances an <img>'s clock sits in that first
    # millisecond, before the animation begins, and so shows the authored,
    # finished drawing instead of a blank card.  Strokes are binned into
    # waves that share keyframes, which keeps the stylesheet small.
    total = DRAW_TIME + 1.8
    waves = {}

    def wave(start, dur):
        key = (round(start * 10), round(dur * 5))
        if key not in waves:
            p0 = min(99.0, key[0] / 10 / total * 100)
            p1 = min(100.0, p0 + key[1] / 5 / total * 100)
            waves[key] = (f"w{len(waves)}", p0, p1)
        return waves[key][0]

    body = []
    for pts, t, attrs in strokes:
        d, length = path_d(pts)
        dur = 0.6 + min(1.0, length / 700)
        body.append(f'<path class="{wave(t, dur)}" d="{d}" {attrs} '
                    f'stroke-dasharray="{length}" style="--l:{length}"/>')
    for cx, cy, r, start in rings:
        circ = math.ceil(math.tau * r) + 1
        body.append(f'<circle class="{wave(start, 1.2)}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                    f'stroke-width="1.3" stroke-dasharray="{circ}" style="--l:{circ}"/>')

    ease = "cubic-bezier(.45,.05,.25,1)"
    css = "".join(
        f".{name}{{animation-name:{name}}}"
        f"@keyframes {name}{{0%,{p0:.2f}%{{stroke-dashoffset:var(--l);animation-timing-function:{ease}}}"
        f"{p1:.2f}%,100%{{stroke-dashoffset:0}}}}"
        for name, p0, p1 in waves.values())

    cap_y = H - CAPTION / 2 - 4
    font = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    cap = f'font-family="{font}" font-size="12" fill="{ink}" letter-spacing=".08em"'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="No. {edition:04d}, a generative drawing for {day.isoformat()}.">
<style>
path,circle{{animation-duration:{total:.1f}s;animation-delay:.001s;animation-timing-function:linear}}
{css}
@media (prefers-reduced-motion:reduce){{path,circle{{animation:none}}}}
</style>
<rect width="{W}" height="{H}" rx="14" fill="{bg}"/>
<g fill="none" stroke="{ink}" stroke-linecap="round">{"".join(body)}</g>
<text x="{MARGIN}" y="{cap_y}" {cap}>No. {edition:04d}</text>
<text x="{W - MARGIN}" y="{cap_y}" text-anchor="end" opacity=".55" {cap}>{day:%Y.%m.%d}</text>
</svg>
"""


if __name__ == "__main__":
    day = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime.now(dt.timezone.utc).date()
    out = sys.argv[2] if len(sys.argv) > 2 else "art/today.svg"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(draw(day))
