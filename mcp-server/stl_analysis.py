# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 CNC Kitchen / ScanRuler Contributors (https://github.com/CNCKitchen/ScanRuler)
# Copyright (C) 2026 CadQuery Contrib / MCP Server Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
CAD-Exported STL -> BRep Analyzer and 3-Axis Ray-Cast Voxel Mesh Comparator.

Adapted from CNCKitchen/ScanRuler:
  - plugins/reverse/skills/cad-stl-reverse-engineering/scripts/cadlib.py
  - plugins/reverse/skills/cad-stl-reverse-engineering/scripts/analyze.py
  - plugins/reverse/skills/cad-stl-reverse-engineering/scripts/compare.py
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TOL = 3e-4          # Vertex-on-surface tolerance (mm)
STRICT = 2e-5       # Analytic face interior vertex precision
LOOSE = 6e-3        # Loose boundary edge curve tolerance
SMOOTH_DEG = 32.0   # Max dihedral angle between facets of one smooth curved face
AXES = "xyz"


# ---------------------------------------------------------------------------
# Mesh I/O & Topology Helpers
# ---------------------------------------------------------------------------

def load_stl(path: str | Path):
    """Binary or ASCII STL -> welded vertices V (n,3 float64), triangles F (m,3 int)."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0] if len(b) >= 84 else -1
    if len(b) == 84 + 50 * n:
        a = np.frombuffer(
            b[84:84 + n * 50],
            dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        )
        tri = a["v"].astype(np.float64)
    else:
        txt = b.decode("ascii", "replace").split()
        xyz = [
            (float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3]))
            for i in range(len(txt))
            if txt[i] == "vertex"
        ]
        if not xyz:
            raise ValueError(f"No STL triangles found in '{path}'.")
        tri = np.array(xyz, dtype=np.float64).reshape(-1, 3, 3)
    flat = tri.reshape(-1, 3)
    u, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    return u, F[ok]


def save_stl(path: str | Path, V: np.ndarray, F: np.ndarray):
    P = V[F].astype(np.float32)
    N = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]).astype(np.float64)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = N
    rec["v"] = P
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())


def tri_normals(V: np.ndarray, F: np.ndarray):
    P = V[F]
    N = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    A = np.linalg.norm(N, axis=1)
    return N / np.where(A == 0, 1, A)[:, None], A / 2


def edges_of(F: np.ndarray, nV: int):
    """Manifold interior edges: (t1, t2, (v0, v1)) arrays, plus the count of open edges."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    T = np.tile(np.arange(len(F)), 3)
    key = E[:, 0].astype(np.int64) * nV + E[:, 1]
    o = np.argsort(key, kind="stable")
    key, T, E = key[o], T[o], E[o]
    _, start, cnt = np.unique(key, return_index=True, return_counts=True)
    s = start[cnt == 2]
    return T[s], T[s + 1], E[s], int((cnt == 1).sum())


def pieces_of(F: np.ndarray, nV: int):
    t1, t2, _, _ = edges_of(F, nV)
    g = coo_matrix((np.ones(len(t1)), (t1, t2)), shape=(len(F), len(F)))
    return connected_components(g, directed=False)[1]


def volume(V: np.ndarray, F: np.ndarray) -> float:
    P = V[F]
    return float(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6)


# ---------------------------------------------------------------------------
# Frames & Directions
# ---------------------------------------------------------------------------

def frame(ax):
    """Two unit vectors square to ax and to each other (e1 along the world axis nearest square)."""
    ax = np.asarray(ax, float)
    ref = np.eye(3)[int(np.argmin(np.abs(ax)))]
    e1 = ref - (ref @ ax) * ax
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(ax, e1)


def snap_dir(d, tol=1e-5):
    """A direction within tol (rad) of a world axis is that axis; otherwise unit, largest component positive."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    i = int(np.argmax(np.abs(d)))
    if d[i] < 0:
        d = -d
    if np.linalg.norm(d - np.eye(3)[i]) < tol:
        return np.eye(3)[i]
    return d


def world_axis(d, tol=1e-5):
    """'x' / 'y' / 'z' when d is (anti)parallel to a world axis within tol rad, else None."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    i = int(np.argmax(np.abs(d)))
    return AXES[i] if np.linalg.norm(np.abs(d) - np.eye(3)[i]) < tol else None


# ---------------------------------------------------------------------------
# Primitive Surface Fitting
# ---------------------------------------------------------------------------

def fit_plane(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    return n, float(n @ c), float(np.abs((P - c) @ n).max())


def circle2d(q):
    A = np.c_[2 * q, np.ones(len(q))]
    s, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    c = s[:2]
    r2 = s[2] + c @ c
    r = np.sqrt(r2) if r2 > 0 else np.inf
    return c, r, (np.abs(np.linalg.norm(q - c, axis=1) - r).max() if np.isfinite(r) else np.inf)


def _ang(d):
    d = d / np.linalg.norm(d)
    return np.array([np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0])])


def _dir(t):
    return np.array([np.sin(t[0]) * np.cos(t[1]), np.sin(t[0]) * np.sin(t[1]), np.cos(t[0])])


def fit_cylinder(P, N=None, ax0=None):
    """Axis direction, a point on the axis, radius, max residual."""
    if ax0 is None:
        M = N.T @ N
        ax0 = np.linalg.eigh(M)[1][:, 0]
    e1, e2 = frame(ax0)
    c2, r, res0 = circle2d(np.c_[P @ e1, P @ e2])
    if not np.isfinite(r):
        return None
    c0 = c2[0] * e1 + c2[1] * e2
    if res0 > 0.02 * max(r, 1):
        return ax0, c0, float(r), float(res0)

    def res(x):
        a = _dir(x[:2])
        c = x[2:5]
        d = P - c
        d -= np.outer(d @ a, a)
        return np.linalg.norm(d, axis=1) - x[5]

    try:
        s = least_squares(res, np.r_[_ang(ax0), c0, r], method="lm", xtol=1e-15, ftol=1e-15, max_nfev=150)
    except Exception:
        return None
    a = snap_dir(_dir(s.x[:2]))
    c = s.x[2:5]
    if world_axis(a):
        e1, e2 = frame(a)
        c2, r, res_ = circle2d(np.c_[P @ e1, P @ e2])
        c = c2[0] * e1 + c2[1] * e2 + (c @ a) * a
        return a, c, float(r), float(res_)
    c = c - (c @ a) * a
    return a, c, float(abs(s.x[5])), float(np.abs(res(s.x)).max())


def fit_cone(P, N, C):
    """Axis, apex, half-angle (rad), max residual."""
    A = np.c_[N, -np.ones(len(N))]
    w = np.linalg.svd(A, full_matrices=False)[2][-1]
    a = w[:3]
    nrm = np.linalg.norm(a)
    if nrm < 1e-9:
        return None
    a = snap_dir(a / nrm)
    e1, e2 = frame(a)
    h = P @ a
    key = np.round(h, 4)
    rings = []
    for v in np.unique(key):
        q = P[key == v]
        if len(q) >= 3:
            c2, r, rres = circle2d(np.c_[q @ e1, q @ e2])
            if np.isfinite(r) and rres < 1e-3:
                rings.append((float(v), c2, r))
    if len(rings) >= 2:
        hs = np.array([x[0] for x in rings])
        rs = np.array([x[2] for x in rings])
        cs = np.array([x[1] for x in rings]).mean(0)
        k, b = np.polyfit(hs, rs, 1)
        if abs(k) < 1e-9:
            return None
        h0 = -b / k
        apex0 = cs[0] * e1 + cs[1] * e2 + h0 * a
        if k < 0:
            a = -a
            k = -k
    else:
        D = np.einsum("ij,ij->i", N, C)
        apex0, *_ = np.linalg.lstsq(N, D, rcond=None)
        hh = (P - apex0) @ a
        if np.median(hh) < 0:
            a = -a
            hh = -hh
        rr = np.linalg.norm((P - apex0) - np.outer(hh, a), axis=1)
        k = np.median(rr / np.maximum(hh, 1e-9))
    d = P - apex0
    hh = d @ a
    rr = np.linalg.norm(d - np.outer(hh, a), axis=1)
    res0 = np.abs((rr - k * hh) * np.cos(np.arctan(k))).max()
    if res0 > 0.02 * max(rr.max(), 1):
        return a, apex0, float(np.arctan(k)), float(res0)

    def res(x):
        ax = _dir(x[:2])
        ap = x[2:5]
        d = P - ap
        hh = d @ ax
        r = np.linalg.norm(d - np.outer(hh, ax), axis=1)
        return (r - np.tan(x[5]) * hh) * np.cos(x[5])

    try:
        s = least_squares(res, np.r_[_ang(a), apex0, np.arctan(k)], method="lm", xtol=1e-15, ftol=1e-15, max_nfev=200)
    except Exception:
        return None
    ax = _dir(s.x[:2])
    ap = s.x[2:5]
    half = float(s.x[5])
    if half < 0:
        half = -half
    return ax, ap, half, float(np.abs(res(s.x)).max())


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    s, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = s[:3]
    r2 = s[3] + c @ c
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    return c, float(r), float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())


def fit_torus(P, N, C):
    """Axis, centre, major R, minor r, residual."""
    best = None
    for sgn in (1, -1):
        for r0 in np.geomspace(0.1, 20, 12):
            Q = C - sgn * r0 * N
            n, d, res = fit_plane(Q)
            if res > 0.2 * r0 + 0.05:
                continue
            e1, e2 = frame(n)
            c2, R, cres = circle2d(np.c_[Q @ e1, Q @ e2])
            if not np.isfinite(R):
                continue
            c = c2[0] * e1 + c2[1] * e2 + d * n
            score = cres + res
            if best is None or score < best[0]:
                best = (score, n, c, R, r0)
    if best is None:
        return None
    if best[0] > 0.05 * best[4] + 0.02:
        return None
    _, n0, c0, R0, r0 = best

    def res(x):
        a = _dir(x[:2])
        c = x[2:5]
        d = P - c
        h = d @ a
        rho = np.linalg.norm(d - np.outer(h, a), axis=1)
        return np.hypot(rho - x[5], h) - x[6]

    try:
        s = least_squares(res, np.r_[_ang(n0), c0, R0, r0], method="lm", xtol=1e-15, ftol=1e-15, max_nfev=80)
    except Exception:
        return None
    a = snap_dir(_dir(s.x[:2]))
    return a, s.x[2:5], float(s.x[5]), float(abs(s.x[6])), float(np.abs(res(s.x)).max())


def _accept(prim, P, tol):
    d = dist_to(prim, P)
    if d.max() < tol:
        return True
    return len(P) >= 10 and np.quantile(d, 0.6) < tol and d.max() < LOOSE


def fit_best(P, N, C, tol=TOL, allow_torus=True, kinds=None):
    """The simplest primitive the points lie on: dict(kind, ..., res) or None."""
    if len(P) < 3:
        return None
    n, d, res = fit_plane(P)
    ns = snap_dir(n) * np.sign(snap_dir(n) @ n)
    pl = dict(
        kind="plane",
        n=ns if world_axis(ns) else n,
        d=float(np.median(P @ (ns if world_axis(ns) else n))),
        res=res,
    )
    if _accept(pl, P, tol):
        return pl
    cands = []
    allow = lambda k: kinds is None or k in kinds
    if len(P) >= 5 and allow("cylinder"):
        cy = fit_cylinder(P, N)
        if cy and cy[3] < LOOSE and np.median(np.abs(N @ cy[0])) < 0.02:
            cands.append(dict(kind="cylinder", axis=cy[0], c=cy[1], r=cy[2], res=cy[3]))
    if len(P) >= 6 and kinds is None:
        co = fit_cone(P, N, C)
        if co and co[3] < LOOSE and 1e-3 < co[2] < np.pi / 2 - 1e-3:
            ax = snap_dir(co[0]) * np.sign(snap_dir(co[0]) @ co[0])
            cands.append(dict(kind="cone", axis=ax, apex=co[1], half=co[2], res=co[3]))
        sp = fit_sphere(P)
        if sp and sp[2] < LOOSE:
            cands.append(dict(kind="sphere", c=sp[0], r=sp[1], res=sp[2]))
    cands = [c for c in cands if _accept(c, P, tol)]
    if not cands and allow_torus and kinds is None and len(P) >= 12:
        to = fit_torus(P, N, C)
        if to and to[4] < LOOSE and to[3] < to[2] * 50:
            pr = dict(kind="torus", axis=to[0], c=to[1], R=to[2], r=to[3], res=to[4])
            if _accept(pr, P, tol):
                cands.append(pr)
    if not cands:
        return None
    errs = [normal_err(c, C, N) for c in cands]
    order = {"cylinder": 0, "cone": 1, "sphere": 2, "torus": 3}
    ok = [c for c, e in zip(cands, errs) if e <= min(errs) + 1.0]
    return min(ok, key=lambda c: order[c["kind"]])


def surface_normal(prim, Q):
    """Unit normals of a primitive at (points near) Q, up to sign."""
    k = prim["kind"]
    if k == "plane":
        return np.tile(prim["n"], (len(Q), 1))
    if k == "sphere":
        d = Q - prim["c"]
    elif k == "cylinder":
        d = Q - prim["c"]
        d -= np.outer(d @ prim["axis"], prim["axis"])
    elif k == "cone":
        a, t = prim["axis"], prim["half"]
        d = Q - prim["apex"]
        h = d @ a
        rad = d - np.outer(h, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-12)[:, None]
        d = rad * np.cos(t) - np.outer(np.full(len(Q), np.sin(t)), a)
    elif k == "torus":
        a = prim["axis"]
        d0 = Q - prim["c"]
        h = d0 @ a
        rad = d0 - np.outer(h, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-12)[:, None]
        d = d0 - prim["R"] * rad
    else:
        raise ValueError(k)
    return d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]


def normal_err(prim, C, N):
    c = np.abs(np.einsum("ij,ij->i", surface_normal(prim, C), N))
    return float(np.degrees(np.median(np.arccos(np.clip(c, -1, 1)))))


def dist_to(prim, P):
    """Unsigned distance of points to a fitted primitive."""
    k = prim["kind"]
    if k == "plane":
        return np.abs(P @ prim["n"] - prim["d"])
    if k == "cylinder":
        d = P - prim["c"]
        d -= np.outer(d @ prim["axis"], prim["axis"])
        return np.abs(np.linalg.norm(d, axis=1) - prim["r"])
    if k == "cone":
        a, ap, t = prim["axis"], prim["apex"], prim["half"]
        d = P - ap
        h = d @ a
        r = np.linalg.norm(d - np.outer(h, a), axis=1)
        return np.abs((r - np.tan(t) * h) * np.cos(t))
    if k == "sphere":
        return np.abs(np.linalg.norm(P - prim["c"], axis=1) - prim["r"])
    if k == "torus":
        a, c = prim["axis"], prim["c"]
        d = P - c
        h = d @ a
        rho = np.linalg.norm(d - np.outer(h, a), axis=1)
        return np.abs(np.hypot(rho - prim["R"], h) - prim["r"])
    raise ValueError(k)


# ---------------------------------------------------------------------------
# BRep Segmentation of STL Meshes
# ---------------------------------------------------------------------------

class Faces:
    """Face of each triangle (fid), the faces' fitted primitives, and the adjacency between them."""

    def __init__(self, V, F, tol=TOL, smooth_deg=SMOOTH_DEG):
        self.V, self.F, self.tol = V, F, tol
        self.N, self.A = tri_normals(V, F)
        self.C = V[F].mean(1)
        t1, t2, E, self.open_edges = edges_of(F, len(V))
        self.t1, self.t2, self.E = t1, t2, E
        cosd = np.einsum("ij,ij->i", self.N[t1], self.N[t2])
        self.dihedral = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        self.nbr = [[] for _ in range(len(F))]
        for k, (a, b) in enumerate(zip(t1, t2)):
            self.nbr[a].append((b, k))
            self.nbr[b].append((a, k))
        self.piece = pieces_of(F, len(V))
        self._segment(np.radians(smooth_deg))
        self._describe()

    def _grow_planes(self):
        V, F, A = self.V, self.F, self.A
        fid = np.full(len(F), -1)
        prims = []
        order = np.argsort(-A)
        for seed in order:
            if fid[seed] >= 0 or A[seed] < 1e-12:
                continue
            P = V[F[seed]]
            n = self.N[seed]
            d = float(n @ P[0])
            members = [seed]
            fid[seed] = len(prims)
            front = [seed]
            nfit = 1
            while front:
                t = front.pop()
                for u, _ in self.nbr[t]:
                    if fid[u] >= 0:
                        continue
                    if np.abs(V[F[u]] @ n - d).max() < self.tol:
                        fid[u] = len(prims)
                        members.append(u)
                        front.append(u)
                        if len(members) >= 2 * nfit and len(members) >= 3:
                            vs = np.unique(F[members])
                            n2, d2, r2 = fit_plane(V[vs])
                            if r2 < self.tol:
                                n, d = (n2, d2) if n2 @ n > 0 else (-n2, -d2)
                            nfit = len(members)
            prims.append(dict(kind="plane", n=n, d=d, tris=members))
        return fid, prims

    def _segment(self, smooth):
        V, F, N = self.V, self.F, self.N
        _, planes = self._grow_planes()
        real = np.array([len(p["tris"]) >= 3 and len(np.unique(F[p["tris"]])) > 4 for p in planes], bool)
        fid = np.full(len(F), -1)
        prims = []
        for i, p in enumerate(planes):
            if real[i]:
                for t in p["tris"]:
                    fid[t] = len(prims)
                vs = np.unique(F[p["tris"]])
                n, d, res = fit_plane(V[vs]) if len(vs) >= 3 else (p["n"], p["d"], 0.0)
                if n @ p["n"] < 0:
                    n, d = -n, -d
                ns = snap_dir(n)
                ns = ns * np.sign(ns @ n)
                if world_axis(ns):
                    d = float(np.median(V[vs] @ ns))
                    n = ns
                prims.append(dict(kind="plane", n=n, d=d, res=res))
        rest = np.where(fid < 0)[0]
        inrest = np.zeros(len(F), bool)
        inrest[rest] = True
        ok = inrest[self.t1] & inrest[self.t2] & (self.dihedral < np.degrees(smooth))
        g = coo_matrix((np.ones(ok.sum()), (self.t1[ok], self.t2[ok])), shape=(len(F), len(F)))
        _, comp = connected_components(g, directed=False)
        regions = {}
        for t in rest:
            regions.setdefault(comp[t], []).append(t)
        for tris in regions.values():
            for sub, prim in self._explain(np.array(tris)):
                for t in sub:
                    fid[t] = len(prims)
                prims.append(prim)
        self.fid, self.prims = fid, prims
        self._absorb_strips()
        self._merge_same()

    def _components(self, tris):
        inr = np.zeros(len(self.F), bool)
        inr[tris] = True
        ok = inr[self.t1] & inr[self.t2] & (self.dihedral < SMOOTH_DEG)
        g = coo_matrix((np.ones(ok.sum()), (self.t1[ok], self.t2[ok])), shape=(len(self.F), len(self.F)))
        comp = connected_components(g, directed=False)[1][tris]
        return [tris[comp == c] for c in np.unique(comp)]

    def _explain(self, tris, depth=0):
        V, F, N = self.V, self.F, self.N
        vs = np.unique(F[tris])
        ch = self._chain(tris)
        prim = fit_best(V[vs], N[tris], self.C[tris], self.tol, kinds=("cylinder",) if ch["kind"] == "chain" else None)
        if prim is not None:
            return [(tris, prim)]
        if ch["kind"] == "chain":
            return [(tris, ch)]
        if depth < 2:
            for j in range(3):
                sq = np.abs(N[tris, j]) < 2e-3
                if self.A[tris[sq]].sum() > 0.5 * self.A[tris].sum() and (~sq).any():
                    out = []
                    for part in (tris[sq], tris[~sq]):
                        for comp in self._components(part):
                            out += self._explain(comp, depth + 1)
                    return out
        out = []
        left = set(int(t) for t in tris)
        guard = 0
        failed = set()
        while left and guard < 400:
            guard += 1
            best = None
            seeds = [t for t in sorted(left, key=lambda t: -self.A[t]) if t not in failed][:4]
            if not seeds:
                break
            for s in seeds:
                patch, pr = self._grow_prim(s, left)
                if pr is not None and len(patch) >= 4 and self._exact(patch, pr):
                    if best is None or len(patch) > len(best[0]):
                        best = (patch, pr)
                else:
                    failed |= patch if pr is not None else {s}
            if best is None:
                continue
            out.append((np.array(sorted(best[0])), best[1]))
            left -= best[0]
        if left:
            rest = np.array(sorted(left))
            inr = np.zeros(len(F), bool)
            inr[rest] = True
            ok = inr[self.t1] & inr[self.t2]
            g = coo_matrix((np.ones(ok.sum()), (self.t1[ok], self.t2[ok])), shape=(len(F), len(F)))
            comp = connected_components(g, directed=False)[1][rest]
            for c in np.unique(comp):
                sub = rest[comp == c]
                pr = fit_best(V[np.unique(F[sub])], N[sub], self.C[sub], self.tol)
                out.append((sub, pr if pr is not None else self._chain(sub)))
        return out

    def _exact(self, patch, pr):
        d = dist_to(pr, self.V[np.unique(self.F[sorted(patch)])])
        return np.quantile(d, 0.4) < STRICT

    def _grow_prim(self, seed, allowed):
        V, F, N = self.V, self.F, self.N
        patch = {seed}
        front = [seed]
        while front and len(patch) < 6:
            t = front.pop(0)
            for u, k in self.nbr[t]:
                if u in allowed and u not in patch and self.dihedral[k] < SMOOTH_DEG:
                    patch.add(u)
                    front.append(u)
        pl = sorted(patch)
        pr = fit_best(V[np.unique(F[pl])], N[pl], self.C[pl], self.tol)
        if pr is None:
            return {seed}, None
        for _ in range(6):
            front = list(patch)
            stalled = set()
            while front:
                t = front.pop()
                for u, k in self.nbr[t]:
                    if u in allowed and u not in patch and self.dihedral[k] < SMOOTH_DEG:
                        if dist_to(pr, V[F[u]]).max() < self.tol:
                            patch.add(u)
                            front.append(u)
                        else:
                            stalled.add(u)
            stalled -= patch
            if not stalled:
                break
            tl = sorted(patch | stalled)
            p2 = fit_best(V[np.unique(F[tl])], N[tl], self.C[tl], self.tol, allow_torus=len(tl) > 20)
            if p2 is None:
                break
            pr = p2
            patch |= stalled
        return patch, pr

    def _chain(self, tris):
        N = self.N
        M = (N[tris] * self.A[tris, None]).T @ N[tris]
        ev, U = np.linalg.eigh(M)
        if ev[0] / max(ev.sum(), 1e-30) < 1e-5:
            ax = U[:, 0]
            w = world_axis(ax, 2e-3)
            return dict(kind="chain", axis=np.eye(3)[AXES.index(w)] if w else snap_dir(ax), res=None)
        return dict(kind="freeform", res=None)

    def _absorb_strips(self):
        V, F = self.V, self.F
        changed = True
        while changed:
            changed = False
            for i, p in enumerate(self.prims):
                if p["kind"] != "plane" or p.get("dead"):
                    continue
                tris = np.where(self.fid == i)[0]
                if len(tris) > 8:
                    continue
                nb = {}
                for t in tris:
                    for u, k in self.nbr[t]:
                        j = self.fid[u]
                        if j != i and self.dihedral[k] < SMOOTH_DEG:
                            nb[j] = nb.get(j, 0) + 1
                for j in sorted(nb, key=lambda j: -nb[j]):
                    q = self.prims[j]
                    if q["kind"] in ("cylinder", "cone", "sphere", "torus") and dist_to(q, V[np.unique(F[tris])]).max() < self.tol:
                        self.fid[tris] = j
                        p["dead"] = True
                        changed = True
                        break
        alive = [i for i, p in enumerate(self.prims) if not p.get("dead") and (self.fid == i).any()]
        remap = -np.ones(len(self.prims), int)
        remap[alive] = np.arange(len(alive))
        self.fid = remap[self.fid]
        self.prims = [self.prims[i] for i in alive]

    def _merge_same(self):
        V, F = self.V, self.F
        nf = len(self.prims)
        parent = list(range(nf))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        fa, fb = self.fid[self.t1], self.fid[self.t2]
        pairs = set()
        for k in np.where((fa != fb) & (self.dihedral < SMOOTH_DEG))[0]:
            pairs.add((min(fa[k], fb[k]), max(fa[k], fb[k])))
        cnt = np.bincount(self.fid, minlength=nf)
        for a, b in sorted(pairs):
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            pa, pb = self.prims[ra], self.prims[rb]
            if pa["kind"] != pb["kind"] or pa["kind"] not in ("plane", "cylinder", "cone", "sphere", "torus"):
                continue
            big, small = (ra, rb) if cnt[ra] >= cnt[rb] else (rb, ra)
            vs = np.unique(F[self.fid == small])
            if dist_to(self.prims[big], V[vs]).max() < self.tol:
                parent[small] = big
                cnt[big] += cnt[small]
        r = np.array([root(i) for i in range(nf)])
        alive = sorted(set(r.tolist()))
        remap = -np.ones(nf, int)
        remap[alive] = np.arange(len(alive))
        self.fid = remap[r[self.fid]]
        self.prims = [self.prims[i] for i in alive]

    def _describe(self):
        V, F, A = self.V, self.F, self.A
        nf = len(self.prims)
        self.area = np.bincount(self.fid, weights=A, minlength=nf)
        self.tris = [np.where(self.fid == i)[0] for i in range(nf)]
        self.verts = [np.unique(F[t]) for t in self.tris]
        self.fpiece = np.array([self.piece[t[0]] for t in self.tris])
        fa, fb = self.fid[self.t1], self.fid[self.t2]
        cross = fa != fb
        self.adj = {}
        C = V[F].mean(1)
        for k in np.where(cross)[0]:
            a, b = int(fa[k]), int(fb[k])
            ta, tb = self.t1[k], self.t2[k]
            conv = (C[tb] - C[ta]) @ self.N[ta] < 0
            key = (min(a, b), max(a, b))
            e = self.adj.setdefault(key, dict(n=0, dmin=180.0, dmax=0.0, convex=0, length=0.0))
            e["n"] += 1
            e["dmin"] = min(e["dmin"], self.dihedral[k])
            e["dmax"] = max(e["dmax"], self.dihedral[k])
            e["convex"] += int(conv)
            e["length"] += float(np.linalg.norm(V[self.E[k][0]] - V[self.E[k][1]]))
        for e in self.adj.values():
            e["convex"] = e["convex"] * 2 > e["n"]
            e["tangent"] = e["dmax"] < SMOOTH_DEG
        self.neigh = [[] for _ in range(nf)]
        for (a, b), e in self.adj.items():
            self.neigh[a].append(b)
            self.neigh[b].append(a)
        for i, p in enumerate(self.prims):
            P = V[self.verts[i]]
            p["bbox"] = [P.min(0), P.max(0)]
            if p["kind"] in ("cylinder", "cone", "torus"):
                ax = p["axis"]
                base = p["c"] if "c" in p else p["apex"]
                h = (P - base) @ ax
                p["h"] = (float(h.min()), float(h.max()))
                e1, e2 = frame(ax)
                d = P - base
                ang = np.degrees(np.arctan2(d @ e2, d @ e1))
                p["sweep"] = _extent(ang)
                tri = self.tris[i]
                cen = V[F[tri]].mean(1) - base
                rad = cen - np.outer(cen @ ax, ax)
                p["convex"] = bool((np.einsum("ij,ij->i", self.N[tri], rad) > 0).mean() > 0.5)
            if p["kind"] == "sphere":
                tri = self.tris[i]
                p["convex"] = bool((np.einsum("ij,ij->i", self.N[tri], V[F[tri]].mean(1) - p["c"]) > 0).mean() > 0.5)


def _extent(ang):
    a = np.sort(ang)
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.r_[a, a[0] + 360])
    return float(360 - gaps.max())


def nominal(x, steps=(1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001), within=2e-4):
    """The round design value x reads as, coarsest step first; None when nothing is within."""
    for s in steps:
        v = round(x / s) * s
        if abs(x - v) < within:
            return round(v, 6)
    return None


def fmt_num(x, within=2e-4):
    v = nominal(x, within=within)
    return f"{x:.5f}" + (f" (={v:g})" if v is not None and abs(v) > 1e-9 else "")


# ---------------------------------------------------------------------------
# STL Analysis Report (ported from analyze.py)
# ---------------------------------------------------------------------------

def part_frame(fs: Faces):
    """The three directions the planar area lines up with, and whether they are the world axes."""
    dirs = []
    for i, p in enumerate(fs.prims):
        if p["kind"] == "plane":
            dirs.append((np.abs(p["n"]), fs.area[i], p["n"]))
    if not dirs:
        return None
    clusters = []
    for _, a, n in sorted(dirs, key=lambda d: -d[1]):
        n = n * np.sign(n[np.argmax(np.abs(n))])
        for c in clusters:
            if abs(c["n"] @ n) > np.cos(np.radians(0.05)):
                c["a"] += a
                break
        else:
            clusters.append(dict(n=n, a=a))
    clusters.sort(key=lambda c: -c["a"])
    e1 = clusters[0]["n"]
    tot = sum(c["a"] for c in clusters)
    e2 = next(
        (
            c["n"]
            for c in clusters[1:]
            if abs(c["n"] @ e1) < np.sin(np.radians(0.05)) and c["a"] > 0.02 * tot
        ),
        None,
    )
    return clusters, e1, e2


def axis_groups(fs: Faces, tol=1e-3):
    """Cylinders and cones grouped by axis line."""
    groups = []
    for i, p in enumerate(fs.prims):
        if p["kind"] not in ("cylinder", "cone"):
            continue
        a = p["axis"]
        o = p["c"] if p["kind"] == "cylinder" else p["apex"]
        w = world_axis(a, 3e-4)
        if w:
            a = np.eye(3)[AXES.index(w)] * np.sign(a[AXES.index(w)])
        for g in groups:
            if abs(abs(g["a"] @ a) - 1) < 1e-6:
                d = o - g["o"]
                d -= (d @ g["a"]) * g["a"]
                if np.linalg.norm(d) < tol:
                    if world_axis(g["a"]) is None and world_axis(a):
                        g["a"] = a
                    g["faces"].append(i)
                    break
        else:
            groups.append(dict(a=a, o=o, faces=[i]))
    return groups


def axis_point(a, o):
    """The axis line's point in the plane square to it through the origin."""
    o = o - (o @ a) * a
    w = world_axis(a)
    if w:
        return w, [float(o[j]) for j in range(3) if AXES[j] != w]
    return None, o.tolist()


def analyze_stl_file(stl_path: str | Path) -> str:
    """Segment an STL mesh into exact BRep primitive faces and return a structured report."""
    path = Path(stl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STL file not found: '{path}'")

    V, F = load_stl(path)
    fs = Faces(V, F)
    out = io.StringIO()

    lo, hi = V.min(0), V.max(0)
    out.write(
        f"{len(V)} vertices, {len(F)} triangles, open edges {fs.open_edges}"
        f'{" (T-junction cracks: harmless for reading)" if fs.open_edges else " (closed)"}\n'
    )
    out.write(f"box {lo.round(4).tolist()} .. {hi.round(4).tolist()}  size {(hi - lo).round(4).tolist()}\n")
    for pc in np.unique(fs.piece):
        Fp = F[fs.piece == pc]
        out.write(f"  piece {pc}: {len(Fp)} tris, volume {volume(V, Fp):.3f}\n")

    total = float(fs.area.sum()) or 1.0
    analytic_area = sum(
        float(fs.area[i])
        for i, p in enumerate(fs.prims)
        if p["kind"] in ("plane", "cylinder", "cone", "sphere", "torus", "chain")
    )
    analytic_pct = 100.0 * analytic_area / total
    classification = "Clean CAD export" if analytic_pct >= 75.0 else "Noisy scan or freeform mesh"
    out.write(f"\nmesh classification: {classification} ({analytic_pct:.1f}% analytic BRep surface area across {len(fs.prims)} segmented faces)\n")

    # ---- frame
    fr = part_frame(fs)
    aligned = True
    if fr:
        clusters, e1, e2 = fr
        aligned = bool(world_axis(e1, 1e-6) and (e2 is None or world_axis(e2, 1e-6)))
        out.write(
            "\nframe: plane normals (area share): "
            + ", ".join(
                f"{np.round(c['n'], 5).tolist()} {100 * c['a'] / total:.1f}%"
                for c in clusters[:6]
            )
            + "\n"
        )
        if aligned:
            out.write("  the main faces are square to the world axes: build in the STL frame as it is\n")
        else:
            out.write("  the main faces are NOT square to the world axes: rotate into part frame before rebuilding\n")

    # ---- per-axis 2.5D
    out.write("\nper axis: caps square to it (levels = design heights) and walls along it, % of area\n")
    share = {}
    for j, ax in enumerate(AXES):
        e = np.eye(3)[j]
        caps, walls = {}, 0.0
        for i, p in enumerate(fs.prims):
            k = p["kind"]
            if k == "plane":
                c = p["n"] @ e
                if abs(abs(c) - 1) < 1e-9:
                    lv = round(float(p["d"] * np.sign(c)), 5)
                    caps.setdefault(lv, [0.0, 0, 0])
                    caps[lv][0] += fs.area[i]
                    caps[lv][1 if c > 0 else 2] += 1
                elif abs(c) < 1e-9:
                    walls += fs.area[i]
            elif k in ("cylinder", "chain") and abs(abs(p["axis"] @ e) - 1) < 1e-9:
                walls += fs.area[i]
        cap_a = sum(v[0] for v in caps.values())
        share[ax] = (cap_a + walls) / total
        out.write(
            f"  {ax}: caps {100 * cap_a / total:5.1f}%  walls {100 * walls / total:5.1f}%  "
            f"-> 2.5D along {ax}: {100 * (cap_a + walls) / total:5.1f}%\n"
        )
        if caps:
            lv = sorted(caps)
            out.write(
                "     levels: "
                + ", ".join(
                    f"{fmt_num(v)}[{'+' * caps[v][1]}{'-' * caps[v][2]} {caps[v][0]:.1f}]"
                    for v in lv
                )
                + "\n"
            )

    # ---- axes
    out.write("\naxes (cylinders and cones on one axis line; hole = concave, boss = convex; h along the axis):\n")
    groups = axis_groups(fs)
    groups.sort(key=lambda g: -sum(fs.area[i] for i in g["faces"]))
    for gi, g in enumerate(groups):
        w, pt = axis_point(g["a"], g["o"])
        where = (
            f"{w} axis at {[fmt_num(v) for v in pt]}"
            if w
            else f"axis {np.round(g['a'], 6).tolist()} through {np.round(pt, 5).tolist()}"
        )
        out.write(f"  A{gi}: {where}\n")
        for i in sorted(g["faces"], key=lambda i: fs.prims[i].get("h", (0, 0))[0]):
            p = fs.prims[i]
            h0, h1 = p["h"]
            sw = p["sweep"]
            tag = "boss" if p["convex"] else "hole"
            if p["kind"] == "cylinder":
                out.write(
                    f'      cyl  {tag} R {fmt_num(p["r"])}  h {h0:.4f}..{h1:.4f}  sweep {sw:.1f}  area {fs.area[i]:.2f}  [face {i}]\n'
                )
            else:
                t = np.tan(p["half"])
                out.write(
                    f'      cone {tag} half-angle {np.degrees(p["half"]):.4f}  r {abs(h0) * t:.4f}..{abs(h1) * t:.4f}'
                    f"  h {h0:.4f}..{h1:.4f} (from apex)  sweep {sw:.1f}  area {fs.area[i]:.2f}  [face {i}]\n"
                )

    # ---- build recommendation
    turned = None
    for g in groups:
        w = world_axis(g["a"])
        if not w:
            continue
        e = g["a"]
        cov = sum(fs.area[i] for i in g["faces"])
        for i, p in enumerate(fs.prims):
            if p["kind"] == "plane" and abs(abs(p["n"] @ e) - 1) < 1e-6:
                cov += fs.area[i]
            elif p["kind"] in ("torus", "sphere"):
                c = p["c"] - g["o"]
                c -= (c @ e) * e
                if np.linalg.norm(c) < 1e-3:
                    cov += fs.area[i]
        if turned is None or cov > turned[0]:
            turned = (cov, w, g)
    best = max(share, key=share.get) if share else None
    out.write("\n")
    if fr and not aligned:
        out.write("recommend: align frame first - the part stands turned off the world axes\n")
    elif turned and turned[0] / total > 0.8 and turned[0] / total >= share.get(best, 0) - 0.02:
        w, g = turned[1], turned[2]
        pt = [round(float(g["o"][k]), 5) for k in range(3) if AXES[k] != w]
        out.write(
            f"recommend: revolve --axis {w} --at {pt[0]},{pt[1]}  ({100 * turned[0] / total:.0f}% of the area turned about it), "
            f"then cut off-axis holes along --main {w}\n"
        )
    elif best:
        out.write(
            f"recommend: slabs --axis {best}  ({100 * share[best]:.0f}% of the area 2.5D along it), then cut holes along --main {best}\n"
        )

    # ---- other faces
    out.write("\nother faces:\n")
    for i, p in enumerate(fs.prims):
        k = p["kind"]
        b0, b1 = p["bbox"]
        if k == "plane" and not world_axis(p["n"], 1e-9):
            out.write(
                f'  plane oblique n {np.round(p["n"], 6).tolist()} d {p["d"]:.5f} area {fs.area[i]:.2f} box {b0.round(3).tolist()}..{b1.round(3).tolist()} [face {i}]\n'
            )
        elif k == "chain" and not world_axis(p["axis"], 1e-9):
            out.write(
                f'  chain along {np.round(p["axis"], 6).tolist()} area {fs.area[i]:.2f} box {b0.round(3).tolist()}..{b1.round(3).tolist()} [face {i}]\n'
            )
        elif k in ("freeform", "torus", "sphere"):
            extra = ""
            if k == "torus":
                extra = f' R {fmt_num(p["R"])} r {fmt_num(p["r"])} axis {np.round(p["axis"], 5).tolist()} c {np.round(p["c"], 4).tolist()}'
            if k == "sphere":
                extra = f' R {fmt_num(p["r"])} c {np.round(p["c"], 4).tolist()}'
            out.write(
                f"  {k}{extra} area {fs.area[i]:.2f} box {b0.round(3).tolist()}..{b1.round(3).tolist()} [face {i}]\n"
            )

    return out.getvalue().rstrip()


# ---------------------------------------------------------------------------
# 3-Axis Ray-Cast Voxel Comparison (ported from compare.py)
# ---------------------------------------------------------------------------

def ray_intervals(V: np.ndarray, F: np.ndarray, axis: int, g0: np.ndarray, step: float, shape: tuple[int, int]):
    """For rays along `axis` through the grid, the hits per ray as (coordinate, +1 entering / -1 leaving)."""
    j = axis
    u, v = [k for k in range(3) if k != j]
    P = V[F]
    Nj = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])[:, j]
    hits = [[] for _ in range(shape[0] * shape[1])]
    for t in range(len(F)):
        p = P[t]
        A = p[:, [u, v]]
        lo = np.floor((A.min(0) - g0) / step - 0.5).astype(int)
        hi = np.ceil((A.max(0) - g0) / step - 0.5).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(shape) - 1)
        if (hi < lo).any():
            continue
        iu = np.arange(lo[0], hi[0] + 1)
        iv = np.arange(lo[1], hi[1] + 1)
        GU, GV = np.meshgrid(iu, iv, indexing="ij")
        X = g0[0] + (GU + 0.5) * step + 1.3e-7
        Y = g0[1] + (GV + 0.5) * step + 0.7e-7
        a, b, c = A
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-18:
            continue
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
        l3 = 1 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any():
            continue
        z = l1[m] * p[0, j] + l2[m] * p[1, j] + l3[m] * p[2, j]
        s = 1 if Nj[t] < 0 else -1
        for ru, rv, zz in zip(GU[m], GV[m], z):
            hits[ru * shape[1] + rv].append((zz, s))
    return hits


def occupancy(V: np.ndarray, F: np.ndarray, lo: np.ndarray, step: float, dims: np.ndarray) -> np.ndarray:
    """Inside/outside of every voxel, by rays along each axis: (3, nx, ny, nz) booleans."""
    occ = np.zeros((3,) + tuple(dims), bool)
    for j in range(3):
        u, v = [k for k in range(3) if k != j]
        shape = (int(dims[u]), int(dims[v]))
        hits = ray_intervals(V, F, j, np.array([lo[u], lo[v]]), step, shape)
        cz = lo[j] + (np.arange(dims[j]) + 0.5) * step
        for ru in range(shape[0]):
            for rv in range(shape[1]):
                h = hits[ru * shape[1] + rv]
                if len(h) < 2:
                    continue
                h.sort()
                z = np.array([a for a, _ in h])
                s = np.array([b for _, b in h])
                keep = np.r_[True, (np.diff(z) > 1e-9) | (s[1:] != s[:-1])]
                z, s = z[keep], s[keep]
                w = np.cumsum(s)
                inside = np.zeros(dims[j], bool)
                if w[-1] == 0 and w.min() >= 0:
                    for k in np.flatnonzero(w[:-1] > 0):
                        inside |= (cz > z[k]) & (cz < z[k + 1])
                else:
                    z = z[np.r_[True, np.diff(z) > 1e-9]]
                    if len(z) % 2:
                        z = z[:-1]
                    for a, b in zip(z[0::2], z[1::2]):
                        inside |= (cz > a) & (cz < b)
                idx = [slice(None)] * 3
                idx[u] = ru
                idx[v] = rv
                occ[j][tuple(idx)] = inside
    return occ


def compare_meshes(
    reference_stl_path: str | Path,
    model_stl_path: str | Path,
    voxel_size: float = 0.2,
    min_cluster_vol: float = 0.05,
    top_k: int = 15,
) -> str:
    """Compare a generated model STL against a reference STL using 3-axis ray-cast voxel occupancy."""
    ref_path = Path(reference_stl_path).expanduser().resolve()
    mod_path = Path(model_stl_path).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference STL file not found: '{ref_path}'")
    if not mod_path.is_file():
        raise FileNotFoundError(f"Model STL file not found: '{mod_path}'")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    Vm, Fm = load_stl(ref_path)
    Vb, Fb = load_stl(mod_path)
    grid = float(voxel_size)
    lo = np.minimum(Vm.min(0), Vb.min(0)) - 2 * grid
    hi = np.maximum(Vm.max(0), Vb.max(0)) + 2 * grid
    dims = np.ceil((hi - lo) / grid).astype(int)
    cells = int(np.prod(dims))
    out = io.StringIO()
    if cells > 6e7:
        g = float(np.cbrt(np.prod(hi - lo) / 6e7))
        out.write(f"grid {grid} would make {cells:.2e} voxels: using {g:.3f}\n")
        grid = g
        dims = np.ceil((hi - lo) / grid).astype(int)

    om = occupancy(Vm, Fm, lo, grid, dims)
    ob = occupancy(Vb, Fb, lo, grid, dims)
    vox = grid ** 3
    extra = (ob & ~om).sum(0) >= 2
    missing = (om & ~ob).sum(0) >= 2
    vm, vb = volume(Vm, Fm), volume(Vb, Fb)
    ob_open = edges_of(Fb, len(Vb))[3]
    vm_safe = vm if abs(vm) > 1e-12 else 1.0
    out.write(
        f"volume: mesh {vm:.3f}  model {vb:.3f}  ({100 * (vb - vm) / vm_safe:+.3f} %)"
        + (
            f"  - the model mesh has {ob_open} open edges (faces meshed apart along an edge): its volume is"
            " only roughly right; the body volume and the difference below hold"
            if ob_open
            else ""
        )
        + "\n"
    )
    out.write(
        f"difference on a {grid:g} mm grid: extra {extra.sum() * vox:.3f} mm^3, missing {missing.sum() * vox:.3f} mm^3 "
        f"= {100 * (extra.sum() + missing.sum()) * vox / vm_safe:.3f} % of the mesh\n"
    )

    for name, E in (("EXTRA (model has, mesh not)", extra), ("MISSING (mesh has, model not)", missing)):
        lab, n = ndimage.label(E, structure=np.ones((3, 3, 3)))
        if n == 0:
            out.write(f"{name}: none\n")
            continue
        sizes = np.atleast_1d(ndimage.sum(E, lab, index=np.arange(1, n + 1))) * vox
        order = np.argsort(-sizes)
        small = sizes[sizes < min_cluster_vol]
        out.write(
            f"{name}: {n} clusters, {int((sizes >= min_cluster_vol).sum())} of {min_cluster_vol:g} mm^3 or more"
            f" ({len(small)} smaller, {small.sum():.3f} mm^3 in all)\n"
        )
        objs = ndimage.find_objects(lab)
        for rank, k in enumerate(order):
            if sizes[k] < min_cluster_vol or rank >= top_k:
                break
            sl = objs[k]
            b0 = lo + np.array([s.start for s in sl]) * grid
            b1 = lo + np.array([s.stop for s in sl]) * grid
            idx = np.argwhere(lab[sl] == k + 1) + np.array([s.start for s in sl])
            cen = lo + (idx.mean(0) + 0.5) * grid
            out.write(
                f"  {sizes[k]:9.3f} mm^3  box {np.round(b0, 2).tolist()}..{np.round(b1, 2).tolist()}  centre {np.round(cen, 2).tolist()}\n"
            )

    return out.getvalue().rstrip()
