# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) CadQuery Development Team.
# Copyright (c) 2025-2026 CNC Kitchen / ScanRuler Contributors (https://github.com/CNCKitchen/ScanRuler).
# Copyright (c) 2026 CadQuery Contrib / MCP Server Contributors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Core CadQuery execution, rendering, topology audit, cross-sectioning, and STL analysis logic.

CAD operations for the MCP server, independent of MCP response types.
This module handles script execution via CQGI, SVG/PNG rendering, multi-view contact sheets,
2D cross-section slicing, geometry inspection with topology audit and boolean volume guard,
parameter extraction, file export, and CAD-exported STL BRep analysis/comparison.
"""

import io
import math
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cadquery as cq
from cadquery import cqgi
from cadquery.occ_impl.exporters.svg import getSVG
from cadquery.occ_impl.exporters import export


# Standard view projection directions
VIEWS = {
    "isometric": (-1.0, -1.0, 1.0),     # Standard CAD isometric view (Z-up)
    "front": (0, -1, 0),                # Looking at XZ plane from -Y
    "back": (0, 1, 0),                  # Looking at XZ plane from +Y
    "top": (0, 0, 1),                   # Looking at XY plane from +Z
    "bottom": (0, 0, -1),               # Looking at XY plane from -Z
    "left": (-1, 0, 0),                 # Looking at YZ plane from -X
    "right": (1, 0, 0),                 # Looking at YZ plane from +X
    "isometric_back": (1.0, 1.0, 1.0),  # Isometric from opposite corner
}

MULTI_VIEW_ANGLES = ["isometric", "front", "top", "right"]

# Audit & Boolean Guard thresholds (ported from ScanRuler)
VOLUME_SLACK = 0.002
TINY_FACE_MM2 = 0.01
NARROW_FACE_MM = 0.01
SHORT_EDGE_MM = 0.1


# ---------------------------------------------------------------------------
# Result types — plain data, no MCP dependencies
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    """Result of a render operation."""
    images: list[bytes] = field(default_factory=list)
    mime_type: str = ""
    view_names: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TextResult:
    """Result of a geometry inspection or parameter query."""
    text: str = ""
    error: Optional[str] = None


@dataclass
class ExportResult:
    """Result of an export operation."""
    filename: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Boolean Volume Guard (ported from ScanRuler booleanGuard.ts)
# ---------------------------------------------------------------------------

def _fmt_mm3(v: float) -> str:
    return f"{v:.6g} mm³"


def check_boolean_volume(
    kind: str,
    body_vol: float = 0.0,
    tool_vol: float = 0.0,
    result_vol: float = 0.0,
    *,
    common_vol: Optional[float] = None,
    parts: Optional[Sequence[float]] = None,
) -> Optional[str]:
    """Validate boolean operation volume identities within VOLUME_SLACK (0.2%).

    Returns None when the result volume is sound, or a refusal message string
    describing why the boolean failed (e.g., lost volume on a tangency).
    """
    def _off(got: float, want: float, of: float) -> bool:
        return abs(got - want) > VOLUME_SLACK * max(of, 1e-9)

    if kind == "join":
        pts = list(parts) if parts is not None else [body_vol, tool_vol]
        largest = max([0.0, *pts])
        total = sum(pts)
        if result_vol < largest * (1 - VOLUME_SLACK):
            return f"the join lost volume — {_fmt_mm3(result_vol)}, less than the largest part alone, {_fmt_mm3(largest)}"
        if result_vol > total * (1 + VOLUME_SLACK):
            return f"the join made volume — {_fmt_mm3(result_vol)}, more than its parts together, {_fmt_mm3(total)}"
        return None
    if kind == "cut":
        if result_vol > body_vol * (1 + VOLUME_SLACK):
            return f"the cut gained volume — {_fmt_mm3(result_vol)} from {_fmt_mm3(body_vol)}"
        if result_vol < 0:
            return "the cut left less than nothing"
        return None
    if kind == "union":
        largest = max(body_vol, tool_vol)
        total = body_vol + tool_vol
        if result_vol < largest * (1 - VOLUME_SLACK):
            return f"the join lost volume — {_fmt_mm3(result_vol)}, less than the larger body alone, {_fmt_mm3(largest)}"
        if result_vol > total * (1 + VOLUME_SLACK):
            return f"the join made volume — {_fmt_mm3(result_vol)}, more than the two bodies together, {_fmt_mm3(total)}"
        if common_vol is not None and _off(result_vol, total - common_vol, total):
            return f"the join is {_fmt_mm3(result_vol)} where the two bodies less what they share come to {_fmt_mm3(total - common_vol)}"
        return None
    if kind == "subtract":
        if result_vol > body_vol * (1 + VOLUME_SLACK):
            return f"the cut gained volume — {_fmt_mm3(result_vol)} from {_fmt_mm3(body_vol)}"
        if result_vol < body_vol - tool_vol - VOLUME_SLACK * body_vol:
            return f"the cut took more than the tool holds — {_fmt_mm3(result_vol)} left of {_fmt_mm3(body_vol)} by a tool of {_fmt_mm3(tool_vol)}"
        if common_vol is not None and _off(result_vol, body_vol - common_vol, body_vol):
            return f"the cut left {_fmt_mm3(result_vol)} where the body less what it shares with the tool comes to {_fmt_mm3(body_vol - common_vol)}"
        return None
    if kind == "intersect":
        smaller = min(body_vol, tool_vol)
        if result_vol > smaller * (1 + VOLUME_SLACK):
            return f"what the bodies share came to {_fmt_mm3(result_vol)}, more than the smaller holds, {_fmt_mm3(smaller)}"
        return None
    raise ValueError(f"Unknown boolean kind '{kind}'. Expected join, cut, union, subtract, or intersect.")


volume_guard = check_boolean_volume


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opacity(value):
    """Validate opacity before passing it to OCCT or VTK."""
    if not 0 <= value <= 1:
        raise ValueError("Opacity must be between 0 (transparent) and 1 (opaque).")
    return value


def _display_color(options) -> cq.Color | None:
    """Read CQ-editor display options, with rgba taking precedence."""
    if "rgba" in options:
        r, g, b, alpha = options["rgba"]
        options = {"color": (r, g, b), "alpha": alpha}

    color = options.get("color")
    if isinstance(color, str):
        color = cq.Color(color)
    elif color is not None and not isinstance(color, cq.Color):
        components = tuple(color)
        if len(components) not in (3, 4):
            raise ValueError("Display colors must contain RGB or RGBA components.")
        # CQ-editor selects the RGB scale from the first component's type.
        byte_rgb = isinstance(components[0], int)
        rgb = tuple(value / 255 for value in components[:3]) if byte_rgb else components[:3]
        alpha = components[3] if len(components) == 4 else 1.0
        if byte_rgb and isinstance(alpha, int):
            alpha /= 255
        color = cq.Color(*rgb, _opacity(alpha))

    if color is None and "alpha" not in options:
        return None
    r, g, b, alpha = color.toTuple() if color is not None else (0.8, 0.8, 0.8, 1.0)
    return cq.Color(r, g, b, _opacity(options.get("alpha", alpha)))


def _extract_shape(build_result):
    """Preserve all show_object outputs and their display colors."""
    displayed = build_result.results
    if len(displayed) == 1 and not displayed[0].options:
        return displayed[0].shape
    if displayed:
        assembly = cq.Assembly(name="displayed")
        names = [
            str((item.options or {}).get("name") or f"object_{index}")
            for index, item in enumerate(displayed)
        ]
        reserved = set(names)
        for item, base in zip(displayed, names):
            name = base
            if name in assembly.objects:
                suffix = 2
                while f"{base}_{suffix}" in reserved or f"{base}_{suffix}" in assembly.objects:
                    suffix += 1
                name = f"{base}_{suffix}"
            assembly.add(item.shape, name=name, color=_display_color(item.options or {}))
        return assembly
    return build_result.env.get("result")


def _to_occ_shape(shape):
    """Flatten geometry only when needed, retaining every part and location."""
    if isinstance(shape, cq.Assembly):
        return shape.toCompound()
    if isinstance(shape, cq.Workplane):
        return cq.Compound.makeCompound(
            item for item in shape.vals() if isinstance(item, cq.Shape)
        )
    return shape


def _build_model(model):
    """Execute a CQGI model with independent options for each displayed object."""
    collector = cqgi.ScriptCallback()

    def show_object(shape, options=None, **kwargs):
        # CQGI's callback otherwise mutates and reuses its default options dict.
        collector.show_object(shape, options={**(options or {}), **kwargs})

    env = (
        cqgi.EnvironmentBuilder()
        .with_real_builtins()
        .with_cadquery_objects()
        .add_entry("__name__", "__cqgi__")
        .add_entry("show_object", show_object)
        .add_entry("debug", collector.debug)
        .add_entry("describe_parameter", collector.describe_parameter)
        .build()
    )
    # stdout belongs to the MCP transport; keep Python script output on stderr.
    with redirect_stdout(sys.stderr):
        exec(compile(model.ast_tree, cqgi.CQSCRIPT, "exec"), env)
    result = cqgi.BuildResult()
    result.set_success_result(collector.outputObjects)
    result.env = env
    return result


def _execute_code(code: str):
    """Execute CQGI code, retaining assembly metadata and reporting CAD errors."""
    try:
        result = _build_model(cqgi.parse(code))
        shape = _extract_shape(result)
        if shape is None:
            return None, "No shape produced. Use show_object(shape) or assign to 'result' variable."
        geometry = _to_occ_shape(shape)
        if not isinstance(geometry, cq.Shape):
            raise TypeError("Expected a CadQuery Shape, Workplane, or Assembly.")
        if geometry.isNull() or not (geometry.Vertices() or geometry.Edges() or geometry.Faces()):
            return None, "No geometry produced. The model is empty."
        return shape, None
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except (Exception, SystemExit):
        # A script's sys.exit() must not stop the MCP server process.
        return None, f"Execution error:\n{traceback.format_exc()}"


def render_svg(shape, view_name: str, width: int, height: int,
               show_hidden: bool = True, background_card: bool = True) -> str:
    """Render an OCC shape to an SVG string from a specific view angle."""
    opts = {
        "width": width,
        "height": height,
        "projectionDir": VIEWS[view_name],
        "showAxes": view_name in ("isometric", "isometric_back"),
        "showHidden": show_hidden,
    }

    svg_str = getSVG(_to_occ_shape(shape), opts=opts)
    if background_card:
        import re
        match = re.search(r"<svg[^>]*>", svg_str, re.DOTALL)
        if match:
            insert_pos = match.end()
            bg_rect = '\n  <rect width="100%" height="100%" fill="#ffffff" rx="8" stroke="#cbd5e1" stroke-width="2"/>'
            svg_str = svg_str[:insert_pos] + bg_rect + svg_str[insert_pos:]
    return svg_str


def render_png(shape, view_name: str, width: int, height: int,
               show_axes: bool = True, show_edges: bool = True) -> bytes:
    """Render a shaded PNG offscreen, preserving assembly colors and locations."""
    from cadquery.occ_impl.assembly import toVTKAssy
    from vtkmodules.vtkRenderingCore import (
        vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter, vtkRenderWindowInteractor,
    )
    from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkIOImage import vtkPNGWriter
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - registers the rendering backend

    assembly = shape if isinstance(shape, cq.Assembly) else cq.Assembly(
        shape, color=cq.Color(0.7, 0.75, 0.8)
    )
    renderer = vtkRenderer()
    renderer.SetBackground(0.95, 0.96, 0.98)
    for part, _, location, color in assembly:
        wire_only = not part.Faces()
        rgba = color.toTuple() if color is not None else (1.0, 1.0, 1.0, 1.0)
        leaf = cq.Assembly(part, loc=location, color=color)
        edges_to_draw = wire_only or show_edges
        edge_color = rgba if wire_only else (0.15, 0.15, 0.2, 1.0)
        for actor in toVTKAssy(leaf, edges=edges_to_draw, edgecolor=edge_color, tolerance=0.1):
            if wire_only:
                actor.GetProperty().SetOpacity(rgba[3])
            renderer.AddActor(actor)
    window = vtkRenderWindow()
    try:
        window.SetOffScreenRendering(1)
        window.SetSize(width, height)
        window.SetMultiSamples(0)
        window.AddRenderer(renderer)
        camera = renderer.GetActiveCamera()
        camera.SetPosition(*VIEWS[view_name])
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(*((0, 1, 0) if view_name in ("top", "bottom") else (0, 0, 1)))
        camera.ParallelProjectionOn()
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()

        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)

        if show_axes:
            axes = vtkAxesActor()
            for actor, col in [
                (axes.GetXAxisCaptionActor2D(), (0.8, 0.1, 0.1)),
                (axes.GetYAxisCaptionActor2D(), (0.1, 0.55, 0.1)),
                (axes.GetZAxisCaptionActor2D(), (0.1, 0.35, 0.8)),
            ]:
                tp = actor.GetCaptionTextProperty()
                tp.ShadowOff()
                tp.ItalicOff()
                tp.BoldOn()
                tp.SetColor(*col)

            widget = vtkOrientationMarkerWidget()
            widget.SetOrientationMarker(axes)
            widget.SetInteractor(interactor)
            widget.SetViewport(0.0, 0.0, 0.22, 0.22)
            widget.SetEnabled(1)
            widget.InteractiveOff()

        window.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtkPNGWriter()
        writer.SetWriteToMemory(True)
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        return bytes(memoryview(writer.GetResult()))
    finally:
        window.Finalize()


def _composite_contact_sheet(images: list[bytes], view_names: list[str], width: int, height: int) -> bytes:
    """Composite 4 rendered PNG views into a single 2x2 grid contact sheet with view labels."""
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (width * 2, height * 2), color=(242, 245, 250))
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    for img_bytes, name, (x0, y0) in zip(images, view_names, positions):
        tile = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if tile.size != (width, height):
            tile = tile.resize((width, height))
        sheet.paste(tile, (x0, y0))

    draw = ImageDraw.Draw(sheet)
    # Divider lines
    draw.line([(width, 0), (width, height * 2)], fill=(203, 213, 225), width=2)
    draw.line([(0, height), (width * 2, height)], fill=(203, 213, 225), width=2)

    # View label badges
    for name, (x0, y0) in zip(view_names, positions):
        label = name.upper()
        badge_w = max(56, len(label) * 8 + 16)
        badge_h = 22
        bx, by = x0 + 8, y0 + 8
        draw.rectangle([bx, by, bx + badge_w, by + badge_h], fill=(30, 41, 59), outline=(148, 163, 184))
        draw.text((bx + 8, by + 5), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def _slice_shape_2d(shape, section_axis: str, section_offset: float):
    """Slice an OCC shape with a plane perpendicular to X, Y, or Z at section_offset."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    axis = section_axis.strip().upper()
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"Unknown section_axis '{section_axis}'. Must be 'X', 'Y', or 'Z'.")

    occ_shape = _to_occ_shape(shape)
    if axis == "X":
        pln = gp_Pln(gp_Pnt(float(section_offset), 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
        u_idx, v_idx = 1, 2
        u_label, v_label = "Y", "Z"
    elif axis == "Y":
        pln = gp_Pln(gp_Pnt(0.0, float(section_offset), 0.0), gp_Dir(0.0, 1.0, 0.0))
        u_idx, v_idx = 0, 2
        u_label, v_label = "X", "Z"
    else:
        pln = gp_Pln(gp_Pnt(0.0, 0.0, float(section_offset)), gp_Dir(0.0, 0.0, 1.0))
        u_idx, v_idx = 0, 1
        u_label, v_label = "X", "Y"

    sec = BRepAlgoAPI_Section(occ_shape.wrapped, pln)
    sec.Build()
    sec_shape = cq.Shape(sec.Shape())
    edges = sec_shape.Edges() if not sec_shape.isNull() else []
    polylines: list[list[tuple[float, float]]] = []
    umin, umax = float("inf"), float("-inf")
    vmin, vmax = float("inf"), float("-inf")

    for edge in edges:
        curve = BRepAdaptor_Curve(edge.wrapped)
        defl = GCPnts_QuasiUniformDeflection(curve, 0.02)
        pts_2d: list[tuple[float, float]] = []
        if defl.IsDone() and defl.NbPoints() >= 2:
            for i in range(1, defl.NbPoints() + 1):
                p = curve.Value(defl.Parameter(i))
                coords = (p.X(), p.Y(), p.Z())
                u, v = coords[u_idx], coords[v_idx]
                pts_2d.append((u, v))
                umin, umax = min(umin, u), max(umax, u)
                vmin, vmax = min(vmin, v), max(vmax, v)
        if len(pts_2d) >= 2:
            polylines.append(pts_2d)

    return axis, polylines, (u_label, v_label), (umin, umax, vmin, vmax)


def _choose_grid_step(span: float) -> float:
    """Choose a clean millimeter grid step for a 2D cross-section."""
    if span <= 0:
        return 1.0
    raw = span / 8.0
    mag = 10.0 ** math.floor(math.log10(max(raw, 1e-6)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if mult * mag >= raw:
            return mult * mag
    return 10.0 * mag


def render_section_svg(
    shape,
    section_axis: str,
    section_offset: float,
    width: int,
    height: int,
    background_card: bool = True,
) -> str:
    """Render a 2D cross-section slice onto an SVG with a millimeter coordinate grid and dimensions."""
    axis, polylines, (u_label, v_label), (umin, umax, vmin, vmax) = _slice_shape_2d(
        shape, section_axis, section_offset
    )
    if not polylines:
        raise ValueError(
            f"Section plane {axis}={section_offset} does not intersect the model."
        )

    du = max(umax - umin, 1e-6)
    dv = max(vmax - vmin, 1e-6)
    span = max(du, dv)
    step = _choose_grid_step(span)
    pad = max(step, 0.15 * span)

    view_u0, view_u1 = umin - pad, umax + pad
    view_v0, view_v1 = vmin - pad, vmax + pad
    margin = 48.0
    avail_w = max(width - 2 * margin, 32.0)
    avail_h = max(height - 2 * margin - 24.0, 32.0)
    scale = min(avail_w / (view_u1 - view_u0), avail_h / (view_v1 - view_v0))

    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)
    cx = width * 0.5
    cy = (height - 20.0) * 0.5

    def to_px(u: float, v: float) -> tuple[float, float]:
        return cx + (u - uc) * scale, cy - (v - vc) * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    if background_card:
        lines.append(
            '  <rect width="100%" height="100%" fill="#ffffff" rx="8" stroke="#cbd5e1" stroke-width="2"/>'
        )

    # Grid lines
    g_u0 = math.floor(view_u0 / step) * step
    g_u1 = math.ceil(view_u1 / step) * step
    g_v0 = math.floor(view_v0 / step) * step
    g_v1 = math.ceil(view_v1 / step) * step

    lines.append('  <g class="grid" stroke="#e2e8f0" stroke-width="1">')
    u_val = g_u0
    while u_val <= g_u1 + 1e-9:
        x_px, _ = to_px(u_val, 0.0)
        if margin <= x_px <= width - margin:
            stroke = "#94a3b8" if abs(u_val) < 1e-9 else "#e2e8f0"
            sw = "1.5" if abs(u_val) < 1e-9 else "1"
            lines.append(
                f'    <line x1="{x_px:.2f}" y1="{margin:.2f}" x2="{x_px:.2f}" y2="{height - margin - 20:.2f}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
            lines.append(
                f'    <text x="{x_px:.2f}" y="{height - margin - 4:.2f}" font-family="monospace" font-size="10" fill="#475569" text-anchor="middle">{u_val:g}</text>'
            )
        u_val += step

    v_val = g_v0
    while v_val <= g_v1 + 1e-9:
        _, y_px = to_px(0.0, v_val)
        if margin <= y_px <= height - margin - 20:
            stroke = "#94a3b8" if abs(v_val) < 1e-9 else "#e2e8f0"
            sw = "1.5" if abs(v_val) < 1e-9 else "1"
            lines.append(
                f'    <line x1="{margin:.2f}" y1="{y_px:.2f}" x2="{width - margin:.2f}" y2="{y_px:.2f}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
            lines.append(
                f'    <text x="{margin - 6:.2f}" y="{y_px + 3:.2f}" font-family="monospace" font-size="10" fill="#475569" text-anchor="end">{v_val:g}</text>'
            )
        v_val += step
    lines.append("  </g>")

    # Cross-section polylines
    lines.append('  <g class="section-edges" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">')
    for poly in polylines:
        d_parts = []
        for idx, (u, v) in enumerate(poly):
            px, py = to_px(u, v)
            cmd = "M" if idx == 0 else "L"
            d_parts.append(f"{cmd} {px:.2f} {py:.2f}")
        lines.append(f'    <path d="{" ".join(d_parts)}"/>')
    lines.append("  </g>")

    # Bounding dimensions & title annotation
    title = (
        f"Cross-Section {axis} = {section_offset:.4g} mm | "
        f"{u_label}: [{umin:.4g}, {umax:.4g}] (size: {umax - umin:.4g} mm) | "
        f"{v_label}: [{vmin:.4g}, {vmax:.4g}] (size: {vmax - vmin:.4g} mm) | "
        f"grid: {step:g} mm"
    )
    lines.append(
        f'  <text x="{width / 2:.2f}" y="26" font-family="monospace" font-size="12" font-weight="bold" fill="#0f172a" text-anchor="middle">{title}</text>'
    )
    lines.append(
        f'  <text x="{width - margin:.2f}" y="{height - 10:.2f}" font-family="monospace" font-size="11" fill="#334155" text-anchor="end">+{u_label} (mm) → / ↑ +{v_label} (mm)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def render_section_png(
    shape,
    section_axis: str,
    section_offset: float,
    width: int,
    height: int,
) -> bytes:
    """Render a 2D cross-section slice with millimeter coordinate grid and dimensions as PNG."""
    from PIL import Image, ImageDraw

    axis, polylines, (u_label, v_label), (umin, umax, vmin, vmax) = _slice_shape_2d(
        shape, section_axis, section_offset
    )
    if not polylines:
        raise ValueError(
            f"Section plane {axis}={section_offset} does not intersect the model."
        )

    du = max(umax - umin, 1e-6)
    dv = max(vmax - vmin, 1e-6)
    span = max(du, dv)
    step = _choose_grid_step(span)
    pad = max(step, 0.15 * span)

    view_u0, view_u1 = umin - pad, umax + pad
    view_v0, view_v1 = vmin - pad, vmax + pad
    margin = 48.0
    avail_w = max(width - 2 * margin, 32.0)
    avail_h = max(height - 2 * margin - 24.0, 32.0)
    scale = min(avail_w / (view_u1 - view_u0), avail_h / (view_v1 - view_v0))

    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)
    cx = width * 0.5
    cy = (height - 20.0) * 0.5

    def to_px(u: float, v: float) -> tuple[float, float]:
        return cx + (u - uc) * scale, cy - (v - vc) * scale

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    g_u0 = math.floor(view_u0 / step) * step
    g_u1 = math.ceil(view_u1 / step) * step
    g_v0 = math.floor(view_v0 / step) * step
    g_v1 = math.ceil(view_v1 / step) * step

    u_val = g_u0
    while u_val <= g_u1 + 1e-9:
        x_px, _ = to_px(u_val, 0.0)
        if margin <= x_px <= width - margin:
            col = (148, 163, 184) if abs(u_val) < 1e-9 else (226, 232, 240)
            draw.line([(x_px, margin), (x_px, height - margin - 20)], fill=col, width=1)
            draw.text((x_px - 8, height - margin - 14), f"{u_val:g}", fill=(71, 85, 105))
        u_val += step

    v_val = g_v0
    while v_val <= g_v1 + 1e-9:
        _, y_px = to_px(0.0, v_val)
        if margin <= y_px <= height - margin - 20:
            col = (148, 163, 184) if abs(v_val) < 1e-9 else (226, 232, 240)
            draw.line([(margin, y_px), (width - margin, y_px)], fill=col, width=1)
            draw.text((8, y_px - 6), f"{v_val:g}", fill=(71, 85, 105))
        v_val += step

    for poly in polylines:
        pts = [to_px(u, v) for u, v in poly]
        if len(pts) >= 2:
            draw.line(pts, fill=(15, 23, 42), width=2)

    title = (
        f"Section {axis}={section_offset:.4g}mm | "
        f"{u_label}:[{umin:.4g},{umax:.4g}] ({umax - umin:.4g}mm) "
        f"{v_label}:[{vmin:.4g},{vmax:.4g}] ({vmax - vmin:.4g}mm)"
    )
    draw.text((12, 10), title, fill=(15, 23, 42))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CAD operations — return failures before MCP response handling
# ---------------------------------------------------------------------------

def handle_render(
    code: str,
    *,
    view: str = "isometric",
    multi_view: bool = False,
    contact_sheet: bool = False,
    width: int = 800,
    height: int = 600,
    show_hidden: bool = True,
    image_format: str = "svg",
    show_axes: bool = True,
    show_edges: bool = True,
    background_card: bool = True,
    section_axis: Optional[str] = None,
    section_offset: float = 0.0,
) -> RenderResult:
    """Execute CadQuery code and return SVGs, shaded PNGs, contact sheets, or 2D cross-sections."""
    try:
        if image_format not in ("svg", "png"):
            return RenderResult(error="Render error: image_format must be 'svg' or 'png'.")
        if view not in VIEWS:
            return RenderResult(error=f"Render error: unknown view '{view}'.")
        # Reject boolean dimensions.
        if not all(type(size) is int and 16 <= size <= 4096 for size in (width, height)):
            return RenderResult(error="Render error: width and height must be integers from 16 to 4096.")
        if section_axis is not None and (not isinstance(section_axis, str) or section_axis.strip().upper() not in ("X", "Y", "Z")):
            return RenderResult(error=f"Render error: section_axis must be 'X', 'Y', or 'Z', got '{section_axis}'.")

        shape, err = _execute_code(code)
        if err:
            return RenderResult(error=err)

        if section_axis is not None:
            axis_norm = section_axis.strip().upper()
            sec_name = f"section_{axis_norm.lower()}_{section_offset:g}"
            if image_format == "png":
                img_bytes = render_section_png(shape, axis_norm, float(section_offset), width, height)
                return RenderResult(images=[img_bytes], mime_type="image/png", view_names=[sec_name])
            else:
                svg_bytes = render_section_svg(
                    shape, axis_norm, float(section_offset), width, height, background_card
                ).encode("utf-8")
                return RenderResult(images=[svg_bytes], mime_type="image/svg+xml", view_names=[sec_name])

        names = list(MULTI_VIEW_ANGLES) if multi_view else [view]
        if image_format == "png":
            images = [
                render_png(shape, name, width, height, show_axes, show_edges)
                for name in names
            ]
            if multi_view and contact_sheet:
                sheet = _composite_contact_sheet(images, names, width, height)
                return RenderResult(
                    images=[sheet],
                    mime_type="image/png",
                    view_names=["contact_sheet"],
                )
            mime_type = "image/png"
        else:
            images = [
                render_svg(shape, name, width, height, show_hidden, background_card).encode("utf-8")
                for name in names
            ]
            mime_type = "image/svg+xml"
        return RenderResult(images=images, mime_type=mime_type, view_names=names)
    except Exception as e:
        return RenderResult(error=f"Render error: {type(e).__name__}: {e}")


def handle_inspect(
    code: str,
    *,
    section_axis: Optional[str] = None,
    section_offset: float = 0.0,
) -> TextResult:
    """Execute CadQuery code and return geometry information, BRep validity, and topology audit."""
    try:
        if section_axis is not None and (not isinstance(section_axis, str) or section_axis.strip().upper() not in ("X", "Y", "Z")):
            return TextResult(error=f"Inspect error: section_axis must be 'X', 'Y', or 'Z', got '{section_axis}'.")

        shape, err = _execute_code(code)
        if err:
            return TextResult(error=err)

        shape = _to_occ_shape(shape)
        bb = shape.BoundingBox()
        lines = [
            "Geometry Information:",
            "  Bounding Box:",
            f"    X: {bb.xmin:.4f} to {bb.xmax:.4f} (size: {bb.xlen:.4f})",
            f"    Y: {bb.ymin:.4f} to {bb.ymax:.4f} (size: {bb.ylen:.4f})",
            f"    Z: {bb.zmin:.4f} to {bb.zmax:.4f} (size: {bb.zlen:.4f})",
        ]

        try:
            lines.append(f"  Volume: {shape.Volume():.4f}")
        except Exception:
            pass

        try:
            lines.append(f"  Surface Area: {shape.Area():.4f}")
        except Exception:
            pass

        try:
            com = shape.Center()
            lines.append(f"  Center of Mass: ({com.x:.4f}, {com.y:.4f}, {com.z:.4f})")
        except Exception:
            pass

        try:
            solids = shape.Solids()
            faces = shape.Faces()
            edges = shape.Edges()
            vertices = shape.Vertices()
            lines.extend([
                "  Topology:",
                f"    Solids: {len(solids)}",
                f"    Faces: {len(faces)}",
                f"    Edges: {len(edges)}",
                f"    Vertices: {len(vertices)}",
            ])

            try:
                valid_str = "Yes" if shape.isValid() else "No"
                lines.append(f"    Valid (BRepCheck): {valid_str}")
            except Exception:
                pass

            try:
                closed_count = 0
                for s in solids:
                    shells = s.Shells() if hasattr(s, "Shells") else []
                    is_closed = (bool(shells) and all(sh.Closed() for sh in shells)) or (
                        hasattr(s, "Closed") and s.Closed()
                    )
                    if is_closed:
                        closed_count += 1
                lines.append(f"    Closed Solids: {closed_count}/{len(solids)}")
            except Exception:
                pass

            try:
                warnings: list[str] = []
                for idx, f in enumerate(faces):
                    f_area = f.Area()
                    f_edges = f.Edges() if hasattr(f, "Edges") else []
                    perim = sum(e.Length() for e in f_edges) if f_edges else 0.0
                    width = (2.0 * f_area / perim) if perim > 1e-12 else float("inf")
                    if f_area < TINY_FACE_MM2:
                        warnings.append(
                            f"Tiny/sliver face #{idx}: area={f_area:.6f} mm² (< {TINY_FACE_MM2} mm²)"
                        )
                    elif width < NARROW_FACE_MM:
                        warnings.append(
                            f"Narrow/sliver face #{idx}: width={width:.6f} mm (< {NARROW_FACE_MM} mm)"
                        )
                for idx, e in enumerate(edges):
                    e_len = e.Length()
                    if 0.0 < e_len < SHORT_EDGE_MM:
                        warnings.append(
                            f"Short edge #{idx}: length={e_len:.6f} mm (< {SHORT_EDGE_MM} mm, may block fillets)"
                        )
                if warnings:
                    lines.append("  Audit Warnings:")
                    for w in warnings[:15]:
                        lines.append(f"    - {w}")
                else:
                    lines.append("  Audit Warnings: None")
            except Exception:
                pass
        except Exception:
            pass

        if section_axis is not None:
            axis, polylines, (u_label, v_label), (umin, umax, vmin, vmax) = _slice_shape_2d(
                shape, section_axis, float(section_offset)
            )
            lines.append(f"  Cross-Section ({axis} = {section_offset:.4f}):")
            lines.append(f"    Section Edges: {len(polylines)}")
            if polylines:
                lines.append(f"    {u_label}: {umin:.4f} to {umax:.4f} (size: {umax - umin:.4f})")
                lines.append(f"    {v_label}: {vmin:.4f} to {vmax:.4f} (size: {vmax - vmin:.4f})")

        return TextResult(text="\n".join(lines))
    except Exception as e:
        return TextResult(error=f"Inspect error: {type(e).__name__}: {e}")


def handle_get_parameters(code: str) -> TextResult:
    """Parse CadQuery code and extract parameters."""
    try:
        model = cqgi.parse(code)
        params = model.metadata.parameters
        if not params:
            return TextResult(text="No parameters found in the script.")

        lines = ["Parameters found:"]
        for name, param in params.items():
            type_name = param.varType.__name__ if param.varType else "unknown"
            lines.append(f"  {name}: {type_name} = {param.default_value}")
            if param.desc:
                lines.append(f"    Description: {param.desc}")
            if param.valid_values:
                lines.append(f"    Valid values: {param.valid_values}")

        return TextResult(text="\n".join(lines))
    except SyntaxError as e:
        return TextResult(error=f"Syntax error: {e}")
    except Exception as e:
        return TextResult(error=f"Parameter error: {type(e).__name__}: {e}")


def handle_export(code: str, filename: str,
                  export_format: str | None = None) -> ExportResult:
    """Export a model, replacing an existing file only after a successful write."""
    try:
        shape, err = _execute_code(code)
        if err:
            return ExportResult(error=err)

        destination = Path(filename).expanduser().absolute()
        fmt = (export_format or destination.suffix.lstrip(".")).upper()
        fmt = {"STP": "STEP"}.get(fmt, fmt)
        if not destination.parent.is_dir():
            return ExportResult(error=(
                f"Export error: parent directory '{destination.parent}' "
                "does not exist or is not a directory."
            ))
        if not fmt:
            return ExportResult(error=(
                f"Export error: cannot infer format from '{filename}'; pass format explicitly."
            ))
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.", dir=destination.parent,
        ) as directory:
            temporary = Path(directory) / f"model.{fmt.lower()}"
            if isinstance(shape, cq.Assembly) and fmt == "STEP":
                shape.export(str(temporary), exportType=fmt)
            else:
                export(_to_occ_shape(shape), str(temporary), exportType=fmt)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Exporter did not write a non-empty file.")
            temporary.replace(destination)
        return ExportResult(filename=str(destination))
    except Exception as e:
        return ExportResult(error=f"Export error: {type(e).__name__}: {e}")


def handle_analyze_stl(stl_path: str) -> TextResult:
    """Segment an STL mesh into exact BRep primitive faces and recommend a CadQuery rebuild strategy."""
    try:
        from stl_analysis import analyze_stl_file

        report = analyze_stl_file(stl_path)
        return TextResult(text=report)
    except Exception as e:
        return TextResult(error=f"Analyze STL error: {type(e).__name__}: {e}")


def handle_compare_mesh(
    code: str,
    reference_stl_path: str,
    voxel_size: float = 0.2,
) -> TextResult:
    """Execute CadQuery code, export to STL, and run 3-axis ray-cast voxel comparison against reference_stl_path."""
    try:
        from stl_analysis import compare_meshes

        shape, err = _execute_code(code)
        if err:
            return TextResult(error=err)

        with tempfile.TemporaryDirectory(prefix=".cq_compare_") as directory:
            model_stl = Path(directory) / "model.stl"
            export(_to_occ_shape(shape), str(model_stl), exportType="STL")
            if not model_stl.is_file() or model_stl.stat().st_size == 0:
                raise RuntimeError("Exporter did not write a non-empty STL file.")
            report = compare_meshes(reference_stl_path, model_stl, voxel_size=voxel_size)
        return TextResult(text=report)
    except Exception as e:
        return TextResult(error=f"Compare mesh error: {type(e).__name__}: {e}")
