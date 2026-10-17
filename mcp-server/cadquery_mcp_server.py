# Copyright (c) CadQuery Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CadQuery MCP (Model Context Protocol) Server.

This module provides an MCP server that allows AI assistants like Claude
to execute CadQuery scripts and receive rendered images of 3D models.

Usage:
    Run as a standalone server:
        python -m cadquery_mcp_server

    Or use the entry point:
        cadquery-mcp

Configuration in Claude Code (~/.claude/settings.json):
    {
        "mcpServers": {
            "cadquery": {
                "command": "cadquery-mcp"
            }
        }
    }
"""

import asyncio
import base64
from contextlib import redirect_stdout
import math
import sys
import tempfile
import traceback
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

import cadquery as cq
from cadquery import cqgi
from cadquery.occ_impl.exporters import export
from cadquery.occ_impl.shapes import Shape, Compound, TOLERANCE
from cadquery.occ_impl.exporters.svg import getPaths, PATHTEMPLATE

from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCP.BRepLib import BRepLib
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector


server = Server("cadquery")

# Standard view projection directions (Z-up CAD conventions)
VIEWS = {
    "isometric": (-1.0, -1.0, 1.0),     # Standard CAD isometric view (looking from -X, -Y, +Z)
    "front": (0.0, -1.0, 0.0),          # Looking at XZ plane from -Y (+X right, +Z up)
    "back": (0.0, 1.0, 0.0),            # Looking at XZ plane from +Y (-X right, +Z up)
    "top": (0.0, 0.0, 1.0),             # Looking at XY plane from +Z (+X right, +Y up)
    "bottom": (0.0, 0.0, -1.0),         # Looking at XY plane from -Z (+X right, -Y up)
    "left": (-1.0, 0.0, 0.0),           # Looking at YZ plane from -X (-Y right, +Z up)
    "right": (1.0, 0.0, 0.0),           # Looking at YZ plane from +X (+Y right, +Z up)
    "isometric_back": (1.0, 1.0, 1.0),  # Isometric from opposite corner (+X, +Y, +Z)
}


SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   width="%(width)s"
   height="%(height)s"
>
    <defs>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#c62828" />
        </marker>
        <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#2e7d32" />
        </marker>
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1565c0" />
        </marker>
    </defs>
    <g transform="scale(%(unitScale)s, -%(unitScale)s) translate(%(xTranslate)s,%(yTranslate)s)" stroke-width="%(strokeWidth)s" fill="none">
       <!-- hidden lines -->
       <g stroke="rgb(%(hiddenColor)s)" fill="none" stroke-dasharray="%(strokeWidth)s,%(strokeWidth)s" >
%(hiddenContent)s
       </g>

       <!-- solid lines -->
       <g stroke="rgb(%(strokeColor)s)" fill="none">
%(visibleContent)s
       </g>
    </g>
    %(axesIndicator)s
</svg>
"""


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available CadQuery tools."""
    return [
        Tool(
            name="render",
            description=(
                "Execute CadQuery Python code and return a rendered image of the 3D model. "
                "The code should use show_object() to output shapes, or assign the final result to 'result'. "
                "Example: result = cq.Workplane('XY').box(1, 2, 3). "
                "Returns PNG by default (works headlessly, no display required). "
                "Use 'view' to specify camera angle, or 'multi_view' to get multiple angles at once."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "view": {
                        "type": "string",
                        "description": "Camera view angle. Options: isometric (default), front, back, top, bottom, left, right, isometric_back",
                        "enum": ["isometric", "front", "back", "top", "bottom", "left", "right", "isometric_back"],
                        "default": "isometric",
                    },
                    "multi_view": {
                        "type": "boolean",
                        "description": "If true, returns multiple images from different angles (isometric, front, top, right). Useful for complex models.",
                        "default": False,
                    },
                    "format": {
                        "type": "string",
                        "description": "Image format: png (default) or svg",
                        "enum": ["png", "svg"],
                        "default": "png",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 800)",
                        "default": 800,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 600)",
                        "default": 600,
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Whether to show hidden lines (default: true)",
                        "default": True,
                    },
                    "show_axes": {
                        "type": "boolean",
                        "description": "Whether to show the dynamic coordinate triad axes indicator (default: true)",
                        "default": True,
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="inspect",
            description=(
                "Execute CadQuery code and return geometry information about the resulting shape, "
                "including bounding box dimensions, volume, surface area, and center of mass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="get_parameters",
            description=(
                "Parse CadQuery code and extract the parameters (variables) that can be customized. "
                "Returns parameter names, types, and default values."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to parse",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="export",
            description=(
                "Execute CadQuery code and export the result to a file. "
                "Supported formats: STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename (format determined by extension)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format (optional, inferred from filename if not provided)",
                        "enum": ["STEP", "STL", "SVG", "DXF", "AMF", "3MF", "VRML", "BREP"],
                    },
                },
                "required": ["code", "filename"],
            },
        ),
    ]


def _extract_shape(build_result, env):
    """Extract the shape from a build result or environment."""
    # First try to get from show_object() calls
    if build_result.first_result is not None:
        return build_result.first_result.shape

    # Fall back to 'result' variable in environment
    if "result" in env:
        return env["result"]

    return None


def _execute_script(code: str):
    """Execute CadQuery code via CQGI with stdout redirected to stderr.

    Redirecting stdout prevents user print() statements or library output from
    corrupting the JSON-RPC stream over stdio.
    """
    with redirect_stdout(sys.stderr):
        model = cqgi.parse(code)
        return model.build()


def _generate_dynamic_axes(vx: gp_Dir, vy: gp_Dir, width: float, height: float, axis_len: float = 28.0) -> str:
    """Generate a dynamic SVG coordinate triad that matches the camera's projection."""
    x0 = 45.0
    y0 = height - 40.0

    axes = [
        ("X", (1.0, 0.0, 0.0), "#c62828", "arrow-red"),
        ("Y", (0.0, 1.0, 0.0), "#2e7d32", "arrow-green"),
        ("Z", (0.0, 0.0, 1.0), "#1565c0", "arrow-blue"),
    ]

    elements = [
        '<g id="coordinate-triad" font-family="sans-serif" font-size="11px" font-weight="bold">',
        f'  <circle cx="{x0}" cy="{y0}" r="2.5" fill="#424242" />',
    ]

    for label, vec, color, marker in axes:
        # Project axis unit vector onto screen Right (vx) and screen Up (vy)
        dx = vec[0] * vx.X() + vec[1] * vx.Y() + vec[2] * vx.Z()
        dy = vec[0] * vy.X() + vec[1] * vy.Y() + vec[2] * vy.Z()

        proj_len = math.sqrt(dx * dx + dy * dy)
        if proj_len < 0.15:
            # Axis points nearly directly in or out of the screen (along line of sight)
            elements.append(f'  <circle cx="{x0}" cy="{y0}" r="4.5" stroke="{color}" stroke-width="1.5" fill="none" />')
            elements.append(f'  <circle cx="{x0}" cy="{y0}" r="1.5" fill="{color}" />')
            elements.append(f'  <text x="{x0 - 8}" y="{y0 + 12}" fill="{color}" text-anchor="end">{label}</text>')
        else:
            x2 = x0 + (dx / proj_len) * axis_len
            y2 = y0 - (dy / proj_len) * axis_len  # Inverted because SVG screen Y is downward

            tx = x0 + (dx / proj_len) * (axis_len + 14)
            ty = y0 - (dy / proj_len) * (axis_len + 14) + 4

            elements.append(f'  <line x1="{x0}" y1="{y0}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2" marker-end="url(#{marker})" />')
            elements.append(f'  <text x="{tx:.1f}" y="{ty:.1f}" fill="{color}" text-anchor="middle">{label}</text>')

    elements.append("</g>")
    return "\n".join(elements)


def _compute_camera_cs(
    projection_dir: tuple[float, float, float],
    up_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> gp_Ax2:
    """Compute camera coordinate system (gp_Ax2) with an explicit Z-up direction."""
    nx, ny, nz = projection_dir
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_len < 1e-6:
        nx, ny, nz = (-1.0, -1.0, 1.0)
        n_len = math.sqrt(3.0)
    norm = gp_Dir(nx / n_len, ny / n_len, nz / n_len)

    ux, uy, uz = up_dir
    u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
    if u_len < 1e-6:
        ux, uy, uz = (0.0, 0.0, 1.0)
        u_len = 1.0
    up = (ux / u_len, uy / u_len, uz / u_len)

    # Cross product: up x norm
    cross = (
        up[1] * norm.Z() - up[2] * norm.Y(),
        up[2] * norm.X() - up[0] * norm.Z(),
        up[0] * norm.Y() - up[1] * norm.X(),
    )
    cross_len = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)

    if cross_len < 1e-4:
        # Camera is looking parallel to Up vector (e.g. top/bottom view)
        if norm.Z() > 0:  # Top view looking down (+Z)
            vx = gp_Dir(1.0, 0.0, 0.0)
        else:             # Bottom view looking up (-Z)
            vx = gp_Dir(1.0, 0.0, 0.0)
    else:
        vx = gp_Dir(cross[0] / cross_len, cross[1] / cross_len, cross[2] / cross_len)

    return gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), norm, vx)


def _render_svg(
    shape,
    view_name: str,
    width: int,
    height: int,
    show_hidden: bool = True,
    show_axes: bool = True,
) -> str:
    """Render a shape to SVG from a specific view angle using Z-up CAD orientation."""
    projection_dir = VIEWS.get(view_name, VIEWS["isometric"])
    cs = _compute_camera_cs(projection_dir, up_dir=(0.0, 0.0, 1.0))

    hlr = HLRBRep_Algo()
    hlr.Add(shape.wrapped)
    projector = HLRAlgo_Projector(cs)
    hlr.Projector(projector)
    hlr.Update()
    hlr.Hide()

    hlr_shapes = HLRBRep_HLRToShape(hlr)
    visible = []
    for getter in (hlr_shapes.VCompound, hlr_shapes.Rg1LineVCompound, hlr_shapes.OutLineVCompound):
        comp = getter()
        if not comp.IsNull():
            visible.append(comp)

    hidden = []
    for getter in (hlr_shapes.HCompound, hlr_shapes.OutLineHCompound):
        comp = getter()
        if not comp.IsNull():
            hidden.append(comp)

    for el in visible + hidden:
        BRepLib.BuildCurves3d_s(el, TOLERANCE)

    visible_cq = list(map(Shape, visible))
    hidden_cq = list(map(Shape, hidden))
    hidden_paths, visible_paths = getPaths(visible_cq, hidden_cq)

    all_shapes = hidden_cq + visible_cq
    if not all_shapes:
        bb_xlen = bb_ylen = 1.0
        bb_xmin = bb_xmax = bb_ymin = bb_ymax = 0.0
    else:
        bb = Compound.makeCompound(all_shapes).BoundingBox()
        bb_xlen = max(bb.xlen, 1e-4)
        bb_ylen = max(bb.ylen, 1e-4)
        bb_xmin, bb_xmax = bb.xmin, bb.xmax
        bb_ymin, bb_ymax = bb.ymin, bb.ymax

    bb_scale = 0.75
    unit_scale = min((width / bb_xlen) * bb_scale, (height / bb_ylen) * bb_scale)

    # Center model on canvas
    x_translate = (width / 2.0) / unit_scale - (bb_xmin + bb_xmax) / 2.0
    y_translate = -(height / 2.0) / unit_scale - (bb_ymin + bb_ymax) / 2.0

    stroke_width = 1.0 / unit_scale
    hidden_content = "".join(PATHTEMPLATE % p for p in hidden_paths) if show_hidden else ""
    visible_content = "".join(PATHTEMPLATE % p for p in visible_paths)

    axes_indicator = (
        _generate_dynamic_axes(cs.XDirection(), cs.YDirection(), float(width), float(height))
        if show_axes
        else ""
    )

    return SVG_TEMPLATE % {
        "unitScale": str(unit_scale),
        "strokeWidth": str(stroke_width),
        "strokeColor": "0,0,0",
        "hiddenColor": "160,160,160",
        "hiddenContent": hidden_content,
        "visibleContent": visible_content,
        "xTranslate": str(x_translate),
        "yTranslate": str(y_translate),
        "width": str(width),
        "height": str(height),
        "axesIndicator": axes_indicator,
    }


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""

    if name == "render":
        return await _handle_render(arguments)
    elif name == "inspect":
        return await _handle_inspect(arguments)
    elif name == "get_parameters":
        return await _handle_get_parameters(arguments)
    elif name == "export":
        return await _handle_export(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def _encode_image(svg_content: str, image_format: str, width: int, height: int) -> ImageContent:
    """Encode SVG content into the requested ImageContent format (PNG or SVG)."""
    if image_format == "png":
        try:
            import resvg_py
            png_bytes = resvg_py.svg_to_bytes(
                svg_string=svg_content,
                background="#ffffff",
                width=width,
                height=height,
            )
            png_data = base64.standard_b64encode(png_bytes).decode("ascii")
            return ImageContent(type="image", data=png_data, mimeType="image/png")
        except ImportError:
            # Fall back to SVG if resvg_py is not available
            pass

    svg_data = base64.standard_b64encode(svg_content.encode("utf-8")).decode("ascii")
    return ImageContent(type="image", data=svg_data, mimeType="image/svg+xml")


async def _handle_render(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and return rendered image(s)."""
    code = arguments["code"]
    view = arguments.get("view", "isometric")
    multi_view = arguments.get("multi_view", False)
    image_format = arguments.get("format", "png").lower()
    width = arguments.get("width", 800)
    height = arguments.get("height", 600)
    show_hidden = arguments.get("show_hidden", True)
    show_axes = arguments.get("show_axes", True)

    try:
        # Parse and execute the script using CQGI
        result = _execute_script(code)

        if result.exception:
            return [TextContent(
                type="text",
                text=f"Execution error:\n{traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__)}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

        # Get the underlying Shape object if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        if multi_view:
            # Return multiple views for complex models
            views_to_render = ["isometric", "front", "top", "right"]
            results = []

            for view_name in views_to_render:
                svg_content = _render_svg(shape, view_name, width, height, show_hidden, show_axes)
                results.append(_encode_image(svg_content, image_format, width, height))

            # Add a text description of the views
            results.insert(0, TextContent(
                type="text",
                text=f"Rendered {len(views_to_render)} views: {', '.join(views_to_render)}"
            ))
            return results
        else:
            # Single view
            svg_content = _render_svg(shape, view, width, height, show_hidden, show_axes)
            return [_encode_image(svg_content, image_format, width, height)]

    except SyntaxError as e:
        return [TextContent(type="text", text=f"Syntax error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_inspect(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and return geometry information."""
    code = arguments["code"]

    try:
        result = _execute_script(code)

        if result.exception:
            return [TextContent(
                type="text",
                text=f"Execution error: {result.exception}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

        # Get the underlying Shape object if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        # Gather geometry information
        bb = shape.BoundingBox()
        info_lines = [
            "Geometry Information:",
            f"  Bounding Box:",
            f"    X: {bb.xmin:.4f} to {bb.xmax:.4f} (size: {bb.xlen:.4f})",
            f"    Y: {bb.ymin:.4f} to {bb.ymax:.4f} (size: {bb.ylen:.4f})",
            f"    Z: {bb.zmin:.4f} to {bb.zmax:.4f} (size: {bb.zlen:.4f})",
        ]

        # Try to get volume (only works for solids)
        try:
            volume = shape.Volume()
            info_lines.append(f"  Volume: {volume:.4f}")
        except Exception:
            pass

        # Try to get surface area
        try:
            area = shape.Area()
            info_lines.append(f"  Surface Area: {area:.4f}")
        except Exception:
            pass

        # Try to get center of mass
        try:
            com = shape.Center()
            info_lines.append(f"  Center of Mass: ({com.x:.4f}, {com.y:.4f}, {com.z:.4f})")
        except Exception:
            pass

        # Count topological entities
        try:
            info_lines.append(f"  Topology:")
            info_lines.append(f"    Solids: {len(shape.Solids())}")
            info_lines.append(f"    Faces: {len(shape.Faces())}")
            info_lines.append(f"    Edges: {len(shape.Edges())}")
            info_lines.append(f"    Vertices: {len(shape.Vertices())}")
        except Exception:
            pass

        info_lines.append(f"  Build Time: {result.buildTime:.4f}s")

        return [TextContent(type="text", text="\n".join(info_lines))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_get_parameters(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Parse CadQuery code and extract parameters."""
    code = arguments["code"]

    try:
        with redirect_stdout(sys.stderr):
            model = cqgi.parse(code)
        params = model.metadata.parameters

        if not params:
            return [TextContent(type="text", text="No parameters found in the script.")]

        lines = ["Parameters found:"]
        for name, param in params.items():
            type_name = param.varType.__name__ if param.varType else "unknown"
            lines.append(f"  {name}: {type_name} = {param.default_value}")
            if param.desc:
                lines.append(f"    Description: {param.desc}")
            if param.valid_values:
                lines.append(f"    Valid values: {param.valid_values}")

        return [TextContent(type="text", text="\n".join(lines))]

    except SyntaxError as e:
        return [TextContent(type="text", text=f"Syntax error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_export(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Execute CadQuery code and export to file."""
    code = arguments["code"]
    filename = arguments["filename"]
    export_format = arguments.get("format")

    try:
        result = _execute_script(code)

        if result.exception:
            return [TextContent(
                type="text",
                text=f"Execution error: {result.exception}"
            )]

        shape = _extract_shape(result, result.env)

        if shape is None:
            return [TextContent(
                type="text",
                text="No shape produced. Use show_object(shape) or assign to 'result' variable."
            )]

        # Get the underlying Shape if it's a Workplane
        if hasattr(shape, "val"):
            shape = shape.val()

        # Export
        with redirect_stdout(sys.stderr):
            export(shape, filename, exportType=export_format)

        return [TextContent(type="text", text=f"Exported to: {filename}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


import sys
import anyio
import json

class InterceptingStdin:
    """Wraps stdin to intercept and rewrite Antigravity's non-standard JSON-RPC probes."""
    def __init__(self, original_stdin):
        self.original_stdin = original_stdin
        self._iter = None

    def __aiter__(self):
        self._iter = self.original_stdin.__aiter__()
        return self

    async def __anext__(self):
        if self._iter is None:
            self._iter = self.original_stdin.__aiter__()
        while True:
            line = await self._iter.__anext__()
            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get("method") == "server/discover":
                    # Antigravity probe: manual response to stdout to avoid SDK pre-init crash
                    if "id" in data:
                        response = {
                            "jsonrpc": "2.0",
                            "id": data["id"],
                            "result": {"status": "ok"}
                        }
                        sys.stdout.write(json.dumps(response) + "\n")
                        sys.stdout.flush()
                    continue
            except Exception:
                pass
            return line

async def main():
    """Run the MCP server."""
    original_stdin = anyio.wrap_file(sys.stdin)
    patched_stdin = InterceptingStdin(original_stdin)
    
    async with stdio_server(stdin=patched_stdin) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run():
    """Entry point for the cadquery-mcp command."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
