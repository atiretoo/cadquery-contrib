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
CadQuery MCP (Model Context Protocol) Server.

This module provides an MCP server that allows AI assistants like Claude
to execute CadQuery scripts, receive rendered images or 2D cross-sections of 3D models,
audit BRep topology, and reverse-engineer/compare STL meshes.

Usage:
    Run as a standalone server:
        python -m cadquery_mcp_server

    Or use the entry point:
        cadquery-mcp

Project configuration in Claude Code (.mcp.json in the project root):
    {
        "mcpServers": {
            "cadquery": {
                "type": "stdio",
                "command": "/absolute/path/to/your/environment/bin/python",
                "args": ["-u", "/absolute/path/to/mcp-server/cadquery_mcp_server.py"]
            }
        }
    }
"""

import asyncio
import base64
import json
import sys
from typing import Any

import anyio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, CallToolResult

from cadquery_core import (
    VIEWS,
    handle_render,
    handle_inspect,
    handle_get_parameters,
    handle_export,
    handle_analyze_stl,
    handle_compare_mesh,
)


server = Server("cadquery")
ToolResponse = list[TextContent | ImageContent] | CallToolResult


def _error_response(message: str) -> CallToolResult:
    """Report tool failures through MCP's error flag, not just their text."""
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])


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
                "Returns shaded PNG by default, preserving assembly colors and locations. "
                "Use image_format='svg' for a monochrome line drawing. "
                "Use 'view' to specify camera angle, or 'multi_view' (optionally with 'contact_sheet'=true) "
                "to get multiple angles at once. "
                "Use 'section_axis' ('X', 'Y', or 'Z') and 'section_offset' (mm) to slice the model and "
                "render a 2D cross-section with a millimeter coordinate grid and bounding dimensions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "image_format": {
                        "type": "string",
                        "enum": ["png", "svg"],
                        "default": "png",
                        "description": "PNG for viewable colored images (default); SVG for line drawings.",
                    },
                    "view": {
                        "type": "string",
                        "description": "Camera view angle. Options: isometric (default), front, back, top, bottom, left, right, isometric_back",
                        "enum": list(VIEWS.keys()),
                        "default": "isometric",
                    },
                    "multi_view": {
                        "type": "boolean",
                        "description": "If true, returns multiple images from different angles (isometric, front, top, right). Useful for complex models.",
                        "default": False,
                    },
                    "contact_sheet": {
                        "type": "boolean",
                        "description": "When multi_view=true and image_format='png', composite the 4 views into a single 2x2 contact sheet PNG with labels (default: false).",
                        "default": False,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 800)",
                        "default": 800,
                        "minimum": 16,
                        "maximum": 4096,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 600)",
                        "default": 600,
                        "minimum": 16,
                        "maximum": 4096,
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Whether to show hidden lines in SVG (default: true); unused for PNG.",
                        "default": True,
                    },
                    "show_axes": {
                        "type": "boolean",
                        "description": "Whether to show the 3D coordinate triad indicator (default: true).",
                        "default": True,
                    },
                    "show_edges": {
                        "type": "boolean",
                        "description": "Whether to draw visible boundary and feature edge outlines on shaded solids in PNG (default: true).",
                        "default": True,
                    },
                    "background_card": {
                        "type": "boolean",
                        "description": "Whether to include an opaque card background in SVG for visibility in dark/light themes (default: true).",
                        "default": True,
                    },
                    "section_axis": {
                        "type": "string",
                        "enum": ["X", "Y", "Z", "x", "y", "z"],
                        "description": "Optional cross-section plane axis ('X', 'Y', or 'Z'). Slices the model at section_offset and renders a 2D cross-section with a millimeter grid.",
                    },
                    "section_offset": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Plane offset in mm along section_axis when slicing a 2D cross-section (default: 0.0).",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="inspect",
            description=(
                "Execute CadQuery code and return geometry information about the resulting shape, "
                "including bounding box dimensions, volume, surface area, center of mass, "
                "BRepCheck validity, closed solid count, topology audit warnings (tiny/sliver faces, short edges), "
                "and optional 2D cross-section metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute",
                    },
                    "section_axis": {
                        "type": "string",
                        "enum": ["X", "Y", "Z", "x", "y", "z"],
                        "description": "Optional cross-section axis ('X', 'Y', or 'Z') to inspect 2D slice dimensions at section_offset.",
                    },
                    "section_offset": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Cross-section offset in mm along section_axis (default: 0.0).",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
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
                "additionalProperties": False,
            },
        ),
        Tool(
            name="export",
            description=(
                "Execute CadQuery code and export the result to a file. "
                "Supported formats: STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP. "
                "Assembly STEP exports preserve part names, colors, and locations; mesh formats flatten geometry."
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
                        "description": "Output filename; ~ expands to the server user's home, relative paths use the server working directory. Returns the absolute path.",
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format (optional, inferred from filename if not provided)",
                        "enum": ["STEP", "STL", "SVG", "DXF", "AMF", "3MF", "VRML", "BREP"],
                    },
                },
                "required": ["code", "filename"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="analyze_stl",
            description=(
                "Segment an STL mesh into exact BRep primitive faces (planes, cylinders, cones, spheres, tori), "
                "classify whether it is a clean CAD export or noisy scan, identify 2.5D build levels and coaxial "
                "hole/boss axes, and recommend a CadQuery rebuild strategy (slabs vs revolve)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "Path to the STL file to analyze",
                    },
                },
                "required": ["stl_path"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="compare_mesh",
            description=(
                "Execute CadQuery code, export it to a temporary STL mesh, and run a 3-axis ray-cast voxel "
                "comparison against a reference STL file. Reports volume difference (%), excess/missing volume (mm^3), "
                "and spatial bounding boxes of the largest discrepancy clusters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "CadQuery Python code to execute and compare",
                    },
                    "reference_stl_path": {
                        "type": "string",
                        "description": "Path to the reference STL file",
                    },
                    "voxel_size": {
                        "type": "number",
                        "default": 0.2,
                        "exclusiveMinimum": 0,
                        "description": "Voxel grid spacing in mm for 3-axis ray-cast occupancy comparison (default: 0.2).",
                    },
                },
                "required": ["code", "reference_stl_path"],
                "additionalProperties": False,
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Handle tool calls."""

    try:
        if name == "render":
            return await _render_tool(arguments)
        elif name == "inspect":
            return await _inspect_tool(arguments)
        elif name == "get_parameters":
            return await _get_parameters_tool(arguments)
        elif name == "export":
            return await _export_tool(arguments)
        elif name == "analyze_stl":
            return await _analyze_stl_tool(arguments)
        elif name == "compare_mesh":
            return await _compare_mesh_tool(arguments)
        return _error_response(f"Unknown tool: {name}")
    except Exception as e:
        return _error_response(f"Error: {type(e).__name__}: {e}")


async def _render_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Execute CadQuery code and return rendered PNG or SVG images."""
    result = handle_render(
        arguments["code"],
        view=arguments.get("view", "isometric"),
        multi_view=arguments.get("multi_view", False),
        contact_sheet=arguments.get("contact_sheet", False),
        width=arguments.get("width", 800),
        height=arguments.get("height", 600),
        show_hidden=arguments.get("show_hidden", True),
        image_format=arguments.get("image_format", "png"),
        show_axes=arguments.get("show_axes", True),
        show_edges=arguments.get("show_edges", True),
        background_card=arguments.get("background_card", True),
        section_axis=arguments.get("section_axis"),
        section_offset=arguments.get("section_offset", 0.0),
    )

    if result.error:
        return _error_response(result.error)

    items: list[TextContent | ImageContent] = []
    if len(result.view_names) > 1:
        items.append(
            TextContent(
                type="text",
                text=f"Rendered {len(result.view_names)} views: {', '.join(result.view_names)}",
            )
        )
    for data in result.images:
        items.append(ImageContent(
            type="image", data=base64.b64encode(data).decode("ascii"),
            mimeType=result.mime_type,
        ))
    return items


async def _inspect_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Execute CadQuery code and return geometry information."""
    result = handle_inspect(
        arguments["code"],
        section_axis=arguments.get("section_axis"),
        section_offset=arguments.get("section_offset", 0.0),
    )

    if result.error:
        return _error_response(result.error)

    return [TextContent(type="text", text=result.text)]


async def _get_parameters_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Parse CadQuery code and extract parameters."""
    result = handle_get_parameters(arguments["code"])

    if result.error:
        return _error_response(result.error)

    return [TextContent(type="text", text=result.text)]


async def _export_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Execute CadQuery code and export to file."""
    result = handle_export(
        arguments["code"],
        arguments["filename"],
        arguments.get("format"),
    )

    if result.error:
        return _error_response(result.error)

    return [TextContent(type="text", text=f"Exported to: {result.filename}")]


async def _analyze_stl_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Analyze an STL file and return BRep segmentation report."""
    result = handle_analyze_stl(arguments["stl_path"])

    if result.error:
        return _error_response(result.error)

    return [TextContent(type="text", text=result.text)]


async def _compare_mesh_tool(arguments: dict[str, Any]) -> ToolResponse:
    """Execute CadQuery code and compare against a reference STL mesh."""
    result = handle_compare_mesh(
        arguments["code"],
        arguments["reference_stl_path"],
        arguments.get("voxel_size", 0.2),
    )

    if result.error:
        return _error_response(result.error)

    return [TextContent(type="text", text=result.text)]


class InterceptingStdin:
    """Wrap anyio AsyncFile to intercept client probes (e.g. Antigravity server/discover)."""
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line

    async def readline(self):
        while True:
            line = await self._wrapped.readline()
            if not line:
                return line
            try:
                data = json.loads(line)
                if data.get("method") == "server/discover":
                    if "id" in data:
                        response = {
                            "jsonrpc": "2.0",
                            "id": data["id"],
                            "result": {"status": "ok"}
                        }
                        sys.stdout.buffer.write((json.dumps(response) + "\n").encode("utf-8"))
                        sys.stdout.buffer.flush()
                    continue
            except Exception:
                pass
            return line


async def main():
    """Run the MCP server."""
    from io import TextIOWrapper
    raw_stdin = TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    original_stdin = anyio.wrap_file(raw_stdin)
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
