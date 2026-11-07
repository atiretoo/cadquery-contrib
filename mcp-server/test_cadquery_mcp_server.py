"""
Tests for the CadQuery MCP server and CAD operations.

These tests verify that:
- The core module executes CadQuery scripts correctly
- SVG rendering works correctly (headless)
- Geometry inspection returns correct values, BRep validity, and topology audits
- Boolean volume guard validates boolean identities
- Multi-view rendering and 2x2 PNG contact sheets work
- 2D cross-section slicing and grid rendering work
- STL BRep segmentation (analyze_stl) and 3-axis voxel comparison (compare_mesh) work
- Parameter extraction works
- Export functionality works
- Error handling is correct
- The MCP server dispatches to core correctly

Run with: pytest test_cadquery_mcp_server.py -v
"""

import pytest
import asyncio
import base64
import struct
import tempfile
import os
from types import SimpleNamespace


# ============================================================================
# CAD operation tests
# ============================================================================

class TestCore:
    """Test CAD operations independently of MCP responses."""

    def test_import(self):
        """Test that the core module can be imported."""
        import cadquery_core
        assert cadquery_core.VIEWS is not None
        assert len(cadquery_core.VIEWS) == 8

    def test_render_simple_box(self):
        """Test SVG rendering of a simple box."""
        from cadquery_core import handle_render

        result = handle_render(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)"
        )

        assert result.error is None
        assert len(result.images) == 1
        assert result.mime_type == "image/svg+xml"
        assert b"<svg" in result.images[0]
        assert b"</svg>" in result.images[0]
        assert b"<path" in result.images[0]

    def test_render_with_show_object(self):
        """Test SVG rendering using show_object() instead of result variable."""
        from cadquery_core import handle_render

        code = "import cadquery as cq\nbox = cq.Workplane('XY').box(5, 5, 5)\nshow_object(box)"
        result = handle_render(code)

        assert result.error is None
        assert len(result.images) == 1
        assert b"<svg" in result.images[0]

    def test_render_no_shape_error(self):
        """Test that render returns an error when no shape is produced."""
        from cadquery_core import handle_render

        result = handle_render("x = 1 + 1")
        assert result.error is not None
        assert "No shape produced" in result.error

    def test_render_syntax_error(self):
        """Test that render handles syntax errors gracefully."""
        from cadquery_core import handle_render

        result = handle_render("this is not valid python!!!")
        assert result.error is not None
        assert "Syntax error" in result.error

    def test_render_execution_error(self):
        """Test that render handles execution errors gracefully."""
        from cadquery_core import handle_render

        result = handle_render("raise ValueError('test error')")
        assert result.error is not None
        assert "Execution error" in result.error

    def test_render_front_view(self):
        """Test rendering from front view."""
        from cadquery_core import handle_render

        result = handle_render(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)",
            view="front",
        )
        assert result.error is None
        assert len(result.images) == 1
        assert result.view_names == ["front"]

    def test_render_all_views(self):
        """Test that all standard views render successfully."""
        from cadquery_core import handle_render, VIEWS

        code = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)"
        for view_name in VIEWS.keys():
            result = handle_render(code, view=view_name)
            assert result.error is None, f"View '{view_name}' failed: {result.error}"
            assert b"<svg" in result.images[0]

    def test_render_multi_view(self):
        """Test that multi_view returns 4 SVGs."""
        from cadquery_core import handle_render

        result = handle_render(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)",
            multi_view=True,
        )

        assert result.error is None
        assert len(result.images) == 4
        assert len(result.view_names) == 4
        assert "isometric" in result.view_names
        assert "front" in result.view_names
        for svg in result.images:
            assert b"<svg" in svg

    def test_render_dimensions(self):
        """Test that width/height parameters are accepted."""
        from cadquery_core import handle_render

        result = handle_render(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)",
            width=400,
            height=300,
        )
        assert result.error is None
        assert b"<svg" in result.images[0]

    def test_render_show_hidden(self):
        """Test that show_hidden option affects output."""
        from cadquery_core import handle_render

        code = "import cadquery as cq\nresult = cq.Workplane('XY').box(20, 20, 10).faces('>Z').workplane().hole(5)"

        result_with = handle_render(code, show_hidden=True)
        result_without = handle_render(code, show_hidden=False)

        assert result_with.error is None
        assert result_without.error is None
        # SVG with hidden lines should have more content
        assert len(result_with.images[0]) > len(result_without.images[0])

    def test_render_empty_code(self):
        """Test handling of empty code."""
        from cadquery_core import handle_render

        result = handle_render("")
        assert result.error is not None
        assert "No shape produced" in result.error

    def test_render_whitespace_code(self):
        """Test handling of whitespace-only code."""
        from cadquery_core import handle_render

        result = handle_render("   \n\n   ")
        assert result.error is not None
        assert "No shape produced" in result.error

    def test_inspect_box(self):
        """Test geometry inspection of a simple box."""
        from cadquery_core import handle_inspect

        result = handle_inspect(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)"
        )

        assert result.error is None
        assert "Bounding Box" in result.text
        assert "10.0000" in result.text
        assert "20.0000" in result.text
        assert "30.0000" in result.text
        assert "Volume" in result.text
        assert "6000" in result.text

    def test_inspect_topology(self):
        """Test that topology information is returned."""
        from cadquery_core import handle_inspect

        result = handle_inspect(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)"
        )

        assert "Topology" in result.text
        assert "Solids: 1" in result.text
        assert "Faces: 6" in result.text
        assert "Edges: 12" in result.text
        assert "Vertices: 8" in result.text
        assert "Valid (BRepCheck): Yes" in result.text
        assert "Closed Solids: 1/1" in result.text
        assert "Audit Warnings: None" in result.text

    def test_inspect_with_show_object(self):
        """Test inspect works with show_object() syntax."""
        from cadquery_core import handle_inspect

        code = "import cadquery as cq\nbox = cq.Workplane('XY').box(5, 10, 15)\nshow_object(box)"
        result = handle_inspect(code)

        assert result.error is None
        assert "5.0000" in result.text
        assert "10.0000" in result.text
        assert "15.0000" in result.text

    def test_get_parameters(self):
        """Test that get_parameters extracts script parameters."""
        from cadquery_core import handle_get_parameters

        code = 'height = 10.0\nwidth = 20.0\nname = "test"\nenabled = True\n\nimport cadquery as cq\nresult = cq.Workplane(\'XY\').box(width, height, 5)'
        result = handle_get_parameters(code)

        assert result.error is None
        assert "height" in result.text
        assert "width" in result.text
        assert "name" in result.text
        assert "enabled" in result.text
        assert "10.0" in result.text
        assert "20.0" in result.text

    def test_get_parameters_none(self):
        """Test get_parameters when no parameters are found."""
        from cadquery_core import handle_get_parameters

        result = handle_get_parameters(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)"
        )

        assert result.error is None
        assert "No parameters found" in result.text

    def test_export_step(self):
        """Test STEP export functionality."""
        from cadquery_core import handle_export

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            filename = f.name

        try:
            result = handle_export(
                "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)",
                filename,
            )

            assert result.error is None
            assert result.filename == filename
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_export_stl(self):
        """Test STL export functionality."""
        from cadquery_core import handle_export

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            filename = f.name

        try:
            result = handle_export(
                "import cadquery as cq\nresult = cq.Workplane('XY').box(5, 5, 5)",
                filename,
            )

            assert result.error is None
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_export_no_shape(self):
        """Test that export returns an error when no shape is produced."""
        from cadquery_core import handle_export

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            filename = f.name

        try:
            result = handle_export("x = 1", filename)
            assert result.error is not None
            assert "No shape produced" in result.error
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_views_dictionary(self):
        """Test that VIEWS dictionary is properly defined."""
        from cadquery_core import VIEWS

        expected = ["isometric", "front", "back", "top", "bottom", "left", "right", "isometric_back"]
        for view in expected:
            assert view in VIEWS
            assert len(VIEWS[view]) == 3

    def test_complex_model_render(self):
        """Test rendering a more complex model."""
        from cadquery_core import handle_render

        code = """
import cadquery as cq
result = (
    cq.Workplane('XY')
    .box(20, 20, 10)
    .faces('>Z')
    .workplane()
    .hole(5)
)
"""
        result = handle_render(code)
        assert result.error is None
        assert b"<svg" in result.images[0]


# ============================================================================
# MCP server tests
# ============================================================================

class TestMCPServer:
    """Test the MCP server wrapper layer."""

    def test_import(self):
        """Test that the MCP server module can be imported."""
        import cadquery_mcp_server
        assert cadquery_mcp_server.server is not None

    def test_list_tools(self):
        """Test that list_tools returns the expected tools."""
        from cadquery_mcp_server import list_tools

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]

        assert "render" in tool_names
        assert "inspect" in tool_names
        assert "get_parameters" in tool_names
        assert "export" in tool_names
        assert "analyze_stl" in tool_names
        assert "compare_mesh" in tool_names

    def test_render_returns_image_content(self):
        """Test that MCP render returns ImageContent with base64 SVG."""
        from cadquery_mcp_server import _render_tool

        result = asyncio.run(_render_tool({
            "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)",
            "image_format": "svg",
        }))

        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/svg+xml"

        svg_content = base64.b64decode(result[0].data).decode("utf-8")
        assert "<svg" in svg_content

    def test_render_error_returns_text_content(self):
        """Test that MCP render errors return TextContent."""
        from cadquery_mcp_server import _render_tool

        result = asyncio.run(_render_tool({"code": "x = 1"}))
        assert result.isError is True
        assert result.content[0].type == "text"
        assert "No shape produced" in result.content[0].text

    def test_multi_view_returns_multiple_images(self):
        """Test that multi_view returns text + 4 images."""
        from cadquery_mcp_server import _render_tool

        result = asyncio.run(_render_tool({
            "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)",
            "multi_view": True,
            "image_format": "svg",
        }))

        assert len(result) == 5
        assert result[0].type == "text"
        assert "4 views" in result[0].text
        for i in range(1, 5):
            assert result[i].type == "image"
            assert result[i].mimeType == "image/svg+xml"

    def test_inspect_returns_text_content(self):
        """Test that MCP inspect returns TextContent."""
        from cadquery_mcp_server import _inspect_tool

        result = asyncio.run(_inspect_tool({
            "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)",
        }))

        assert result[0].type == "text"
        assert "Bounding Box" in result[0].text

    def test_call_tool_dispatch(self):
        """Test that call_tool correctly dispatches to handlers."""
        from cadquery_mcp_server import call_tool

        # render
        result = asyncio.run(call_tool("render", {
            "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)",
            "image_format": "svg",
        }))
        assert result[0].type == "image"

        # inspect
        result = asyncio.run(call_tool("inspect", {
            "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)",
        }))
        assert "Bounding Box" in result[0].text

        # unknown
        result = asyncio.run(call_tool("unknown_tool", {}))
        assert result.isError is True
        assert "Unknown tool" in result.content[0].text


@pytest.mark.parametrize("image_format", ["svg", "png"])
@pytest.mark.parametrize("multi_view", [False, True])
def test_renderer_failure_is_a_cad_error(monkeypatch, image_format, multi_view):
    import cadquery_core as core
    from cadquery_mcp_server import _render_tool

    def render(shape, view, *args):
        if not multi_view or view == "front":
            raise RuntimeError("renderer failed")
        return b"partial PNG" if image_format == "png" else "partial SVG"

    monkeypatch.setattr(core, f"render_{image_format}", render)
    arguments = {
        "code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)",
        "image_format": image_format,
        "multi_view": multi_view,
    }
    result = core.handle_render(**arguments)
    assert result.error == "Render error: RuntimeError: renderer failed"
    assert result.images == []
    assert result.mime_type == ""
    assert result.view_names == []

    # Call the adapter directly: the dispatcher must not be needed to catch this.
    response = asyncio.run(_render_tool(arguments))
    assert response.isError is True
    assert response.content[0].text == result.error


@pytest.mark.parametrize("name", ["render", "inspect", "get_parameters", "export"])
def test_parser_failure_stays_in_cad_layer(monkeypatch, tmp_path, name):
    import cadquery_core as core
    import cadquery_mcp_server as server

    def parse(code):
        raise RuntimeError("parser failed")

    monkeypatch.setattr(core.cqgi, "parse", parse)
    arguments = {"code": "unused"}
    if name == "export":
        arguments["filename"] = str(tmp_path / "unused.step")
    result = getattr(core, f"handle_{name}")(**arguments)
    assert "RuntimeError: parser failed" in result.error
    response = asyncio.run(getattr(server, f"_{name}_tool")(arguments))
    assert response.isError is True
    assert response.content[0].text == result.error
    assert list(tmp_path.iterdir()) == []


def test_inspection_formatting_failure_stays_in_cad_layer(monkeypatch):
    import cadquery_core as core
    from cadquery_mcp_server import _inspect_tool

    # BoundingBox succeeds, but its result cannot be formatted.
    shape = SimpleNamespace(BoundingBox=lambda: object())
    monkeypatch.setattr(core, "_execute_code", lambda code: (shape, None))
    result = core.handle_inspect("unused")
    assert result.error.startswith("Inspect error: AttributeError:")
    response = asyncio.run(_inspect_tool({"code": "unused"}))
    assert response.isError is True
    assert response.content[0].text == result.error


def test_parameter_formatting_failure_stays_in_cad_layer(monkeypatch):
    import cadquery_core as core
    from cadquery_mcp_server import _get_parameters_tool

    model = SimpleNamespace(metadata=SimpleNamespace(parameters={"broken": object()}))
    monkeypatch.setattr(core.cqgi, "parse", lambda code: model)
    result = core.handle_get_parameters("unused")
    assert result.error.startswith("Parameter error: AttributeError:")
    response = asyncio.run(_get_parameters_tool({"code": "unused"}))
    assert response.isError is True
    assert response.content[0].text == result.error


def test_inspection_omits_incomplete_topology(monkeypatch):
    import cadquery_core as core

    shape = core.cq.Workplane("XY").box(1, 1, 1).val()

    def fail_faces():
        raise RuntimeError("topology unavailable")

    monkeypatch.setattr(shape, "Faces", fail_faces)
    monkeypatch.setattr(core, "_execute_code", lambda code: (shape, None))
    result = core.handle_inspect("unused")
    assert result.error is None
    assert "Bounding Box:" in result.text
    assert "Volume:" in result.text
    assert "Topology:" not in result.text
    assert "Solids:" not in result.text


def test_render_png_axes_option():
    """Verify render_png works with show_axes True and False."""
    from cadquery_core import handle_render

    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(5, 5, 5)"
    res_with_axes = handle_render(code, image_format="png", show_axes=True)
    assert res_with_axes.error is None
    assert len(res_with_axes.images) == 1
    assert res_with_axes.mime_type == "image/png"

    res_without_axes = handle_render(code, image_format="png", show_axes=False)
    assert res_without_axes.error is None
    assert len(res_without_axes.images) == 1
    assert res_without_axes.mime_type == "image/png"


def test_render_png_edges_option():
    """Verify render_png works with show_edges True and False."""
    from cadquery_core import handle_render

    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(5, 5, 5)"
    res_with_edges = handle_render(code, image_format="png", show_edges=True)
    assert res_with_edges.error is None
    assert len(res_with_edges.images) == 1
    assert res_with_edges.mime_type == "image/png"

    res_without_edges = handle_render(code, image_format="png", show_edges=False)
    assert res_without_edges.error is None
    assert len(res_without_edges.images) == 1
    assert res_without_edges.mime_type == "image/png"


def test_render_svg_background_card_option():
    """Verify render_svg inserts background card when requested."""
    from cadquery_core import handle_render

    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(5, 5, 5)"
    res_with_card = handle_render(code, image_format="svg", background_card=True)
    assert res_with_card.error is None
    svg_str = res_with_card.images[0].decode("utf-8")
    assert '<rect width="100%" height="100%" fill="#ffffff"' in svg_str

    res_without_card = handle_render(code, image_format="svg", background_card=False)
    assert res_without_card.error is None
    svg_no_card = res_without_card.images[0].decode("utf-8")
    assert '<rect width="100%" height="100%" fill="#ffffff"' not in svg_no_card


def test_intercepting_stdin_server_discover():
    """Verify InterceptingStdin captures server/discover and passes through normal input."""
    import io
    import json
    import anyio
    from cadquery_mcp_server import InterceptingStdin

    raw = io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"server/discover"}\n{"jsonrpc":"2.0","id":2,"method":"initialize"}\n')
    wrapped = anyio.wrap_file(raw)
    interceptor = InterceptingStdin(wrapped)

    # First line should be consumed and interceptor should continue to next line
    async def read_next():
        return await interceptor.readline()

    line = asyncio.run(read_next())
    assert json.loads(line)["method"] == "initialize"


# ============================================================================
# ScanRuler Integration Tests (Audit, Boolean Guard, Contact Sheet, Section, STL)
# ============================================================================

def test_inspect_topology_audit_warnings():
    """Verify handle_inspect detects tiny/sliver faces and short edges."""
    from cadquery_core import handle_inspect

    # A tiny box (0.02 x 10 x 10) has 0.02 mm edges (< 0.1 mm) and narrow faces
    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(0.02, 10, 10)"
    result = handle_inspect(code)
    assert result.error is None
    assert "Valid (BRepCheck): Yes" in result.text
    assert "Closed Solids: 1/1" in result.text
    assert "Audit Warnings:" in result.text
    assert "Short edge" in result.text


def test_check_boolean_volume_guard():
    """Verify check_boolean_volume (ported from booleanGuard.ts) detects lost/gained volume."""
    from cadquery_core import check_boolean_volume, volume_guard

    assert volume_guard is check_boolean_volume

    # Valid union / subtract / intersect / join / cut
    assert check_boolean_volume("union", 100.0, 50.0, 130.0, common_vol=20.0) is None
    assert check_boolean_volume("subtract", 100.0, 30.0, 80.0, common_vol=20.0) is None
    assert check_boolean_volume("intersect", 100.0, 30.0, 20.0) is None
    assert check_boolean_volume("join", result_vol=150.0, parts=[100.0, 50.0]) is None
    assert check_boolean_volume("cut", body_vol=100.0, result_vol=75.0) is None

    # Refusals
    assert "lost volume" in check_boolean_volume("union", 100.0, 50.0, 80.0)
    assert "made volume" in check_boolean_volume("union", 100.0, 50.0, 160.0)
    assert "gained volume" in check_boolean_volume("subtract", 100.0, 30.0, 110.0)
    assert "took more than the tool holds" in check_boolean_volume("subtract", 100.0, 30.0, 60.0)
    assert "more than the smaller holds" in check_boolean_volume("intersect", 100.0, 30.0, 35.0)


def test_render_multi_view_contact_sheet():
    """Verify contact_sheet=True composites 4 views into a single 2x2 PNG image."""
    from cadquery_core import handle_render
    from cadquery_mcp_server import _render_tool

    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)"
    res = handle_render(
        code,
        image_format="png",
        multi_view=True,
        contact_sheet=True,
        width=200,
        height=150,
    )
    assert res.error is None, res.error
    assert res.mime_type == "image/png"
    assert res.view_names == ["contact_sheet"]
    assert len(res.images) == 1

    data = res.images[0]
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (400, 300)

    # Also test via MCP _render_tool
    mcp_res = asyncio.run(_render_tool({
        "code": code,
        "image_format": "png",
        "multi_view": True,
        "contact_sheet": True,
        "width": 200,
        "height": 150,
    }))
    assert len(mcp_res) == 1
    assert mcp_res[0].type == "image"
    assert mcp_res[0].mimeType == "image/png"


def test_cross_section_render_and_inspect():
    """Verify 2D cross-section slicing in handle_render (SVG & PNG) and handle_inspect."""
    from cadquery_core import handle_render, handle_inspect

    code = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').box(20, 30, 10).faces('>Z').workplane().hole(8)"
    )

    # 1. SVG cross-section
    res_svg = handle_render(code, image_format="svg", section_axis="Z", section_offset=0.0)
    assert res_svg.error is None, res_svg.error
    assert res_svg.mime_type == "image/svg+xml"
    assert res_svg.view_names == ["section_z_0"]
    svg_text = res_svg.images[0].decode("utf-8")
    assert "Cross-Section Z = 0 mm" in svg_text
    assert 'class="grid"' in svg_text
    assert 'class="section-edges"' in svg_text

    # 2. PNG cross-section
    res_png = handle_render(code, image_format="png", section_axis="Z", section_offset=0.0, width=320, height=240)
    assert res_png.error is None, res_png.error
    assert res_png.mime_type == "image/png"
    assert res_png.images[0].startswith(b"\x89PNG\r\n\x1a\n")

    # 3. Inspect cross-section
    insp = handle_inspect(code, section_axis="Z", section_offset=0.0)
    assert insp.error is None, insp.error
    assert "Cross-Section (Z = 0.0000):" in insp.text
    assert "X: -10.0000 to 10.0000 (size: 20.0000)" in insp.text
    assert "Y: -15.0000 to 15.0000 (size: 30.0000)" in insp.text

    # 4. Out-of-bounds plane returns clean error
    miss = handle_render(code, image_format="svg", section_axis="Z", section_offset=50.0)
    assert miss.error is not None
    assert "does not intersect" in miss.error


def test_analyze_stl_and_compare_mesh(tmp_path):
    """Verify analyze_stl and compare_mesh tools segment BRep faces and detect voxel discrepancies."""
    from cadquery_core import handle_export, handle_analyze_stl, handle_compare_mesh
    from cadquery_mcp_server import call_tool

    ref_code = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').box(20, 20, 10).faces('>Z').workplane().hole(8)"
    )
    ref_stl = tmp_path / "reference.stl"
    exp = handle_export(ref_code, str(ref_stl))
    assert exp.error is None, exp.error

    # 1. Test analyze_stl
    analysis = handle_analyze_stl(str(ref_stl))
    assert analysis.error is None, analysis.error
    assert "Clean CAD export" in analysis.text
    assert "2.5D along z:" in analysis.text
    assert "cyl  hole R" in analysis.text
    assert "recommend:" in analysis.text

    mcp_analysis = asyncio.run(call_tool("analyze_stl", {"stl_path": str(ref_stl)}))
    assert not isinstance(mcp_analysis, SimpleNamespace)
    assert mcp_analysis[0].type == "text"
    assert "Clean CAD export" in mcp_analysis[0].text

    # 2. Test compare_mesh with candidate missing the hole (has EXTRA material in the hole)
    candidate_code = "import cadquery as cq\nresult = cq.Workplane('XY').box(20, 20, 10)"
    comp = handle_compare_mesh(candidate_code, str(ref_stl), voxel_size=0.5)
    assert comp.error is None, comp.error
    assert "volume: mesh" in comp.text
    assert "EXTRA (model has, mesh not):" in comp.text
    assert "box" in comp.text

    mcp_comp = asyncio.run(call_tool("compare_mesh", {
        "code": candidate_code,
        "reference_stl_path": str(ref_stl),
        "voxel_size": 0.5,
    }))
    assert mcp_comp[0].type == "text"
    assert "EXTRA (model has, mesh not):" in mcp_comp[0].text
