# CadQuery MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that enables AI assistants like Claude to execute CadQuery scripts and render 3D CAD models.

## Features

- **render** - Execute CadQuery code and return rendered images of the 3D model (PNG or SVG)
  - **CAD Z-Up Standard**: Enforces mechanical CAD conventions ($Z$ vertical, $XY$ ground plane) with dynamic RGB coordinate triads. See [Coordinate Systems and Rendering Guide](docs/coordinate_systems_and_rendering.md).
  - Multiple camera angles: isometric, front, back, top, bottom, left, right
  - Multi-view mode for complex models
  - Configurable image dimensions
  - Format selection (PNG by default for inline chat and LLM vision, or SVG)
  - Hidden line rendering

- **inspect** - Get geometry information about a shape
  - Bounding box dimensions
  - Volume and surface area
  - Center of mass
  - Topology counts (solids, faces, edges, vertices)

- **get_parameters** - Extract customizable parameters from CadQuery scripts

- **export** - Export models to various formats
  - STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP

## Installation

### Prerequisites

CadQuery must be installed first. The recommended method is via conda:

```bash
conda install -c conda-forge cadquery
```

### Install from Source

```bash
git clone https://github.com/CadQuery/cadquery-contrib.git
cd cadquery-contrib/mcp-server
pip install .
```

For development (editable install):

```bash
pip install -e .
```

### Run Tests

```bash
pip install pytest
pytest test_cadquery_mcp_server.py -v
```

## Configuration

### Claude Code

Add to your `~/.claude/settings.json`:

```json
{
    "mcpServers": {
        "cadquery": {
            "command": "cadquery-mcp"
        }
    }
}
```

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
    "mcpServers": {
        "cadquery": {
            "command": "cadquery-mcp"
        }
    }
}
```

**Note:** If using conda, you may need to specify the full path:

```json
{
    "mcpServers": {
        "cadquery": {
            "command": "/path/to/conda/envs/yourenv/bin/cadquery-mcp"
        }
    }
}
```

## Usage Examples

Once configured, you can ask Claude to create 3D models:

> "Create a box with a hole through it"

Claude will execute:

```python
import cadquery as cq

result = (
    cq.Workplane('XY')
    .box(20, 20, 10)
    .faces('>Z')
    .workplane()
    .hole(5)
)
```

And return a rendered image of the model (PNG by default).

### Multi-View Rendering

For complex models, request multiple views:

> "Show me this bracket from multiple angles"

The server will return isometric, front, top, and right views.

### Parametric Models

CadQuery scripts can define parameters:

```python
height = 10.0  # Height of the box
width = 20.0   # Width of the box
depth = 5.0    # Depth of the box

import cadquery as cq
result = cq.Workplane('XY').box(width, height, depth)
```

Use the `get_parameters` tool to extract these for modification.

### Exporting Models

Export to STEP for manufacturing or STL for 3D printing:

> "Export this model as a STEP file to ~/models/bracket.step"

## API Reference

### render

Execute CadQuery code and return rendered image(s).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| code | string | required | CadQuery Python code to execute |
| view | string | "isometric" | Camera angle (isometric, front, back, top, bottom, left, right, isometric_back) |
| multi_view | boolean | false | Return multiple views |
| format | string | "png" | Image format: "png" (default, ideal for LLM vision and chat display) or "svg" |
| width | integer | 800 | Image width in pixels |
| height | integer | 600 | Image height in pixels |
| show_hidden | boolean | true | Show hidden lines |

### inspect

Get geometry information about the resulting shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| code | string | required | CadQuery Python code to execute |

### get_parameters

Extract customizable parameters from a script.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| code | string | required | CadQuery Python code to parse |

### export

Export the model to a file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| code | string | required | CadQuery Python code to execute |
| filename | string | required | Output filename |
| format | string | auto | Export format (STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP) |

## Writing CadQuery Scripts for MCP

Scripts should either:

1. Assign the final shape to a variable named `result`:
   ```python
   result = cq.Workplane('XY').box(1, 2, 3)
   ```

2. Use `show_object()` to output shapes:
   ```python
   box = cq.Workplane('XY').box(1, 2, 3)
   show_object(box)
   ```

### Why not `cadquery.vis.show()` or `cadquery.fig.show()`?

You may come across `show()` functions in CadQuery documentation:
- `cadquery.vis.show()` opens an interactive desktop GUI window using VTK. This blocks script execution (`inter.Start()`) until a user manually closes the window, requires local display drivers (failing in headless or remote server environments), and does not return image payloads to the caller.
- `cadquery.fig.show()` starts a local Trame/Vuetify web server and opens a new browser tab for interactive human viewing.

Neither function returns image data over the Model Context Protocol. AI assistants like Claude cannot see desktop windows or browser sessions. The MCP server's `render` tool solves this by headlessly extracting the OpenCASCADE geometry in memory, projecting it to vector lines (SVG), and rasterizing it to PNG via `resvg-py` so the model can inspect and verify the 3D model inline within the chat.

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

## Contributing

Contributions are welcome! Please see the [cadquery-contrib](https://github.com/CadQuery/cadquery-contrib) repository for guidelines.
