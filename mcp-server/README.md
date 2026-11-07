# CadQuery MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that
lets AI assistants execute CadQuery scripts and render, inspect, cross-section,
reverse-engineer STL meshes, and export 3D CAD models. Run it with the
`cadquery-mcp` entry point.

For command-line workflows, use the existing
[cq-cli](https://github.com/CadQuery/cq-cli) project.

## Features

- **render** - Execute CadQuery code and produce shaded PNGs, 2x2 contact sheets, SVG line drawings, or 2D cross-sections
  - Offscreen PNG rendering preserves assembly colors and part locations
  - Wires and edges render as lines alongside shaded parts
  - Multiple camera angles: isometric, isometric_back, front, back, top, bottom, left, right
  - Multi-view mode (isometric, front, top, right) and optional single-image 2x2 `contact_sheet`
  - 2D cross-section slicer (`section_axis`, `section_offset`) with millimeter coordinate grid and bounding dimensions
  - Configurable image dimensions
  - Hidden line rendering for SVG

- **inspect** - Get geometry information and topology audit about a shape
  - Bounding box dimensions, volume, surface area, and center of mass
  - Topology counts (solids, faces, edges, vertices)
  - BRepCheck validity (`Valid (BRepCheck): Yes/No`) and closed solid count (`Closed Solids: X/Y`)
  - Audit warnings for tiny/sliver faces and short edges that may block fillets
  - Optional 2D cross-section slice inspection (`section_axis`, `section_offset`)

- **get_parameters** - Extract customizable parameters from CadQuery scripts

- **export** - Export models to various formats
  - STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP
  - Assembly STEP exports retain names, colors, and locations

- **analyze_stl** - Segment a CAD-exported STL mesh into exact BRep primitive faces (planes, cylinders, cones, spheres, tori), detect coaxial hole/boss axes, and recommend a CadQuery rebuild strategy (`slabs` vs `revolve`)

- **compare_mesh** - Execute CadQuery code and run a 3-axis ray-cast voxel comparison against a reference STL file, reporting volume difference (%) and bounding boxes of the largest excess/missing discrepancy clusters

Shapes, Workplanes with multiple objects, nested assemblies, and multiple
`show_object()` outputs are supported. Failed MCP tool calls set `isError: true`.
All tool schemas reject unrecognized arguments. Empty models return CAD errors.
Python script output from `print()` is sent to the server's stderr.

## License & Attribution

Licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)** — see [LICENSE](../LICENSE) and [LICENSE-MIT](../LICENSE-MIT) for details.

* Original CadQuery MCP Server and `cadquery-contrib` codebase by the **CadQuery Development Team** and **Dave Cowden** ([`CadQuery/cadquery-contrib`](https://github.com/CadQuery/cadquery-contrib)).
* Topology audit warnings, boolean volume guard (`check_boolean_volume`), 2D cross-section grid rendering, and STL BRep primitive analyzer / 3-axis ray-cast voxel comparator (`stl_analysis.py`, `analyze_stl`, `compare_mesh`) adapted from [`CNCKitchen/ScanRuler`](https://github.com/CNCKitchen/ScanRuler).

## Contributing

Contributions are welcome! Please see the [cadquery-contrib](https://github.com/CadQuery/cadquery-contrib) repository for guidelines.
