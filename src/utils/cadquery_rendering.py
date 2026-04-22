"""
Cross-platform offscreen rendering and export for CadQuery/OCP shapes.

Rendering uses OCP's V3d offscreen viewer.
Requires Xvfb on headless Linux (xvfb-run python ...).
Uses Cocoa_Window on macOS.
"""

import os
import shutil
import sys

import cadquery as cq
from cadquery import exporters

VIEW_PROJECTIONS = {
    'toprightiso': (1, -1, 1),
    'front': (0, 0, 1),
    'back': (0, 0, -1),
    'left': (-1, 0, 0),
    'right': (1, 0, 0),
    'top': (0, 1, 0),
    'bottom': (0, -1, 0),
}


def render_to_png(occ_shape, png_path, proj=(1, -1, 1), width=1024, height=1024):
    """Render an OCP shape to a PNG file from a given projection direction.

    Args:
        occ_shape: A CadQuery Shape or raw OCP TopoDS_Shape.
        png_path: Output PNG file path.
        proj: (Vx, Vy, Vz) camera projection direction tuple.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    from OCP.AIS import AIS_InteractiveContext, AIS_Shape, AIS_Shaded
    from OCP.Aspect import Aspect_DisplayConnection, Aspect_TypeOfLine
    from OCP.OpenGl import OpenGl_GraphicDriver
    from OCP.Prs3d import Prs3d_LineAspect
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
    from OCP.V3d import V3d_Viewer

    wrapped = occ_shape.wrapped if hasattr(occ_shape, "wrapped") else occ_shape

    display_connection = Aspect_DisplayConnection()
    driver = OpenGl_GraphicDriver(display_connection)
    viewer = V3d_Viewer(driver)
    viewer.SetDefaultLights()
    viewer.SetLightOn()

    context = AIS_InteractiveContext(viewer)
    view = viewer.CreateView()

    if sys.platform == "linux":
        from OCP.Xw import Xw_Window
        window = Xw_Window(display_connection, "offscreen", 0, 0, width, height)
    else:
        import AppKit
        AppKit.NSApplication.sharedApplication()
        from OCP.Cocoa import Cocoa_Window
        window = Cocoa_Window("offscreen", 0, 0, width, height)

    view.SetWindow(window)
    if not window.IsMapped():
        window.Map()

    view.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

    from OCP.Graphic3d import Graphic3d_MaterialAspect

    ais_shape = AIS_Shape(wrapped)

    mat = Graphic3d_MaterialAspect()
    mat.SetSpecularColor(Quantity_Color(0.15, 0.15, 0.15, Quantity_TOC_RGB))
    mat.SetShininess(0.3)
    ais_shape.SetMaterial(mat)
    ais_shape.SetColor(Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB))

    drawer = ais_shape.Attributes()
    edge_aspect = Prs3d_LineAspect(
        Quantity_Color(Quantity_NOC_BLACK), Aspect_TypeOfLine.Aspect_TOL_SOLID, 2.0
    )
    drawer.SetFaceBoundaryAspect(edge_aspect)
    drawer.SetFaceBoundaryDraw(True)

    context.Display(ais_shape, AIS_Shaded, 0, True)
    view.SetProj(float(proj[0]), float(proj[1]), float(proj[2]))
    view.FitAll()
    view.Dump(str(png_path))


def export_as_step(result, output_dir):
    """Export CadQuery result as STEP file."""
    try:
        output_path = os.path.join(output_dir, "tmp.step")
        if hasattr(result, 'val'):
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            result.save(output_path)
            print(f"Exported assembly to: {output_path}")
            return
        else:
            shape = result

        exporters.export(shape, output_path, exportType="STEP")
        print(f"Exported shape to: {output_path}")
    except Exception as e:
        print(f"Error exporting STEP file: {e}")
        import traceback
        traceback.print_exc()


def export_as_image(result, output_dir, views=None, width=1024, height=1024):
    """Export CadQuery result as PNG image(s).

    Args:
        result: CadQuery Workplane, Shape, or Assembly.
        output_dir: Directory to write PNG files into.
        views: Optional list of view names from VIEW_PROJECTIONS.
            When None (default), renders a single ``tmp.png`` at the
            default iso projection -- fully backwards compatible.
            When a list, renders one ``tmp_{name}.png`` per view.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of exported PNG file paths.
    """
    try:
        if hasattr(result, 'val'):
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            shape = result.toCompound()
        else:
            shape = result

        if sys.platform == "linux" and not os.environ.get("DISPLAY"):
            if not shutil.which("Xvfb") and not shutil.which("xvfb-run"):
                print("Error: Xvfb is not installed. PNG export requires Xvfb.")
                print("Install it with:")
                print("  sudo apt-get install -y xvfb    # Debian/Ubuntu")
                print("  sudo yum install -y xorg-x11-server-Xvfb  # RHEL/Amazon Linux")
                sys.exit(1)
            print("Error: No DISPLAY set. Xvfb is installed but not running.")
            print("Either run with:  xvfb-run python cadquery_script.py ...")
            print("Or start Xvfb manually:")
            print("  Xvfb :99 -screen 0 1024x768x24 &")
            print("  export DISPLAY=:99")
            sys.exit(1)

        exported = []

        if views is None:
            png_path = os.path.join(output_dir, "tmp.png")
            render_to_png(shape, png_path, width=width, height=height)
            print(f"Exported PNG image to: {png_path}")
            exported.append(png_path)
        else:
            for view_name in views:
                proj = VIEW_PROJECTIONS.get(view_name)
                if proj is None:
                    print(f"Warning: Unknown view '{view_name}', skipping.")
                    continue
                png_path = os.path.join(output_dir, f"tmp_{view_name}.png")
                render_to_png(shape, png_path, proj=proj, width=width, height=height)
                print(f"Exported PNG image to: {png_path}")
                exported.append(png_path)

        return exported

    except Exception as e:
        print(f"Error exporting image: {e}")
        import traceback
        traceback.print_exc()
        return []
