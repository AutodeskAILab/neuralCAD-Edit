#!/usr/bin/env python3

import cadquery as cq
from cadquery import exporters
import os
import sys

###
# Function definitions
###

def export_as_step(result, output_dir):
    """Export CadQuery result as STEP file"""
    try:
        output_path = os.path.join(output_dir, "tmp.step")
        # Handle different result types
        if hasattr(result, 'val'):
            # It's a Workplane
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            # It's an Assembly - use assembly export
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


def export_as_image(result, output_dir):
    """Export CadQuery result as an image (SVG or PNG if VTK available)"""
    try:
        # Get the shape from result
        if hasattr(result, 'val'):
            shape = result.val()
        elif isinstance(result, cq.Assembly):
            # For assemblies, try to get combined shape
            shape = result.toCompound()
        else:
            shape = result
        
        # Try PNG export using VTK (requires cadquery with ocp/vtk support)
        png_path = os.path.join(output_dir, "tmp.png")
        try:
            from cadquery.vis import show
            # Use show() with screenshot parameter instead of vis.screenshot()
            show(
                shape,
                screenshot=png_path,
                width=1920,
                height=1080,
                interact=False  # Don't open interactive GUI
            )
            print(f"Exported PNG image to: {png_path}")
            return
        except Exception as e:
            print(f"Error exporting PNG image: {e}")
            pass
        
        # Try using ocp_vscode or jupyter_cadquery for rendering
        try:
            from ocp_tessellate.convert import to_svg
            svg_path = os.path.join(output_dir, "tmp.svg")
            svg_content = to_svg(shape)
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            print(f"Exported SVG image to: {svg_path}")
            return
        except ImportError:
            pass
        
        # Fallback: export SVG using CadQuery's built-in SVG exporter
        svg_path = os.path.join(output_dir, "tmp.svg")
        exporters.export(shape, svg_path, exportType="SVG")
        print(f"Exported SVG image to: {svg_path}")
        
    except Exception as e:
        print(f"Error exporting image: {e}")
        import traceback
        traceback.print_exc()


def parse_cadquery_args(argv, required_args=None):
    """
    Parses command line arguments passed to CadQuery scripts.
    Args will be passed as --passkey value(s)
    Only required args are processed
    """
    if required_args is None:
        required_args = ['--output_dir', '--input_file', '--function_file']
    
    args = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in required_args:
            if i + 1 < len(argv):
                args[arg.lstrip('-')] = argv[i + 1]
                i += 1  # skip next since it's a value
            else:
                print(f"Warning: No value provided for argument {arg}")
        i += 1
    return args


def load_and_execute_function(function_string, args):
    """
    Load a my_cad_function function from a string and execute it.
    
    Args:
        function_string (str): The complete function definition as a string
        args (dict): Arguments to pass to the function
    
    Returns:
        CadQuery Workplane, Shape, or Assembly, or None
    """
    try:
        # Create a safe execution environment with necessary imports
        exec_globals = {
            'cq': cq,
            'cadquery': cq,
            'Workplane': cq.Workplane,
            'Assembly': cq.Assembly,
            'exporters': exporters,
            'os': os,
            'sys': sys,
            '__builtins__': __builtins__
        }
        
        # Execute the function definition
        exec(function_string, exec_globals)
        
        # Get the function from the execution environment
        if 'my_cad_function' not in exec_globals:
            raise ValueError("Function 'my_cad_function' not found in provided code")
        
        my_cad_function = exec_globals['my_cad_function']
        
        # Execute the function with arguments
        return my_cad_function(args)
        
    except Exception as e:
        print(f"Error executing LLM-provided function: {e}")
        import traceback
        traceback.print_exc()
        return None


###
# Example my_cad_function that creates three boxes.
###
# def my_cad_function(args):
#     boxes = []
#     for translation in [(100, 0, 0), (0, 100, 0), (0, 0, 100)]:
#         box = cq.Workplane("XY").box(10, 20, 30).translate(translation)
#         boxes.append(box)
#     # Combine all boxes into a compound
#     result = boxes[0]
#     for box in boxes[1:]:
#         result = result.union(box)
#     print(f"Created compound with multiple boxes at different locations.")
#     return result

###
# Example my_cad_function that loads a STEP file and prints information.
###
# def my_cad_function(args):
#     if "input_file" in args:
#         input_file = os.path.expanduser(args["input_file"])
#         shape = cq.importers.importStep(input_file)
#         # Get the underlying OCC shape for analysis
#         if hasattr(shape, 'val'):
#             occ_shape = shape.val()
#         else:
#             occ_shape = shape
#         print(f"Is Valid: {occ_shape.isValid()}")
#         print(f"Volume: {occ_shape.Volume():.6f} mm^3")
#         print(f"Number of faces: {len(occ_shape.Faces())}")
#         bbox = occ_shape.BoundingBox()
#         center = bbox.center
#         print(f"Bbox Center: ({center.x:.4f}, {center.y:.4f}, {center.z:.4f})")
#         return shape
#     return None


###
# Main script execution
###

if __name__ == "__main__":
    # Process command line arguments
    argv = sys.argv
    args = parse_cadquery_args(argv)
    output_dir = args.get("output_dir", "~/Desktop")
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load function from file or use default
    result = None
    if "function_file" in args:
        function_file = os.path.expanduser(args["function_file"])
        if os.path.exists(function_file):
            try:
                with open(function_file, 'r') as f:
                    function_string = f.read()
                print(f"Loading function from: {function_file}")
                result = load_and_execute_function(function_string, args)
            except Exception as e:
                print(f"Error loading function from file: {e}")
        else:
            print(f"Function file not found: {function_file}")
    else:
        print("No function file provided")

    # Export results if a result was returned
    if result is not None:
        export_as_step(result, output_dir)
        export_as_image(result, output_dir)
    
    print("Script completed.")