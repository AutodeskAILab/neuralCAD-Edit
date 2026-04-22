#!/usr/bin/env python3
"""
Recursive Batch Export Script for CAD Files using CadQuery

Processes STEP files recursively, exporting to multiple formats:
STL and PNG previews from multiple camera angles.
Skips files that already have all output formats.

This script uses CadQuery (open-source) instead of Fusion 360.
"""

import sys
from pathlib import Path
import argparse
import cadquery as cq
from cadquery import exporters

from src.utils.cadquery_rendering import render_to_png, VIEW_PROJECTIONS

EXPORT_VIEWS = list(VIEW_PROJECTIONS.keys())


def load_step_file(step_path):
    """Load a STEP file using CadQuery.
    
    Args:
        step_path: Path to the STEP file
    
    Returns:
        CadQuery Workplane object, or None if loading failed
    """
    try:
        result = cq.importers.importStep(str(step_path))
        if result is None:
            print(f"Failed to load: {step_path}")
            return None
        return result
    except Exception as e:
        print(f"Error loading {step_path}: {e}")
        return None


def export_stl(stl_path, workplane):
    """Export STL format of the workplane.
    
    Args:
        stl_path: Output path for STL file
        workplane: CadQuery Workplane object
    
    Returns:
        True if successful, False otherwise
    """
    try:
        exporters.export(workplane, str(stl_path), exporters.ExportTypes.STL)
        return True
    except Exception as e:
        print(f"Error exporting STL to {stl_path}: {e}")
        return False


def export_step(step_path, workplane):
    """Export STEP format of the workplane.
    
    Args:
        step_path: Output path for STEP file
        workplane: CadQuery Workplane object
    
    Returns:
        True if successful, False otherwise
    """
    try:
        exporters.export(workplane, str(step_path), exporters.ExportTypes.STEP)
        return True
    except Exception as e:
        print(f"Error exporting STEP to {step_path}: {e}")
        return False


def export_png_view(png_path, workplane, view_name, image_size=1024):
    """Export a PNG view from a specific camera angle using V3d offscreen rendering.
    
    Args:
        png_path: Output path for PNG file
        workplane: CadQuery Workplane object
        view_name: Name of the view to export
        image_size: Size of the output image (square)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if view_name not in VIEW_PROJECTIONS:
            print(f"Unknown view: {view_name}")
            return False

        shape = workplane.val() if hasattr(workplane, 'val') else workplane
        render_to_png(shape, png_path, proj=VIEW_PROJECTIONS[view_name],
                      width=image_size, height=image_size)
        return True

    except Exception as e:
        print(f"Error exporting PNG view {view_name} to {png_path}: {e}")
        return False


def export_all_png_views(base_path, workplane, image_size=1024):
    """Export PNG views from all standard camera angles.
    
    Args:
        base_path: Base path for PNG files (view name will be appended)
        workplane: CadQuery Workplane object
        image_size: Size of output images
    
    Returns:
        Number of successful exports
    """
    successful_count = 0
    
    for view_name in EXPORT_VIEWS:
        view_png_path = f"{base_path}_{view_name}.png"
        if export_png_view(view_png_path, workplane, view_name, image_size):
            successful_count += 1
    
    return successful_count


def find_cad_files(root_directory, extensions=['step']):
    """Find CAD files in brep_start/brep_end directories.
    
    Searches for directories named 'brep_start' or 'brep_end', then looks in their
    latest timestamped subdirectory for CAD files with the specified extensions.
    
    Args:
        root_directory: Root directory to search
        extensions: List of file extensions to look for
    
    Returns:
        Sorted list of unique CAD file paths
    """
    root_path = Path(root_directory)
    brep_dirs = []
    
    # Find brep_start and brep_end directories with their latest timestamps
    for brep_type in ['brep_start', 'brep_end']:
        for brep_dir in root_path.rglob(brep_type):
            if brep_dir.is_dir():
                # Find latest timestamp subdirectory
                timestamp_dirs = sorted([d for d in brep_dir.iterdir() if d.is_dir()])
                if timestamp_dirs:
                    brep_dirs.append(timestamp_dirs[-1])
    
    # Collect CAD files from each directory (first matching extension wins)
    cad_files = []
    for brep_dir in brep_dirs:
        available_files = [f for f in brep_dir.iterdir() if f.is_file()]
        for ext in extensions:
            matches = [f for f in available_files if f.suffix.lower() == f'.{ext.lower()}']
            if matches:
                cad_files.extend(matches)
                break
    
    return sorted(set(cad_files))


def process_file(cad_file, skip_existing=True, image_size=1024):
    """Process a single CAD file, exporting to all formats.
    
    Args:
        cad_file: Path to CAD file
        skip_existing: If True, skip files where all outputs exist
        image_size: Size of output images
    
    Returns:
        Tuple of (success, skipped, error_message)
    """
    cad_path = Path(cad_file)
    base_name = cad_path.with_suffix('')
    
    # Define output paths
    stl_path = base_name.with_suffix('.stl')
    step_output_path = base_name.with_suffix('.step')
    
    # Check for all view PNG files
    png_files = [Path(f"{base_name}_{view}.png") for view in EXPORT_VIEWS]
    
    # Check if all outputs already exist
    stl_exists = stl_path.exists()
    step_exists = step_output_path.exists() if str(step_output_path) != str(cad_path) else True
    png_exists = all(png.exists() for png in png_files)
    
    if skip_existing and stl_exists and step_exists and png_exists:
        return False, True, None  # Not processed, skipped, no error
    
    # Load the STEP file
    try:
        workplane = load_step_file(cad_path)
        if workplane is None:
            return False, False, "Failed to load STEP file"
        
        # Export STL if needed
        if not stl_exists:
            if not export_stl(stl_path, workplane):
                return False, False, "STL export failed"
        
        # Export STEP if needed (and it's a different file)
        if not step_exists and str(step_output_path) != str(cad_path):
            if not export_step(step_output_path, workplane):
                return False, False, "STEP export failed"
        
        # Export PNG views if needed
        if not png_exists:
            successful_views = export_all_png_views(str(base_name), workplane, image_size)
            if successful_views == 0:
                return False, False, "All PNG view exports failed"
        
        return True, False, None  # Processed, not skipped, no error
        
    except Exception as e:
        return False, False, str(e)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Batch export STEP files to STL and PNG previews using CadQuery'
    )
    parser.add_argument(
        'root_directory',
        type=str,
        help='Root directory to search for CAD files'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Process all files even if outputs already exist'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=1024,
        help='Size of output images in pixels (default: 1024)'
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    root_path = Path(args.root_directory)
    if not root_path.exists():
        print(f"Error: Directory does not exist: {args.root_directory}")
        sys.exit(1)
    
    # Find all STEP files
    print(f"Searching for STEP files in {args.root_directory}...")
    cad_files = find_cad_files(args.root_directory, extensions=['step'])
    
    if not cad_files:
        print("No STEP files found in brep_start/brep_end directories.")
        sys.exit(0)
    
    print(f"Found {len(cad_files)} STEP files to process.")
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    errors = []
    
    for i, cad_file in enumerate(cad_files, 1):
        print(f"\n[{i}/{len(cad_files)}] Processing: {cad_file.name}")
        
        success, skipped, error = process_file(
            cad_file,
            skip_existing=not args.no_skip,
            image_size=args.image_size
        )
        
        if success:
            processed_count += 1
            print("  ✓ Successfully processed")
        elif skipped:
            skipped_count += 1
            print("  ⊘ Skipped (all outputs exist)")
        else:
            errors.append((cad_file.name, error))
            print(f"  ✗ Error: {error}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"Total files checked: {len(cad_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (existed): {skipped_count}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  • {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == '__main__':
    main()