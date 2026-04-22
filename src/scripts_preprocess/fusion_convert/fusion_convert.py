#!/usr/bin/env python3
"""
Recursive Batch Export Script for CAD Files

Processes F3D, STEP, and SMT files recursively, exporting to multiple formats:
SMT, STL, STEP, F3D, and JPG previews from multiple camera angles.
Skips files that already have all output formats.
"""

import adsk.core
import adsk.fusion
import os

# Views to export for JPG previews
EXPORT_VIEWS = ['toprightiso', 'front', 'back', 'left', 'right', 'top', 'bottom']

# Standard view orientations (Y-up coordinate system, used by F3D/SMT)
VIEW_ORIENTATIONS = {
    'toprightiso': adsk.core.ViewOrientations.IsoTopRightViewOrientation,
    'front': adsk.core.ViewOrientations.FrontViewOrientation,
    'back': adsk.core.ViewOrientations.BackViewOrientation,
    'left': adsk.core.ViewOrientations.LeftViewOrientation,
    'right': adsk.core.ViewOrientations.RightViewOrientation,
    'top': adsk.core.ViewOrientations.TopViewOrientation,
    'bottom': adsk.core.ViewOrientations.BottomViewOrientation,
}

# STEP files use Z-up, so we remap views to show correct faces
# Note: isometric doesn't map correctly for Z-up
STEP_VIEW_ORIENTATIONS = {
    'toprightiso': adsk.core.ViewOrientations.IsoTopRightViewOrientation,
    'front': adsk.core.ViewOrientations.TopViewOrientation,
    'back': adsk.core.ViewOrientations.BottomViewOrientation,
    'left': adsk.core.ViewOrientations.LeftViewOrientation,
    'right': adsk.core.ViewOrientations.RightViewOrientation,
    'top': adsk.core.ViewOrientations.BackViewOrientation,
    'bottom': adsk.core.ViewOrientations.FrontViewOrientation,
}


def set_camera_view(app, view_name, is_step_file=False):
    """Set the camera to a specific standard view.
    
    Args:
        app: Fusion 360 application object
        view_name: Name of the view (e.g., 'front', 'top', 'toprightiso')
        is_step_file: If True, applies coordinate system compensation for STEP files (Z-up)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        orientations = STEP_VIEW_ORIENTATIONS if is_step_file else VIEW_ORIENTATIONS
        orientation = orientations.get(view_name)
        if not orientation:
            return False
        
        viewport = app.activeViewport
        camera = viewport.camera
        camera.isFitView = True
        camera.isSmoothTransition = False
        camera.viewOrientation = orientation
        viewport.camera = camera
        viewport.fit()
        viewport.refresh()
        return True
    except Exception:
        return False


def export_jpg_single_view(jpg_path, app, view_name, is_step_file=False):
    """Export a fitted JPG view from a specific camera angle.
    
    Args:
        jpg_path: Output path for the JPG file
        app: Fusion 360 application object
        view_name: Name of the view to export
        is_step_file: If True, applies coordinate system compensation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not set_camera_view(app, view_name, is_step_file):
            return False
        
        success = app.activeViewport.saveAsImageFile(jpg_path, 1024, 1024)
        if not success:
            raise Exception(f"JPG export failed for {view_name} view")
        return True
    except Exception:
        return False


def export_jpg(jpg_path, app, is_step_file=False):
    """Export JPG views from all standard camera angles.
    
    Args:
        jpg_path: Base path for JPG files (view name will be appended)
        app: Fusion 360 application object
        is_step_file: If True, applies coordinate system compensation
    
    Returns:
        True if at least one view exported successfully
    
    Raises:
        Exception: If all view exports fail
    """
    base_path = jpg_path.replace('.jpg', '')
    successful_count = 0
    
    for view_name in EXPORT_VIEWS:
        view_jpg_path = f"{base_path}_{view_name}.jpg"
        if export_jpg_single_view(view_jpg_path, app, view_name, is_step_file):
            successful_count += 1
    
    if successful_count == 0:
        raise Exception("All view exports failed")
    
    return True

def export_smt(path, export_mgr):
    """Export SMT format of the current design."""
    options = export_mgr.createSMTExportOptions(path)
    if not options:
        raise Exception("Failed to create SMT export options")
    if not export_mgr.execute(options):
        raise Exception("SMT export execution failed")


def export_stl(path, export_mgr, design):
    """Export STL format of the current design."""
    options = export_mgr.createSTLExportOptions(design.rootComponent, path)
    if not options:
        raise Exception("Failed to create STL export options")
    options.isBinaryFormat = True
    options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
    options.sendToPrintUtility = False
    if not export_mgr.execute(options):
        raise Exception("STL export execution failed")


def export_step(path, export_mgr):
    """Export STEP format of the current design."""
    options = export_mgr.createSTEPExportOptions(path)
    if not options:
        raise Exception("Failed to create STEP export options")
    if not export_mgr.execute(options):
        raise Exception("STEP export execution failed")


def export_f3d(path, export_mgr):
    """Export F3D (Fusion archive) format of the current design."""
    options = export_mgr.createFusionArchiveExportOptions(path)
    if not options:
        raise Exception("Failed to create F3D export options")
    if not export_mgr.execute(options):
        raise Exception("F3D export execution failed")


def hide_ui_elements(design):
    """Hide analysis, origin, sketch, and joints folders for cleaner screenshots."""
    design.analyses.isLightBulbOn = False
    root = design.rootComponent
    root.isOriginFolderLightBulbOn = False
    root.isSketchFolderLightBulbOn = False
    root.isJointsFolderLightBulbOn = False


def find_cad_files(root_directory, extensions):
    """Find CAD files in brep_start/brep_end directories.
    
    Searches for directories named 'brep_start' or 'brep_end', then looks in their
    latest timestamped subdirectory for CAD files with the specified extensions.
    
    Args:
        root_directory: Root directory to search
        extensions: List of file extensions to look for (in priority order)
    
    Returns:
        Sorted list of unique CAD file paths
    """
    brep_dirs = []
    
    # Find brep_start and brep_end directories with their latest timestamps
    for root, dirs, _ in os.walk(root_directory):
        if os.path.basename(root) in ['brep_start', 'brep_end']:
            timestamp_dirs = sorted(dirs)
            if timestamp_dirs:
                brep_dirs.append(os.path.join(root, timestamp_dirs[-1]))
    
    # Collect CAD files from each directory (first matching extension wins)
    cad_files = []
    for brep_dir in brep_dirs:
        available_files = [os.path.join(brep_dir, f) for f in os.listdir(brep_dir)]
        for ext in extensions:
            matches = [f for f in available_files if f.lower().endswith(f'.{ext.lower()}')]
            if matches:
                cad_files.extend(matches)
                break
    
    return sorted(set(cad_files))


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        print("Starting batch export (SMT + STL + STEP + F3D + JPG)...")
        
        # Show folder selection dialog
        folder_dialog = ui.createFolderDialog()
        folder_dialog.title = 'Select Root Folder for CAD Export'
        folder_dialog.initialDirectory = os.path.expanduser('~/Desktop')
        
        if folder_dialog.showDialog() != adsk.core.DialogResults.DialogOK:
            return
        
        root_directory = folder_dialog.folder
        cad_files = find_cad_files(root_directory, ['f3d', 'step', 'smt'])

        app.log(f"Found {len(cad_files)} CAD files to process.")
        app.log(f"{cad_files}")

        if not cad_files:
            if ui:
                ui.messageBox('No cad files found in the selected directory tree.', 'No Files')
            return
        
        # Process each CAD file
        processed_count = 0
        skipped_count = 0
        errors = []
        
        for cad_file in cad_files:
            filename = os.path.basename(cad_file)
            extension = os.path.splitext(filename)[-1].lower()[1:]  # get extension without dot
            
            try:
                # Create output paths
                base_name = os.path.splitext(cad_file)[0]
                smt_path = base_name + '.smt'
                stl_path = base_name + '.stl'
                step_path = base_name + '.step'
                f3d_path = base_name + '.f3d'

                # Check for all view JPG files
                jpg_files = {view: f"{base_name}_{view}.jpg" for view in EXPORT_VIEWS}
                
                # Check if all output files already exist
                smt_exists = os.path.exists(smt_path)
                stl_exists = os.path.exists(stl_path)
                step_exists = os.path.exists(step_path)
                jpg_exists = all(os.path.exists(jpg_path) for jpg_path in jpg_files.values())
                f3d_exists = os.path.exists(f3d_path)

                # Skip if all outputs already exist
                if smt_exists and stl_exists and step_exists and jpg_exists and f3d_exists:
                    skipped_count += 1
                    continue
                
                # Import CAD file and process exports
                doc = None
                try:
                    import_mgr = app.importManager
                    import_options = {
                        'f3d': import_mgr.createFusionArchiveImportOptions,
                        'step': import_mgr.createSTEPImportOptions,
                        'smt': import_mgr.createSMTImportOptions,
                    }.get(extension)
                    
                    if not import_options:
                        raise Exception(f"Unsupported file type: {extension}")
                    
                    doc = import_mgr.importToNewDocument(import_options(cad_file))
                    if not doc:
                        raise Exception("Failed to import CAD file")

                    # Get design and export manager
                    design = doc.products.itemByProductType('DesignProductType')
                    if not design:
                        raise Exception("No design found")
                    
                    export_mgr = design.exportManager
                    if not export_mgr:
                        raise Exception("No export manager")

                    hide_ui_elements(design)
                    
                    # Export each format if it doesn't exist
                    if not jpg_exists:
                        export_jpg(base_name + '.jpg', app, is_step_file=(extension == 'step'))
                    if not smt_exists:
                        export_smt(smt_path, export_mgr)
                    if not stl_exists:
                        export_stl(stl_path, export_mgr, design)
                    if not step_exists:
                        export_step(step_path, export_mgr)
                    if not f3d_exists:
                        export_f3d(f3d_path, export_mgr)

                    processed_count += 1
                    
                except Exception as processing_error:
                    errors.append((filename, str(processing_error)))
                    
                finally:
                    # Always ensure document is closed
                    if doc:
                        try:
                            doc.close(False)
                        except Exception:
                            pass

            except Exception as file_error:
                errors.append((filename, str(file_error)))
                # Emergency cleanup - force close any open documents
                try:
                    for open_doc in app.documents:
                        if open_doc.isActive:
                            open_doc.close(False)
                except Exception:
                    pass
        
        # Show results
        if ui:
            if errors:
                error_list = "\n".join([f"• {name}: {err}" for name, err in errors[:3]])  # Show first 3 errors
                if len(errors) > 3:
                    error_list += f"\n... and {len(errors) - 3} more"
                message = f'Export Complete!\n\nChecked: {len(cad_files)} CAD files\nProcessed: {processed_count} files\nSkipped: {skipped_count} files (existed)\n\nErrors ({len(errors)}):\n{error_list}'
            else:
                if skipped_count > 0:
                    message = f'Export Complete!\n\nChecked: {len(cad_files)} CAD files\nProcessed: {processed_count} new files\nSkipped: {skipped_count} files (already existed)\n\nAll files now have SMT + STL + JPG exports!'
                else:
                    message = f'Export Complete!\n\nSuccessfully processed all {processed_count} CAD files!\nAll files now have SMT + STL + JPG exports!'
            
            ui.messageBox(message, 'Triple Format Export Results')
        
    except Exception as e:
        if ui:
            ui.messageBox(f'Export failed:\n\n{str(e)}', 'Export Error')

def stop(context):
    pass 

