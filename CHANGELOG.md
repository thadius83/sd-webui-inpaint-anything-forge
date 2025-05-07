# Changelog - Inpaint Anything Extension Fix

## 2025-04-26

### Fixed
- Fixed indentation issue in lines 1187-1237 that was causing the UI layout to break
- Added support for Forge's Gradio 4+ ImageEditor component with new layer-based format
- Added helper function `extract_mask_from_image_editor()` to handle both old and new Gradio formats
- Updated `select_mask()` function to properly handle brush strokes in the new format
- Updated `input_image_upload()` to handle both old and new Gradio formats
- Updated `apply_mask()`, `add_mask()`, and `expand_mask()` functions to work with the new format
- Fixed "R" and "S" keyboard shortcuts for resetting zoom and fullscreen mode
- Added transform order fix to improve cursor position accuracy

### JavaScript Fixes - Round 1
- Corrected transform order in CSS to be `translate()` first, then `scale()`
- Added a simplified `fixEditorTransform()` function to correct transform issues
- Simplified keyboard event handling to use event.key instead of event.code
- Streamlined the zoom and fullscreen functions to avoid breaking functionality
- Added auto-scale for very large images (>2000px dimensions)
- Removed toolbar and cursor manipulations that were breaking the UI
- Fixed zoom functionality for images of all sizes
- Maintained the core functionality while minimizing changes

### JavaScript Fixes - Round 2 (Simplification)
- Rolled back overly aggressive CSS changes that broke zoom functionality
- Simplified keyboard shortcut handling to just use 'r' and 's' keys directly
- Implemented minimal transform order fix that only changes what's necessary 
- Added special handling for images over 2000px that scales them appropriately
- Removed CSS manipulations that affected cursor visibility
- Fixed fullscreen mode to work with images of all sizes

### JavaScript Fixes - Round 3 (Large Image Handling)
- Added comprehensive large image handling with proper scaling and positioning
- Fixed images appearing outside their containers by adding overflow control
- Improved cursor alignment for large images by consistently setting transformOrigin
- Enhanced fullscreen mode to properly handle large images with correct scaling
- Added SVG overlay fixes to ensure cursor alignment in Gradio 4+
- Implemented special handling for extremely large images (>3000px) with minimum scale
- Added automatic detection of image size to apply appropriate scaling
- Fixed image centering in both container and fullscreen views

### Technical Details
- In Gradio 4+, the ImageEditor returns a dictionary with a "layers" key containing a list of RGBA layers
- We now extract alpha channels from each layer and create a binary mask based on maximum alpha values
- Added proper fallbacks for older Gradio versions to maintain backward compatibility
- Used a consistent helper function to ensure the extraction logic is applied uniformly
- Applied minimal CSS changes to ensure cursor alignment without breaking other functionality
- Used a proportional scaling approach for large images based on viewport dimensions
- Fixed transform origin issues to ensure cursor and brush tools align properly with canvas

The changes ensure that drawing and selection tools now work properly with Forge's Gradio 4+ interface, with accurate cursor positioning and visible editing tools regardless of image size, from small thumbnails to very large images over 3000px in width or height.