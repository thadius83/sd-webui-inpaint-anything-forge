"""Inpaint‑Anything extension – fully working with Gradio 4.40+

Only functional changes are marked with  ⚠  in comments.
"""
from __future__ import annotations
import gc, math, os, platform, random, re, traceback
from typing import Any, Dict, Optional, Tuple

# ‑‑‑ Environment flags -------------------------------------------------------
if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if platform.system() == "Windows":
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

# ‑‑‑ Standard libs -----------------------------------------------------------
import cv2, numpy as np, gradio as gr, torch
from PIL import Image, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from torchvision import transforms
from torch.hub import download_url_to_file

# ‑‑‑ Web‑UI / diffusers ------------------------------------------------------
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    StableDiffusionInpaintPipeline,
)
from modules import devices, script_callbacks, shared
from modules.sd_samplers import samplers_for_img2img
from modules.processing import create_infotext, process_images
from modules.sd_models import get_closet_checkpoint_match
from gradio import ImageEditor, Brush

# ‑‑‑ Local helpers -----------------------------------------------------------
import inpalib  # segmentation + mask utils
from ia_check_versions import ia_check_versions
from ia_config import (
    IAConfig, setup_ia_config_ini, set_ia_config, get_ia_config_index,
    get_webui_setting,
)
from ia_file_manager import IAFileManager, ia_file_manager, download_model_from_hf
from ia_logging import ia_logging, draw_text_image
from ia_threading import (
    clear_cache_decorator, offload_reload_decorator,
    async_post_reload_model_weights, await_backup_reload_ckpt_info,
    await_pre_reload_model_weights,
)
from ia_ui_items import (
    get_sam_model_ids, get_inp_model_ids, get_cleaner_model_ids,
    get_inp_webui_model_ids, get_padding_mode_names, get_sampler_names,
)
from ia_webui_controlnet import (
    find_controlnet, get_sd_img2img_processing, get_controlnet_args_to,
    get_max_args_to, backup_alwayson_scripts, disable_alwayson_scripts_wo_cn,
    disable_all_alwayson_scripts, clear_controlnet_cache, restore_alwayson_scripts,
)
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler

CANVAS_W, CANVAS_H = int(768 * 1.30), int(576 * 1.30)   # ~= 998 × 749
PAD = 20                                                # white border



@clear_cache_decorator
def download_model(sam_model_id):
    """Download SAM model.

    Args:
        sam_model_id (str): SAM model id

    Returns:
        str: download status
    """
    if "_hq_" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/sam-hq/resolve/main/" + sam_model_id
    elif "FastSAM" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/FastSAM/resolve/main/" + sam_model_id
    elif "mobile_sam" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/MobileSAM/resolve/main/" + sam_model_id
    elif "sam2_" in sam_model_id:
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/" + sam_model_id
    else:
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id

    sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        try:
            download_url_to_file(url_sam, sam_checkpoint)
        except Exception as e:
            ia_logging.error(str(e))
            return str(e)

        return IAFileManager.DOWNLOAD_COMPLETE
    else:
        return "Model already exists"


# -----------------------------------------------------------------------------
# ⚠  NEW — robust mask extraction for Gradio 4.40+
# -----------------------------------------------------------------------------

def _to_ndarray(layer: Any) -> np.ndarray:
    """Convert PIL.Image → np.ndarray; leave ndarray untouched."""
    if isinstance(layer, Image.Image):
        return np.array(layer)
    return layer

def extract_mask_from_image_editor(val: Any) -> Optional[np.ndarray]:
    """Return an H×W×1 uint8 mask (255 = painted pixel) or None."""
    if val is None:
        return None

    # ── 4.x format ───────────────────────────────────────────────────────────
    if isinstance(val, dict):
        # 1️⃣ explicit layers list
        if val.get("layers"):
            alphas = []
            for L in val["layers"]:
        # NEW ↓ honour dict-with-"data" as produced by Brush in Gradio ≥4.4
                if isinstance(L, dict) and "data" in L:
                    arr = _to_ndarray(L["data"])
                else:
                    arr = _to_ndarray(L)
        # ------------------------------------------------------
                if arr.ndim == 3 and arr.shape[2] == 4:        # RGBA
                    alphas.append(arr[:, :, 3])
                else:                                          # RGB fallback
                    gray = np.mean(arr[..., :3], 2)
                    alphas.append((gray > 0).astype(np.uint8) * 255)    
            
            if alphas:
                m = (np.max(alphas, 0) > 0).astype(np.uint8) * 255
                return m[:, :, None]

        # 2️⃣ single‑layer shortcut: composite key
        if isinstance(val.get("composite"), (np.ndarray, Image.Image)):
            comp = _to_ndarray(val["composite"])
            if comp.ndim == 3 and comp.shape[2] == 4:
                return comp[:, :, 3:4]
            gray = np.mean(comp[..., :3], 2)
            return ((gray > 0).astype(np.uint8) * 255)[:, :, None]

        # 3️⃣ legacy key hosted inside new dict
        if "mask" in val and isinstance(val["mask"], np.ndarray):
            m = val["mask"]
            return m[:, :, None] if m.ndim == 2 else m

    # ── pure ndarray (old Gradio ≤ 3.44) ─────────────────────────────────────
    if isinstance(val, np.ndarray):
        return val[:, :, None] if val.ndim == 2 else val

    return None

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)


def save_mask_image(mask_image, save_mask_chk=False):
    """Save mask image.

    Args:
        mask_image (np.ndarray): mask image
        save_mask_chk (bool, optional): If True, save mask image. Defaults to False.

    Returns:
        None
    """
    if save_mask_chk:
        save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)


@clear_cache_decorator
def input_image_upload(input_image, sam_image, sel_mask):
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None

    if (sam_dict["mask_image"] is None or not isinstance(sam_dict["mask_image"], np.ndarray) or
            sam_dict["mask_image"].shape != input_image.shape):
        sam_dict["mask_image"] = np.zeros_like(input_image, dtype=np.uint8)

    ret_sel_image = cv2.addWeighted(input_image, 0.5, sam_dict["mask_image"], 0.5, 0)

    # Handle Gradio 4+ ImageEditor output
    if sam_image is None:
        sam_dict["sam_masks"] = None
        ret_sam_image = np.zeros_like(input_image, dtype=np.uint8)
    elif isinstance(sam_image, dict) and "background" in sam_image:
        # For Gradio 4+
        if sam_image["background"].shape == input_image.shape:
            ret_sam_image = gr.update()
        else:
            sam_dict["sam_masks"] = None
            ret_sam_image = gr.update(value=np.zeros_like(input_image, dtype=np.uint8))
    elif isinstance(sam_image, dict) and "image" in sam_image:
        # For old Gradio
        if sam_image["image"].shape == input_image.shape:
            ret_sam_image = gr.update()
        else:
            sam_dict["sam_masks"] = None
            ret_sam_image = gr.update(value=np.zeros_like(input_image, dtype=np.uint8))
    else:
        sam_dict["sam_masks"] = None
        ret_sam_image = gr.update(value=np.zeros_like(input_image, dtype=np.uint8))

    # Handle sel_mask with either Gradio format
    if sel_mask is None:
        ret_sel_mask = ret_sel_image
    elif isinstance(sel_mask, dict) and "image" in sel_mask:
        # Old Gradio format
        if sel_mask["image"].shape == ret_sel_image.shape and np.all(sel_mask["image"] == ret_sel_image):
            ret_sel_mask = gr.update()
        else:
            ret_sel_mask = gr.update(value=ret_sel_image)
    elif isinstance(sel_mask, dict) and "background" in sel_mask:
        # Gradio 4+ format
        if sel_mask["background"].shape == ret_sel_image.shape:
            ret_sel_mask = gr.update()
        else:
            ret_sel_mask = gr.update(value=ret_sel_image)
    else:
        ret_sel_mask = gr.update(value=ret_sel_image)

    return ret_sam_image, ret_sel_mask, gr.update(interactive=True)

# Function to resize an image to fit within a specified dimension while maintaining aspect ratio
def resize_image_to_fit(image, 
                        max_width=CANVAS_W + 2*PAD,
                        max_height=CANVAS_H + 2*PAD):
                        
    if image is None:
        return None
        
    height, width = image.shape[:2]
    
    # Calculate the scaling factor - always scale to fit exactly within the canvas
    # This ensures consistent behavior regardless of window size
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    
    # Always resize to ensure consistent size across all editors
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


@clear_cache_decorator
def run_padding(input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode="edge"):
    global sam_dict
    if input_image is None or sam_dict["orig_image"] is None:
        sam_dict["orig_image"] = None
        sam_dict["pad_mask"] = None
        return None, "Input image not found"

    orig_image = sam_dict["orig_image"]

    height, width = orig_image.shape[:2]
    pad_width, pad_height = (int(width * pad_scale_width), int(height * pad_scale_height))
    ia_logging.info(f"resize by padding: ({height}, {width}) -> ({pad_height}, {pad_width})")

    pad_size_w, pad_size_h = (pad_width - width, pad_height - height)
    pad_size_l = int(pad_size_w * pad_lr_barance)
    pad_size_r = pad_size_w - pad_size_l
    pad_size_t = int(pad_size_h * pad_tb_barance)
    pad_size_b = pad_size_h - pad_size_t

    pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r), (0, 0)]
    if padding_mode == "constant":
        fill_value = get_webui_setting("inpaint_anything_padding_fill", 127)
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode, constant_values=fill_value)
    else:
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode)

    mask_pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r)]
    pad_mask = np.zeros((height, width), dtype=np.uint8)
    pad_mask = np.pad(pad_mask, pad_width=mask_pad_width, mode="constant", constant_values=255)
    sam_dict["pad_mask"] = dict(segmentation=pad_mask.astype(bool))

    return pad_image, "Padding done"


@offload_reload_decorator
@clear_cache_decorator
def run_sam(input_image, sam_model_id, sam_image, anime_style_chk=False):
    global sam_dict
    if not inpalib.sam_file_exists(sam_model_id):
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, f"{sam_model_id} not found, please download"

    if input_image is None:
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Input image not found"

    set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

    if sam_dict["sam_masks"] is not None:
        sam_dict["sam_masks"] = None
        gc.collect()

    ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

    try:
        sam_masks = inpalib.generate_sam_masks(input_image, sam_model_id, anime_style_chk)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

        seg_image = inpalib.create_seg_color_image(input_image, sam_masks)

        sam_dict["sam_masks"] = sam_masks

    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Segment Anything failed"

    # Resize the segmentation image to fit in the window
    resized_seg_image = resize_image_to_fit(seg_image)
    
    # Handle different Gradio output formats
    if sam_image is None:
        return resized_seg_image, "Segment Anything complete"
    elif isinstance(sam_image, dict) and "background" in sam_image:
        # Gradio 4+ format
        return gr.update(value=resized_seg_image), "Segment Anything complete"
    elif isinstance(sam_image, dict) and "image" in sam_image:
        # Old Gradio format
        return gr.update(value=resized_seg_image), "Segment Anything complete"
    else:
        # Fallback
        return gr.update(value=resized_seg_image), "Segment Anything complete"


@clear_cache_decorator

def select_mask(input_image, sam_image, invert_chk, ignore_black_chk, sel_mask):
    global sam_dict
    # If nothing segmented yet, clear the preview
    if sam_dict["sam_masks"] is None or sam_image is None:
        ia_logging.info("No segmentation masks available or sam_image is None")
        return None if sel_mask is None else gr.update()
    
    sam_masks = sam_dict["sam_masks"]
    
    # Extract mask using the helper function
    mask = extract_mask_from_image_editor(sam_image)
    
    # Add diagnostic logging to help debug mask issues
    if mask is None or not mask.any():
        ia_logging.warning("Empty mask extracted – check editor output!")
        # Create an empty mask based on background if possible
        if isinstance(sam_image, dict) and "background" in sam_image:
            background = _to_ndarray(sam_image["background"])
            mask = np.zeros((background.shape[0], background.shape[1], 1), dtype=np.uint8)
            ia_logging.info(f"Created empty mask with shape {mask.shape}")
    else:
        ia_logging.info(f"Extracted mask with shape {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
    
    # Get the original image - this is the full resolution image we need to match
    orig_image = sam_dict["orig_image"]
    if orig_image is None:
        ia_logging.error("Original image not found")
        return None if sel_mask is None else gr.update()
    
    # If we couldn't extract a mask, create an empty one based on display dimensions
    if mask is None:
        if isinstance(sam_image, dict) and "background" in sam_image:
            background = _to_ndarray(sam_image["background"])
            height, width = background.shape[:2]
        elif isinstance(sam_image, np.ndarray):
            height, width = sam_image.shape[:2]
        else:
            # If we can't determine dimensions, return without updating
            ia_logging.error("Could not determine mask dimensions")
            return None if sel_mask is None else gr.update()
            
        mask = np.zeros((height, width, 1), dtype=np.uint8)
    
    # Get dimensions of mask and original image
    orig_height, orig_width = orig_image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    
    ia_logging.info(f"Original image: {orig_width}x{orig_height}, Mask: {mask_width}x{mask_height}")
    
    try:
        # If dimensions don't match, resize mask to match the original image
        if mask_height != orig_height or mask_width != orig_width:
            ia_logging.info(f"Resizing mask from {mask_width}x{mask_height} to {orig_width}x{orig_height}")
            mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            if len(mask_resized.shape) == 2:
                mask_resized = mask_resized[:, :, np.newaxis]
        else:
            mask_resized = mask
        
        # Create mask with properly sized inputs
        seg_image = inpalib.create_mask_image(mask_resized, sam_masks, ignore_black_chk)
        if invert_chk:
            seg_image = inpalib.invert_mask(seg_image)
        sam_dict["mask_image"] = seg_image
        
        # Resize the result back for display
        if mask_height != orig_height or mask_width != orig_width:
            display_seg_image = cv2.resize(seg_image, (mask_width, mask_height), interpolation=cv2.INTER_AREA)
        else:
            display_seg_image = seg_image
    except Exception as e:
        ia_logging.error(f"Error creating mask: {str(e)}")
        ia_logging.error(traceback.format_exc())
        return None if sel_mask is None else gr.update()
    
    # Initialize ret to default value in case of any errors
    ret = display_seg_image
    
    try:
        disp_h, disp_w = display_seg_image.shape[:2]

        if input_image is None:
            ia_logging.info("Preview-overlay: input_image is None → show mask only")
            ret = display_seg_image
        else:
            # ----- make sure base image matches preview size ----------------------
            if input_image.shape[:2] != (disp_h, disp_w):
                ia_logging.info(
                    f"Preview-overlay: resizing base from "
                    f"{input_image.shape[:2]} → {(disp_h, disp_w)}"
                )
                base = cv2.resize(
                    input_image, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                )
            else:
                base = input_image

            # ----- build boolean mask for overlay  ---------------------------
            # Always take the *final* stored mask (sam_dict["mask_image"]):
            try:
                full_mask = sam_dict["mask_image"]               # H×W×3   255 = selected
                if full_mask.ndim == 3:
                    full_mask = full_mask[:, :, 0]               # any channel is fine

                # resize it to the preview size so shapes match
                mask_disp = cv2.resize(
                    full_mask,
                    (disp_w, disp_h),
                    interpolation=cv2.INTER_NEAREST
                ) > 0                                            # boolean

            except Exception as e:
                ia_logging.error(f"Could not build preview mask: {e}")
                mask_disp = np.zeros((disp_h, disp_w), bool)

            # ----- compose --------------------------------------------------------
            overlay = base.copy()
            red     = np.zeros_like(overlay)
            red[:, :, 0] = 255                    # pure red
            alpha = 0.45

            overlay[mask_disp] = (
                alpha * red[mask_disp] + (1 - alpha) * overlay[mask_disp]
            ).astype(np.uint8)

            ret = overlay

    except Exception as e:
        # Any unexpected shape issue ⇒ fall back to the plain mask & log the error
        ia_logging.error(f"Preview-overlay failure: {e}")
        ia_logging.error(traceback.format_exc())
        # ret is already set to display_seg_image as default

    # Update the Gradio preview
    return ret if not (isinstance(sel_mask, dict) and sel_mask.get("image") is ret) else gr.update()

@clear_cache_decorator
def expand_mask(input_image, sel_mask, expand_iteration=1):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    new_sel_mask = sam_dict["mask_image"]

    expand_iteration = int(np.clip(expand_iteration, 1, 100))

    new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    # Check if we need to update based on the format
    if isinstance(sel_mask, dict) and "image" in sel_mask:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
    elif isinstance(sel_mask, dict) and "background" in sel_mask:
        # For Gradio 4+ format, we just return an update with the new value
        pass
        
    return gr.update(value=ret_image)


@clear_cache_decorator
def apply_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    
    # Extract mask using the helper function
    user_mask = extract_mask_from_image_editor(sel_mask)
    
    if user_mask is None:
        ia_logging.error("Could not extract mask from image editor output")
        return gr.update()
    
    # Invert the mask for trim operation
    sel_mask_mask = np.logical_not(user_mask[:, :, 0:3].astype(bool)).astype(np.uint8)
    new_sel_mask = sel_mask_image * sel_mask_mask

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    # Check if we need to update
    if isinstance(sel_mask, dict) and "image" in sel_mask:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
    
    return gr.update(value=ret_image)


@clear_cache_decorator
def add_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    
    # Extract mask using the helper function
    user_mask = extract_mask_from_image_editor(sel_mask)
    
    if user_mask is None:
        ia_logging.error("Could not extract mask from image editor output")
        return gr.update()
        
    # Use the mask for addition operation
    sel_mask_mask = user_mask[:, :, 0:3].astype(bool).astype(np.uint8)
    new_sel_mask = sel_mask_image + (sel_mask_mask * np.invert(sel_mask_image, dtype=np.uint8))

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    # Check if we need to update
    if isinstance(sel_mask, dict) and "image" in sel_mask:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
    
    return gr.update(value=ret_image)


def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height*scale+0.5)
        resize_width = int(width*scale+0.5)
        if height != resize_height or width != resize_width:
            ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


@offload_reload_decorator
@clear_cache_decorator
def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                sampler_name="DDIM", iteration_count=1):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    set_ia_config(IAConfig.KEYS.INP_MODEL_ID, inp_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, save_mask_chk)

    ia_logging.info(f"Loading model {inp_model_id}")
    config_offline_inpainting = get_webui_setting("inpaint_anything_offline_inpainting", False)
    if config_offline_inpainting:
        ia_logging.info("Run Inpainting on offline network: {}".format(str(config_offline_inpainting)))
    local_files_only = False
    local_file_status = download_model_from_hf(inp_model_id, local_files_only=True)
    if local_file_status != IAFileManager.DOWNLOAD_COMPLETE:
        if config_offline_inpainting:
            ia_logging.warning(local_file_status)
            return
    else:
        local_files_only = True
        ia_logging.info("local_files_only: {}".format(str(local_files_only)))

    if platform.system() == "Darwin" or devices.device == devices.cpu or ia_check_versions.torch_on_amd_rocm:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inp_model_id, torch_dtype=torch_dtype, local_files_only=local_files_only, use_safetensors=True)
    except Exception as e:
        ia_logging.error(str(e))
        if not config_offline_inpainting:
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    inp_model_id, torch_dtype=torch_dtype, use_safetensors=True)
            except Exception as e:
                ia_logging.error(str(e))
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        inp_model_id, torch_dtype=torch_dtype, force_download=True, use_safetensors=True)
                except Exception as e:
                    ia_logging.error(str(e))
                    return
        else:
            return
    pipe.safety_checker = None

    ia_logging.info(f"Using sampler {sampler_name}")
    if sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        ia_logging.info("Sampler fallback to DDIM")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if platform.system() == "Darwin":
        pipe = pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
        pipe.enable_attention_slicing()
        torch_generator = torch.Generator(devices.cpu)
    else:
        if ia_check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
            ia_logging.info("Enable model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(devices.device)
        if shared.xformers_available:
            ia_logging.info("Enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            ia_logging.info("Enable attention slicing")
            pipe.enable_attention_slicing()
        if "privateuseone" in str(getattr(devices.device, "type", "")):
            torch_generator = torch.Generator(devices.cpu)
        else:
            torch_generator = torch.Generator(devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    output_list = []
    iteration_count = iteration_count if iteration_count is not None else 1
    for count in range(int(iteration_count)):
        gc.collect()
        if seed < 0 or count > 0:
            seed = random.randint(0, 2147483647)

        generator = torch_generator.manual_seed(seed)

        pipe_args_dict = {
            "prompt": prompt,
            "image": init_image,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": ddim_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": n_prompt,
            "generator": generator,
        }

        output_image = pipe(**pipe_args_dict).images[0]

        if composite_chk:
            dilate_mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
            output_image = Image.composite(output_image, init_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

        generation_params = {
            "Steps": ddim_steps,
            "Sampler": sampler_name,
            "CFG scale": cfg_scale,
            "Seed": seed,
            "Size": f"{width}x{height}",
            "Model": inp_model_id,
        }

        generation_params_text = ", ".join([k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
        prompt_text = prompt if prompt else ""
        negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
        infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

        metadata = PngInfo()
        metadata.add_text("parameters", infotext)

        save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(inp_model_id), str(seed)]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        output_image.save(save_name, pnginfo=metadata)

        output_list.append(output_image)

        yield output_list, max([1, iteration_count - (count + 1)])


@offload_reload_decorator
@clear_cache_decorator
def run_cleaner(input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None

    save_mask_image(mask_image, cleaner_save_mask_chk)

    ia_logging.info(f"Loading model {cleaner_model_id}")
    if platform.system() == "Darwin":
        model = ModelManager(name=cleaner_model_id, device=devices.cpu)
    else:
        model = ModelManager(name=cleaner_model_id, device=devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    init_image = np.array(init_image)
    mask_image = np.array(mask_image.convert("L"))

    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim
    )

    output_image = model(image=init_image, mask=mask_image, config=config)
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(cleaner_model_id)]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    output_image.save(save_name)

    del model
    return [output_image]


@clear_cache_decorator
def run_get_alpha_image(input_image, sel_mask):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None, ""

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None, ""

    alpha_image = Image.fromarray(input_image).convert("RGBA")
    #mask_image = Image.fromarray(mask_image).convert("L")
    mask_binary = Image.fromarray(mask_image).convert("L")
    # ⚠  Invert so that masked pixels become transparent
    mask_binary = ImageOps.invert(mask_binary)
    alpha_image.putalpha(mask_binary)
    #alpha_image.putalpha(mask_image)

    save_name = "_".join([ia_file_manager.savename_prefix, "rgba_image"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    alpha_image.save(save_name)

    return alpha_image, f"saved: {save_name}"


@clear_cache_decorator
def run_get_mask(sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]
    #mask_image = cv2.bitwise_not(sam_dict["mask_image"])

    save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    Image.fromarray(mask_image).save(save_name)

    return mask_image


@clear_cache_decorator
def run_cn_inpaint(input_image, sel_mask,
                   cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                   cn_module_id, cn_model_id, cn_save_mask_chk,
                   cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count=1,
                   cn_ref_module_id=None, cn_ref_image=None, cn_ref_weight=1.0, cn_ref_mode="Balanced", cn_ref_resize_mode="resize",
                   cn_ipa_or_ref=None, cn_ipa_model_id=None):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    await_pre_reload_model_weights()

    if (shared.sd_model.parameterization == "v" and "sd15" in cn_model_id):
        ia_logging.error("The SDv2 model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD v2 model is not compatible with the ControlNet model")
        yield [ret_image], 1
        return

    if (getattr(shared.sd_model, "is_sdxl", False) and "sd15" in cn_model_id):
        ia_logging.error("The SDXL model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD XL model is not compatible with the ControlNet model")
        yield [ret_image], 1
        return

    cnet = sam_dict.get("cnet", None)
    if cnet is None:
        ia_logging.warning("The ControlNet extension is not loaded")
        return

    save_mask_image(mask_image, cn_save_mask_chk)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    input_mask = None if "inpaint_only" in cn_module_id else mask_image
    p = get_sd_img2img_processing(init_image, input_mask,
                                  cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed)

    backup_alwayson_scripts(p.scripts)
    disable_alwayson_scripts_wo_cn(cnet, p.scripts)

    cn_units = [cnet.to_processing_unit(dict(
        enabled=True,
        module=cn_module_id,
        model=cn_model_id,
        weight=cn_weight,
        image={"image": np.array(init_image), "mask": np.array(mask_image)},
        resize_mode=cnet.ResizeMode.RESIZE,
        low_vram=cn_low_vram_chk,
        processor_res=min(width, height),
        guidance_start=0.0,
        guidance_end=1.0,
        pixel_perfect=True,
        control_mode=cn_mode,
        threshold_a=0.5,
        threshold_b=0.5,
    ))]

    if cn_ref_module_id is not None and cn_ref_image is not None:
        if cn_ref_resize_mode == "tile":
            ref_height, ref_width = cn_ref_image.shape[:2]
            num_h = math.ceil(height / ref_height) if height > ref_height else 1
            num_h = num_h + 1 if (num_h % 2) == 0 else num_h
            num_w = math.ceil(width / ref_width) if width > ref_width else 1
            num_w = num_w + 1 if (num_w % 2) == 0 else num_w
            cn_ref_image = np.tile(cn_ref_image, (num_h, num_w, 1))
            cn_ref_image = transforms.functional.center_crop(Image.fromarray(cn_ref_image), (height, width))
            ia_logging.info(f"Reference image is tiled ({num_h}, {num_w}) times and cropped to ({height}, {width})")
        else:
            cn_ref_image = ImageOps.fit(Image.fromarray(cn_ref_image), (width, height), method=Image.Resampling.LANCZOS)
            ia_logging.info(f"Reference image is resized and cropped to ({height}, {width})")
        assert cn_ref_image.size == init_image.size, "The sizes of the reference image and input image do not match"

        cn_ref_model_id = "None"
        if cn_ipa_or_ref is not None and cn_ipa_model_id is not None:
            cn_ipa_module_ids = [cn for cn in cnet.get_modules() if "ip-adapter" in cn and "sd15" in cn]
            if len(cn_ipa_module_ids) > 0 and cn_ipa_or_ref == "IP-Adapter":
                cn_ref_module_id = cn_ipa_module_ids[0]
                cn_ref_model_id = cn_ipa_model_id

        cn_units.append(cnet.to_processing_unit(dict(
            enabled=True,
            module=cn_ref_module_id,
            model=cn_ref_model_id,
            weight=cn_ref_weight,
            image={"image": np.array(cn_ref_image), "mask": None},
            resize_mode=cnet.ResizeMode.RESIZE,
            low_vram=cn_low_vram_chk,
            processor_res=min(width, height),
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cn_ref_mode,
            threshold_a=0.5,
            threshold_b=0.5,
        )))

    p.script_args = np.zeros(get_controlnet_args_to(cnet, p.scripts)).tolist()
    cnet.update_cn_script_in_processing(p, cn_units)

    no_hash_cn_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", cn_model_id).strip()

    output_list = []
    cn_iteration_count = cn_iteration_count if cn_iteration_count is not None else 1
    for count in range(int(cn_iteration_count)):
        gc.collect()
        if cn_seed < 0 or count > 0:
            cn_seed = random.randint(0, 2147483647)

        p.init_images = [init_image]
        p.seed = cn_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            clear_controlnet_cache(cnet, p.scripts)
            restore_alwayson_scripts(p.scripts)
            yield [ret_image], 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_cn_model_id), str(cn_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            output_list.append(output_image)

            yield output_list, max([1, cn_iteration_count - (count + 1)])

    clear_controlnet_cache(cnet, p.scripts)
    restore_alwayson_scripts(p.scripts)


@clear_cache_decorator
def run_webui_inpaint(input_image, sel_mask,
                      webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                      webui_model_id, webui_save_mask_chk,
                      webui_mask_blur, webui_fill_mode, webui_iteration_count=1,
                      webui_enable_refiner_chk=False, webui_refiner_checkpoint="", webui_refiner_switch_at=0.8):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    info = get_closet_checkpoint_match(webui_model_id)
    if info is None:
        ia_logging.error(f"No model found: {webui_model_id}")
        return

    await_backup_reload_ckpt_info(info=info)

    if not getattr(shared.sd_model, "is_sdxl", False) and "sdxl_vae" in getattr(shared.opts, "sd_vae", ""):
        ia_logging.error("The SDXL VAE is not compatible with the inpainting model")
        ret_image = draw_text_image(
            input_image, "The SDXL VAE is not compatible with the inpainting model")
        yield [ret_image], 1
        return

    set_ia_config(IAConfig.KEYS.INP_WEBUI_MODEL_ID, webui_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, webui_save_mask_chk)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    p = get_sd_img2img_processing(init_image, mask_image,
                                  webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                                  webui_mask_blur, webui_fill_mode)

    backup_alwayson_scripts(p.scripts)
    disable_all_alwayson_scripts(p.scripts)

    p.script_args = np.zeros(get_max_args_to(p.scripts)).tolist()

    if ia_check_versions.webui_refiner_is_available and webui_enable_refiner_chk:
        p.refiner_checkpoint = webui_refiner_checkpoint
        p.refiner_switch_at = webui_refiner_switch_at

    no_hash_webui_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", webui_model_id).strip()
    no_hash_webui_model_id = os.path.splitext(no_hash_webui_model_id)[0]

    output_list = []
    webui_iteration_count = webui_iteration_count if webui_iteration_count is not None else 1
    for count in range(int(webui_iteration_count)):
        gc.collect()
        if webui_seed < 0 or count > 0:
            webui_seed = random.randint(0, 2147483647)

        p.init_images = [init_image]
        p.seed = webui_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            restore_alwayson_scripts(p.scripts)
            yield [ret_image], 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_webui_model_id), str(webui_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            output_list.append(output_image)

            yield output_list, max([1, webui_iteration_count - (count + 1)])

    restore_alwayson_scripts(p.scripts)


def on_ui_tabs():
    global sam_dict

    setup_ia_config_ini()
    sampler_names = get_sampler_names()
    sam_model_ids = get_sam_model_ids()
    sam_model_index = get_ia_config_index(IAConfig.KEYS.SAM_MODEL_ID, IAConfig.SECTIONS.USER)
    inp_model_ids = get_inp_model_ids()
    inp_model_index = get_ia_config_index(IAConfig.KEYS.INP_MODEL_ID, IAConfig.SECTIONS.USER)
    cleaner_model_ids = get_cleaner_model_ids()
    padding_mode_names = get_padding_mode_names()
    sam_dict["cnet"] = find_controlnet()

    cn_enabled = False
    if sam_dict["cnet"] is not None:
        cn_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "inpaint" in cn]
        cn_module_index = cn_module_ids.index("inpaint_only") if "inpaint_only" in cn_module_ids else 0

        cn_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "inpaint" in cn]
        cn_modes = [mode.value for mode in sam_dict["cnet"].ControlMode]

        if len(cn_module_ids) > 0 and len(cn_model_ids) > 0:
            cn_enabled = True

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        cn_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        cn_sampler_ids = ["DDIM"]
    cn_sampler_index = cn_sampler_ids.index("DDIM") if "DDIM" in cn_sampler_ids else 0

    cn_ref_only = False
    try:
        if cn_enabled and sam_dict["cnet"].get_max_models_num() > 1:
            cn_ref_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "reference" in cn]
            if len(cn_ref_module_ids) > 0:
                cn_ref_only = True
    except AttributeError:
        pass

    cn_ip_adapter = False
    if cn_ref_only:
        cn_ipa_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "ip-adapter" in cn and "sd15" in cn]
        cn_ipa_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "ip-adapter" in cn and "sd15" in cn]

        if len(cn_ipa_module_ids) > 0 and len(cn_ipa_model_ids) > 0:
            cn_ip_adapter = True

    webui_inpaint_enabled = False
    webui_model_ids = get_inp_webui_model_ids()
    if len(webui_model_ids) > 0:
        webui_inpaint_enabled = True
        webui_model_index = get_ia_config_index(IAConfig.KEYS.INP_WEBUI_MODEL_ID, IAConfig.SECTIONS.USER)

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        webui_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        webui_sampler_ids = ["DDIM"]
    webui_sampler_index = webui_sampler_ids.index("DDIM") if "DDIM" in webui_sampler_ids else 0

    # Gradio 4 gallery parameters used across multiple components
    out_gallery_kwargs = dict(columns=2, height=480, object_fit="contain", preview=True, allow_preview=True)

    with gr.Blocks(analytics_enabled=False, theme=gr.themes.Default()) as inpaint_anything_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        sam_model_id = gr.Dropdown(label="Segment Anything Model ID", elem_id="sam_model_id", choices=sam_model_ids,
                                                   value=sam_model_ids[sam_model_index], show_label=True)
                    with gr.Column():
                        with gr.Row():
                            load_model_btn = gr.Button("Download model", elem_id="load_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", elem_id="status_text", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="ia_input_image", source="upload", type="numpy", interactive=True)

                with gr.Row():
                    with gr.Accordion("Padding options", elem_id="padding_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                pad_scale_width = gr.Slider(label="Scale Width", elem_id="pad_scale_width", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_lr_barance = gr.Slider(label="Left/Right Balance", elem_id="pad_lr_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                pad_scale_height = gr.Slider(label="Scale Height", elem_id="pad_scale_height", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_tb_barance = gr.Slider(label="Top/Bottom Balance", elem_id="pad_tb_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                padding_mode = gr.Dropdown(label="Padding Mode", elem_id="padding_mode", choices=padding_mode_names, value="edge")
                            with gr.Column():
                                padding_btn = gr.Button("Run Padding", elem_id="padding_btn")

                with gr.Row():
                    with gr.Column():
                        anime_style_chk = gr.Checkbox(label="Anime Style (Up Detection, Down mask Quality)", elem_id="anime_style_chk",
                                                      show_label=True, interactive=True)
                    with gr.Column():
                        sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn", variant="primary", interactive=False)

                with gr.Tab("Inpainting", elem_id="inpainting_tab"):
                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_sd_prompt")
                            n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_sd_n_prompt")
                        with gr.Column(scale=0, min_width=128):
                            gr.Markdown("Get prompt from:")
                            get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="get_txt2img_prompt_btn")
                            get_img2img_prompt_btn = gr.Button("img2img", elem_id="get_img2img_prompt_btn")
                    with gr.Accordion("Advanced options", elem_id="inp_advanced_options", open=False):
                        composite_chk = gr.Checkbox(label="Mask area Only", elem_id="composite_chk", value=True, show_label=True, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                sampler_name = gr.Dropdown(label="Sampler", elem_id="sampler_name", choices=sampler_names,
                                                           value=sampler_names[0], show_label=True)
                            with gr.Column():
                                ddim_steps = gr.Slider(label="Sampling Steps", elem_id="ddim_steps", minimum=1, maximum=100, value=20, step=1)
                        cfg_scale = gr.Slider(label="Guidance Scale", elem_id="cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                        seed = gr.Slider(
                            label="Seed",
                            elem_id="sd_seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                        )
                    with gr.Row():
                        with gr.Column():
                            inp_model_id = gr.Dropdown(label="Inpainting Model ID", elem_id="inp_model_id",
                                                       choices=inp_model_ids, value=inp_model_ids[inp_model_index], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                inpaint_btn = gr.Button("Run Inpainting", elem_id="inpaint_btn", variant="primary")
                            with gr.Row():
                                save_mask_chk = gr.Checkbox(label="Save mask", elem_id="save_mask_chk",
                                                            value=False, show_label=False, interactive=False, visible=False)
                                iteration_count = gr.Slider(label="Iterations", elem_id="iteration_count", minimum=1, maximum=10, value=1, step=1)

                    with gr.Row():
                        # Use only the modern Gradio 4 Gallery syntax
                        out_image = gr.Gallery(label="Inpainted image", elem_id="ia_out_image", show_label=False,
                                               **out_gallery_kwargs)

                with gr.Tab("Cleaner", elem_id="cleaner_tab"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_id = gr.Dropdown(label="Cleaner Model ID", elem_id="cleaner_model_id",
                                                           choices=cleaner_model_ids, value=cleaner_model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                cleaner_btn = gr.Button("Run Cleaner", elem_id="cleaner_btn", variant="primary")
                            with gr.Row():
                                cleaner_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cleaner_save_mask_chk",
                                                                    value=False, show_label=False, interactive=False, visible=False)

                    with gr.Row():
                        # Use only the modern Gradio 4 Gallery syntax
                        cleaner_out_image = gr.Gallery(label="Cleaned image", elem_id="ia_cleaner_out_image", show_label=False,
                                                       **out_gallery_kwargs)

                if webui_inpaint_enabled:
                    with gr.Tab("Inpainting webui", elem_id="webui_inpainting_tab"):
                        with gr.Row():
                            with gr.Column():
                                webui_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_webui_sd_prompt")
                                webui_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_webui_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                webui_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="webui_get_txt2img_prompt_btn")
                                webui_get_img2img_prompt_btn = gr.Button("img2img", elem_id="webui_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="webui_advanced_options", open=False):
                            webui_mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id="webui_mask_blur")
                            webui_fill_mode = gr.Radio(label="Masked content", elem_id="webui_fill_mode",
                                                       choices=["fill", "original", "latent noise", "latent nothing"], value="original", type="index")
                            with gr.Row():
                                with gr.Column():
                                    webui_sampler_id = gr.Dropdown(label="Sampling method webui", elem_id="webui_sampler_id",
                                                                   choices=webui_sampler_ids, value=webui_sampler_ids[webui_sampler_index], show_label=True)
                                with gr.Column():
                                    webui_ddim_steps = gr.Slider(label="Sampling steps webui", elem_id="webui_ddim_steps",
                                                                 minimum=1, maximum=150, value=30, step=1)
                            webui_cfg_scale = gr.Slider(label="Guidance scale webui", elem_id="webui_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            webui_strength = gr.Slider(label="Denoising strength webui", elem_id="webui_strength",
                                                       minimum=0.0, maximum=1.0, value=0.75, step=0.01)
                            webui_seed = gr.Slider(
                                label="Seed",
                                elem_id="webui_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        if ia_check_versions.webui_refiner_is_available:
                            with gr.Accordion("Refiner options", elem_id="webui_refiner_options", open=False):
                                with gr.Row():
                                    webui_enable_refiner_chk = gr.Checkbox(label="Enable Refiner", elem_id="webui_enable_refiner_chk",
                                                                           value=False, show_label=True, interactive=True)
                                with gr.Row():
                                    webui_refiner_checkpoint = gr.Dropdown(label="Refiner Model ID", elem_id="webui_refiner_checkpoint",
                                                                           choices=shared.list_checkpoint_tiles(), value="")
                                    webui_refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01,
                                                                        elem_id="webui_refiner_switch_at")

                        with gr.Row():
                            with gr.Column():
                                webui_model_id = gr.Dropdown(label="Inpainting Model ID webui", elem_id="webui_model_id",
                                                             choices=webui_model_ids, value=webui_model_ids[webui_model_index], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    webui_inpaint_btn = gr.Button("Run Inpainting", elem_id="webui_inpaint_btn", variant="primary")
                                with gr.Row():
                                    webui_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="webui_save_mask_chk",
                                                                      value=False, show_label=False, interactive=False, visible=False)
                                    webui_iteration_count = gr.Slider(label="Iterations", elem_id="webui_iteration_count",
                                                                      minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            # Use only the modern Gradio 4 Gallery syntax
                            webui_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_webui_out_image", show_label=False,
                                                         **out_gallery_kwargs)

                with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                    if cn_enabled:
                        with gr.Row():
                            with gr.Column():
                                cn_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_cn_sd_prompt")
                                cn_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_cn_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                cn_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="cn_get_txt2img_prompt_btn")
                                cn_get_img2img_prompt_btn = gr.Button("img2img", elem_id="cn_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="cn_advanced_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_sampler_id = gr.Dropdown(label="Sampling method", elem_id="cn_sampler_id",
                                                                choices=cn_sampler_ids, value=cn_sampler_ids[cn_sampler_index], show_label=True)
                                with gr.Column():
                                    cn_ddim_steps = gr.Slider(label="Sampling steps", elem_id="cn_ddim_steps", minimum=1, maximum=150, value=30, step=1)
                            cn_cfg_scale = gr.Slider(label="Guidance scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising strength", value=0.75, elem_id="cn_strength")
                            cn_seed = gr.Slider(
                                label="Seed",
                                elem_id="cn_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        with gr.Accordion("ControlNet options", elem_id="cn_cn_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_low_vram_chk = gr.Checkbox(label="Low VRAM", elem_id="cn_low_vram_chk", value=True, show_label=True, interactive=True)
                                    cn_weight = gr.Slider(label="Control Weight", elem_id="cn_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                with gr.Column():
                                    cn_mode = gr.Dropdown(label="Control Mode", elem_id="cn_mode", choices=cn_modes, value=cn_modes[-1], show_label=True)

                            if cn_ref_only:
                                with gr.Row():
                                    with gr.Column():
                                        cn_md_text = "Reference Control (enabled with image below)"
                                        if not cn_ip_adapter:
                                            cn_md_text = cn_md_text + ("<br><span style='color: gray;'>"
                                                                       "[IP-Adapter](https://huggingface.co/lllyasviel/sd_control_collection/tree/main) "
                                                                       "is not available. Reference-Only is used.</span>")
                                        gr.Markdown(cn_md_text)
                                        if cn_ip_adapter:
                                            cn_ipa_or_ref = gr.Radio(label="IP-Adapter or Reference-Only", elem_id="cn_ipa_or_ref",
                                                                     choices=["IP-Adapter", "Reference-Only"], value="IP-Adapter", show_label=False)
                                        cn_ref_image = gr.Image(label="Reference Image", elem_id="cn_ref_image", source="upload", type="numpy",
                                                                interactive=True)
                                    with gr.Column():
                                        cn_ref_resize_mode = gr.Radio(label="Reference Image Resize Mode", elem_id="cn_ref_resize_mode",
                                                                      choices=["resize", "tile"], value="resize", show_label=True)
                                        if cn_ip_adapter:
                                            cn_ipa_model_id = gr.Dropdown(label="IP-Adapter Model ID", elem_id="cn_ipa_model_id",
                                                                          choices=cn_ipa_model_ids, value=cn_ipa_model_ids[0], show_label=True)
                                        cn_ref_module_id = gr.Dropdown(label="Reference Type for Reference-Only", elem_id="cn_ref_module_id",
                                                                       choices=cn_ref_module_ids, value=cn_ref_module_ids[-1], show_label=True)
                                        cn_ref_weight = gr.Slider(label="Reference Control Weight", elem_id="cn_ref_weight",
                                                                  minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                        cn_ref_mode = gr.Dropdown(label="Reference Control Mode", elem_id="cn_ref_mode",
                                                                  choices=cn_modes, value=cn_modes[0], show_label=True)
                            else:
                                with gr.Row():
                                    gr.Markdown("The Multi ControlNet setting is currently set to 1.<br>"
                                                "If you wish to use the Reference-Only Control, "
                                                "please adjust the Multi ControlNet setting to 2 or more and restart the Web UI.")

                        with gr.Row():
                            with gr.Column():
                                cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id",
                                                           choices=cn_module_ids, value=cn_module_ids[cn_module_index], show_label=True)
                                cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id",
                                                          choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    cn_inpaint_btn = gr.Button("Run ControlNet Inpaint", elem_id="cn_inpaint_btn", variant="primary")
                                with gr.Row():
                                    cn_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cn_save_mask_chk",
                                                                   value=False, show_label=False, interactive=False, visible=False)
                                    cn_iteration_count = gr.Slider(label="Iterations", elem_id="cn_iteration_count",
                                                                   minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            # Use only the modern Gradio 4 Gallery syntax
                            cn_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_cn_out_image", show_label=False,
                                                      **out_gallery_kwargs)

                    else:
                        if sam_dict["cnet"] is None:
                            gr.Markdown("ControlNet extension is not available.<br>"
                                        "Requires the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension.")
                        elif len(cn_module_ids) > 0:
                            cn_models_directory = os.path.join("extensions", "sd-webui-controlnet", "models")
                            gr.Markdown("ControlNet inpaint model is not available.<br>"
                                        "Requires the [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) inpaint model "
                                        f"in the {cn_models_directory} directory.")
                        else:
                            gr.Markdown("ControlNet inpaint preprocessor is not available.<br>"
                                        "The local version of [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension may be old.")

                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_alpha_image_btn = gr.Button("Get mask as alpha of image", elem_id="get_alpha_image_btn")
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")

                    with gr.Row():
                        with gr.Column():
                            alpha_out_image = gr.Image(label="Alpha channel image", elem_id="alpha_out_image", type="pil", image_mode="RGBA", interactive=False)
                        with gr.Column():
                            mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="numpy", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            get_alpha_status_text = gr.Textbox(label="", elem_id="get_alpha_status_text", max_lines=1, show_label=False, interactive=False)
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")

            with gr.Column():
                with gr.Row():
                    gr.Markdown("Mouse over image: Press `S` key for Fullscreen mode, `R` key to Reset zoom")

                # ─── SAM segmentation canvas ─────────────────────────
                with gr.Row():
                    brush_sam = Brush(default_size=8, default_color="black", colors=["black","white"])
                    sam_image = ImageEditor(
                        label="Segment Anything image",
                        elem_id="ia_sam_image",
                        type="numpy",
                        brush=Brush(               # ← give it any Brush() to expose the tool
                            default_size=8,
                            default_color="white",
                            colors=["white", "black", "red", "green", "blue"]
                            ),
                        show_label=False,
                        interactive=True,
                        height=CANVAS_H + 2*PAD,  # 30px padding on each side
                        canvas_size=(CANVAS_W + 2*PAD, CANVAS_H + 2*PAD),  # 30px padding on each side
                        image_mode="RGBA"
                    )

                # ─── Mask controls ────────────────────────────────────
                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create Mask", elem_id="select_btn", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)
                            ignore_black_chk = gr.Checkbox(label="Ignore black area", elem_id="ignore_black_chk", value=True, show_label=True, interactive=True)

                # ─── Mask preview canvas ─────────────────────────────
                with gr.Row():
                    brush_sel = Brush(default_size=12, default_color="black", colors=["black","white"])
                    sel_mask = ImageEditor(
                        label="Selected mask image",
                        elem_id="ia_sel_mask",
                        type="numpy",
                        brush=brush_sel,
                        show_label=False,
                        interactive=True,
                        height=CANVAS_H + 2*PAD,  # 30px padding on each side
                        canvas_size=(CANVAS_W + 2*PAD, CANVAS_H + 2*PAD),  # 30px padding on each side
                        image_mode="RGBA"
                    )

                with gr.Row():
                    with gr.Column():
                        expand_mask_btn = gr.Button("Expand mask region", elem_id="expand_mask_btn")
                        expand_mask_iteration_count = gr.Slider(label="Expand Mask Iterations",
                                                                elem_id="expand_mask_iteration_count", minimum=1, maximum=100, value=1, step=1)
                    with gr.Column():
                        apply_mask_btn = gr.Button("Trim mask by sketch", elem_id="apply_mask_btn")
                        add_mask_btn = gr.Button("Add mask by sketch", elem_id="add_mask_btn")

            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            # Main workflow events
            # When input image changes, update both image editors with properly resized images
            def prepare_images_for_editors(img):
                if img is None:
                    return None, None, gr.update(interactive=False)
                
                # Store original image for processing
                sam_dict["orig_image"] = img
                
                # Create empty mask of the same size
                if sam_dict["mask_image"] is None or sam_dict["mask_image"].shape != img.shape:
                    sam_dict["mask_image"] = np.zeros_like(img, dtype=np.uint8)
                
                # Resize the image to fit the editor while maintaining aspect ratio
                # Always resize to ensure consistent behavior across all editors
                resized_img = resize_image_to_fit(img)
                
                # Create a resized mask for overlay
                resized_mask = np.zeros_like(resized_img, dtype=np.uint8)
                
                # Overlay for sel_mask - use resized versions for display
                overlay = cv2.addWeighted(resized_img, 0.5, resized_mask, 0.5, 0)
                
                # We return the same sized images for both editors to ensure consistency
                return resized_img, resized_img, gr.update(interactive=True)
            
            # Define a JavaScript function to ensure all canvases are properly resized
            resize_js = """
function() {
    // Force a resize event to ensure all canvases are properly updated
    setTimeout(function() {
        window.dispatchEvent(new Event('resize'));
    }, 100);
    return [];
}
"""
            
            # Use the same function for both change and upload events
            input_image.change(
                fn=prepare_images_for_editors,
                inputs=[input_image],
                outputs=[sam_image, sel_mask, sam_btn]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_initSamSelMask"
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js=resize_js
            )
            
            input_image.upload(
                fn=prepare_images_for_editors,
                inputs=[input_image],
                outputs=[sam_image, sel_mask, sam_btn]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_initSamSelMask"
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js=resize_js
            )
            
            padding_btn.click(
                run_padding,
                inputs=[input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode],
                outputs=[input_image, status_text]
            )
            
            sam_btn.click(
                run_sam,
                inputs=[input_image, sam_model_id, sam_image, anime_style_chk],
                outputs=[sam_image, status_text]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_clearSamMask"
            )
            
            select_btn.click(
                select_mask,
                inputs=[input_image, sam_image, invert_chk, ignore_black_chk, sel_mask],
                outputs=[sel_mask]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_clearSelMask"
            )
            
            expand_mask_btn.click(
                expand_mask,
                inputs=[input_image, sel_mask, expand_mask_iteration_count],
                outputs=[sel_mask]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_clearSelMask"
            )
            
            apply_mask_btn.click(
                apply_mask,
                inputs=[input_image, sel_mask],
                outputs=[sel_mask]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_clearSelMask"
            )
            
            add_mask_btn.click(
                add_mask,
                inputs=[input_image, sel_mask],
                outputs=[sel_mask]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                _js="inpaintAnything_clearSelMask"
            )
            get_txt2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getTxt2imgPrompt")
            get_img2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getImg2imgPrompt")

            inpaint_btn.click(
                run_inpaint,
                inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                        sampler_name, iteration_count],
                outputs=[out_image, iteration_count])
            cleaner_btn.click(
                run_cleaner,
                inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk],
                outputs=[cleaner_out_image])
            get_alpha_image_btn.click(
                run_get_alpha_image,
                inputs=[input_image, sel_mask],
                outputs=[alpha_out_image, get_alpha_status_text])
            get_mask_btn.click(
                run_get_mask,
                inputs=[sel_mask],
                outputs=[mask_out_image])
            mask_send_to_inpaint_btn.click(
                fn=None,
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)
            if cn_enabled:
                cn_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetTxt2imgPrompt")
                cn_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetImg2imgPrompt")
            if cn_enabled:
                cn_inputs = [input_image, sel_mask,
                             cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                             cn_module_id, cn_model_id, cn_save_mask_chk,
                             cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count]
                if cn_ref_only:
                    cn_inputs.extend([cn_ref_module_id, cn_ref_image, cn_ref_weight, cn_ref_mode, cn_ref_resize_mode])
                if cn_ip_adapter:
                    cn_inputs.extend([cn_ipa_or_ref, cn_ipa_model_id])
                cn_inpaint_btn.click(
                    run_cn_inpaint,
                    inputs=cn_inputs,
                    outputs=[cn_out_image, cn_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)
            if webui_inpaint_enabled:
                webui_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetTxt2imgPrompt")
                webui_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetImg2imgPrompt")
                wi_inputs = [input_image, sel_mask,
                             webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                             webui_model_id, webui_save_mask_chk,
                             webui_mask_blur, webui_fill_mode, webui_iteration_count]
                if ia_check_versions.webui_refiner_is_available:
                    wi_inputs.extend([webui_enable_refiner_chk, webui_refiner_checkpoint, webui_refiner_switch_at])
                webui_inpaint_btn.click(
                    run_webui_inpaint,
                    inputs=wi_inputs,
                    outputs=[webui_out_image, webui_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)

    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]


def on_ui_settings():
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option("inpaint_anything_save_folder",
                           shared.OptionInfo(
                               default="inpaint-anything",
                               label="Folder name where output images will be saved",
                               component=gr.Radio,
                               component_args={"choices": ["inpaint-anything", "img2img-images (img2img output setting of web UI)"]},
                               section=section))
    shared.opts.add_option("inpaint_anything_sam_oncpu",
                           shared.OptionInfo(
                               default=False,
                               label="Run Segment Anything on CPU",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_offline_inpainting",
                           shared.OptionInfo(
                               default=False,
                               label="Run Inpainting on offline network (Models not auto-downloaded)",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_padding_fill",
                           shared.OptionInfo(
                               default=127,
                               label="Fill value used when Padding is set to constant",
                               component=gr.Slider,
                               component_args={"minimum": 0, "maximum": 255, "step": 1},
                               section=section))
    shared.opts.add_option("inpain_anything_sam_models_dir",
                           shared.OptionInfo(
                               default="",
                               label="Segment Anything Models Directory; If empty, defaults to [Inpaint Anything extension folder]/models",
                               component=gr.Textbox,
                               component_args={"interactive": True},
                               section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
