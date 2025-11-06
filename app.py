import gradio as gr
from PIL import Image
import numpy as np
import cv2, os, time
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Any, Optional, Tuple
from ultralytics import SAM

def load_sam_int8(model_path="mobilesam.pt"):
    model = SAM("mobile_sam.pt") 
    quant_model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    return quant_model

model = load_sam_int8("mobile_sam.pt")

def to_pil(img: Any) -> Optional[Image.Image]:
    """Convert numpy array / PIL Image / filepath / bytes to PIL.Image or return None."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    try:
        import numpy as _np
        if isinstance(img, _np.ndarray):
            if img.dtype in (np.float32, np.float64):
                arr = (255 * np.clip(img, 0, 1)).astype("uint8")
            else:
                arr = img
            return Image.fromarray(arr)
    except Exception:
        pass
    if isinstance(img, (bytes, bytearray)):
        try:
            from io import BytesIO
            return Image.open(BytesIO(img)).convert("RGBA")
        except Exception:
            return None
    if isinstance(img, str):
        try:
            return Image.open(img)
        except Exception:
            return None
    return None

def extract_image_and_mask(value: Any) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Extract (image_pil, mask_pil) from common Gradio ImageMask return formats:
      - dict with keys 'background','layers','composite' (Gradio's painting format)
      - dicts with 'image'/'mask' variants
      - tuples/lists like (image, mask, ...)
      - single image
    """
    if value is None:
        return None, None

    if isinstance(value, dict):
        image = None
        mask = None

        for k in ("image", "img", "input_image", "original", "background", "composite"):
            if k in value:
                image = value[k]
                break

        for k in ("mask", "masks", "annotation", "annotations"):
            if k in value:
                mask = value[k]
                break

        layers = value.get("layers")
        if mask is None and layers is not None and isinstance(layers, (list, tuple)):
            for layer in layers:
                if isinstance(layer, dict):
                    cand = None
                    for subk in ("mask", "image", "img", "composite", "annotation", "layer_image"):
                        if subk in layer:
                            cand = layer[subk]
                            break
                    if cand is None:
                        cand = layer.get("data") or layer.get("buffer")
                    layer_img = to_pil(cand)
                else:
                    layer_img = to_pil(layer)

                if layer_img is None:
                    continue

                if getattr(layer_img, "mode", None) == "RGBA":
                    alpha = layer_img.split()[-1]
                    if np.any(np.array(alpha) > 0):
                        mask = alpha
                        break

                try:
                    l_gray = layer_img.convert("L")
                    arr = np.array(l_gray)
                    if arr.size > 0 and np.any(arr > 0):
                        mask = l_gray
                        break
                except Exception:
                    continue

        if mask is None and "composite" in value:
            comp = to_pil(value.get("composite"))
            if comp is not None and getattr(comp, "mode", None) == "RGBA":
                alpha = comp.split()[-1]
                if np.any(np.array(alpha) > 0):
                    mask = alpha

        return to_pil(image), to_pil(mask)

    # list/tuple: possibly (image, mask, meta...) or gallery -> take first element
    if isinstance(value, (list, tuple)):
        v = list(value)
        if len(v) > 0 and isinstance(v[0], (list, tuple)):
            v = list(v[0])

        if len(v) >= 2:
            image_raw = v[0]
            mask_raw = v[1]
            if isinstance(mask_raw, (list, tuple)):
                found = None
                for m in mask_raw:
                    if isinstance(m, Image.Image) or isinstance(m, np.ndarray) or isinstance(m, (bytes, str)):
                        found = m
                        break
                mask_raw = found
            return to_pil(image_raw), to_pil(mask_raw)

        if len(v) == 1:
            return to_pil(v[0]), None

    # fallback: single image-like
    return to_pil(value), None

def tensor_to_numpy(x):
    try:
        return x.cpu().numpy()
    except Exception:
        import numpy as _np
        return _np.array(x)
    
def save_segment(image_pil, mask, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    mask_u8 = (mask*255).astype('uint8') if mask.dtype != np.uint8 else mask
    # create RGBA
    rgba = image_pil.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_u8))
    out_path = os.path.join(out_dir, f"segment_{int(time.time())}.png")
    rgba.save(out_path)
    return out_path

def compute_bbox(value):
    image, mask = extract_image_and_mask(value)

    if image is None:
        return None, "No image uploaded."

    if mask is None:
        return image, "No mask painted."
    
    try:
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))
        if mask.mode != "L":
            mask = mask.convert("L")
    except Exception:
        return image, "Mask conversion failed."

    mask_arr = np.array(mask)
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        return image, f"(0, 0, {image.width}, {image.height})"

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    print("extracted co-ords. Starting results...")
    cropped = image.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    #return cropped, f"({int(x0)}, {int(y0)}, {int(x1)}, {int(y1)})"

    results = model(np.array(image), bboxes=[int(x0), int(y0), int(x1) + 1, int(y1) + 1],imgsz=512)
    #results = model(cropped)
    masks = results[0].masks  
    if masks is not None:
        md = masks.data  
        if hasattr(md, "cpu"):
            md = md.cpu().numpy()
        # pick first mask if multiple
        if md.ndim == 3:
            full_mask = md[0]
        elif md.ndim == 2:
            full_mask = md
        else:
            print(f"Unexpected mask shape: {md.shape}")
            return cropped, None
    else:
        print("No mask found.")
        return cropped, None
    H_full, W_full = full_mask.shape
    x0i, y0i = max(0, int(x0)), max(0, int(y0))
    x1i, y1i = min(W_full, int(x1) + 1), min(H_full, int(y1) + 1)

    mask_cropped = full_mask[y0i:y1i, x0i:x1i] 
    img_bgr = results[0].plot()  
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    out_path = save_segment(cropped, mask_cropped)
    print(f"Saved mask to {out_path}")
    return img_pil

with gr.Blocks() as demo:
    gr.Markdown("## Upload an image and paint over the object (brush mask)")

    img_mask = gr.ImageMask(
        label="Upload and Paint Mask",
        interactive=True,
        type="pil",
        image_mode="RGB"
    )

    output_img_seg = gr.Image(label="Segmented Image")
    #output_bbox = gr.Textbox(label="Bounding Box Coordinates")
    #outputs=[output_img, output_bbox]
    btn = gr.Button("Compute Segment")
    btn.click(fn=compute_bbox, inputs=[img_mask], outputs=[output_img_seg])

demo.launch()
