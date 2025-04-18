from __future__ import annotations
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

def _load_font(family: str | None, size: int, bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
    """
    Very small helper that tries to find a reasonable TTF on the system.
    For real projects you’d map (family, bold, italic) → specific font files.
    """
    if family is None:
        family = "arial.ttf"            # Windows & macOS
    try:
        return ImageFont.truetype(family, size)
    except OSError:
        # last‑ditch fallback: Pillow’s built‑in bitmap font
        return ImageFont.load_default()

_supports_direction = features.check("raqm")



class ImageSaveRenderer:
    def __init__(self):
        self._text_blocks: List[Dict[str, Any]] = []

    def _draw_text_block(self, canvas: Image.Image, block: Dict[str, Any], sf: int) -> None:
        """
        Renders a single text block onto *canvas* (a Pillow Image).
        • Handles line spacing, alignment, outline, rotation, scaling.
        • Ignores features that don’t make sense outside Qt (selection outlines, transform origin).
        """
        text = block["text"]
        font_size = int(block.get("font_size", 20) * sf * block.get("scale", 1.0))
        font_family = block.get("font_family")
        bold = block.get("bold", False)
        italic = block.get("italic", False)
        underline = block.get("underline", False)  # underline is ignored here
        align = block.get("alignment", "left")  # 'left'|'center'|'right'
        line_sp = block.get("line_spacing", 1.0)
        direction = block.get("direction", "ltr")
        outline_w = int(block.get("outline_width", 0) * sf)
        outline_col = tuple(block.get("outline_color", (0, 0, 0)))
        fill_col = tuple(block.get("text_color", (255, 255, 255)))
        rotation = block.get("rotation", 0.0)
        pos_x, pos_y = [int(v * sf) for v in block.get("position", (0, 0))]

        # Prepare a transparent layer for the text (makes rotation easier)
        font = _load_font(font_family, font_size, bold, italic)
        draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = draw_dummy.multiline_textbbox((0, 0), text, font=font, spacing=(line_sp - 1) * font_size)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        layer = Image.new("RGBA", (int(text_w + 2 * outline_w), int(text_h + 2 * outline_w)), (0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Alignment offset inside the layer
        offset_x = {
            "left": 0,
            "center": (layer.width - text_w) // 2,
            "right": layer.width - text_w,
        }.get(align, 0)
        offset_y = 0

        # # Outline (simple 8‑direction stroke)
        # if outline_w > 0:
        #     for dx in range(-outline_w, outline_w + 1):
        #         for dy in range(-outline_w, outline_w + 1):
        #             if dx * dx + dy * dy <= outline_w * outline_w:
        #                 draw.multiline_text(
        #                     (offset_x + dx, offset_y + dy),
        #                     text,
        #                     font=font,
        #                     fill=outline_col + (255,),
        #                     spacing=(line_sp - 1) * font_size,
        #                     align=align,
        #                     # direction=direction,
        #                 )

        # Main fill
        draw.multiline_text(
            (offset_x, offset_y),
            text,
            font=font,
            fill=fill_col + (255,),
            spacing=(line_sp - 1) * font_size,
            align=align,
            # direction=direction,
        )

        # Rotate if needed
        if rotation:
            layer = layer.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)

        # Paste onto the big canvas
        canvas.alpha_composite(layer, dest=(pos_x - layer.width // 2, pos_y - layer.height // 2))

    def add_state_to_image(self, state):
        """Collect text blocks for later rendering."""
        self._text_blocks.extend(state.get("text_items_state", []))

    def render_to_image(self, base_rgb, scale_factor: int = 1):
        """
        Draw all collected text blocks at *scale_factor*× resolution,
        then downsample for smoother edges.
        Returns an **OpenCV BGR** image.
        """
        # 1. Upscale base image for nicer antialiasing
        h, w = base_rgb.shape[:2]
        big = Image.fromarray(base_rgb).resize(
            (w * scale_factor, h * scale_factor),
            resample=Image.Resampling.LANCZOS,
        )

        # 2. Draw each block
        for block in self._text_blocks:
            self._draw_text_block(big, block, scale_factor)

        # 3. Downscale back to original size
        small = big.resize((w, h), resample=Image.Resampling.LANCZOS)

        # 4. Convert back to BGR for OpenCV callers
        return cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGRA)

    def save_image(self, output_path):
        final_image = self.render_to_image()
        cv2.imwrite(output_path, final_image)

