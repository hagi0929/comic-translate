import json
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageDraw
from PIL import Image
from PIL import ImageFont

from controller import ComicTranslate
import os
from typing import List, Union, Optional

import cv2, shutil
import tempfile
import numpy as np
import copy
from datetime import datetime

from app.ui.settings.settings_page import SettingsPage
from modules.detection.processor import TextBlockDetector
from modules.detection.utils.general import do_rectangles_overlap, get_inpaint_bboxes
from modules.inpainting.aot import AOT
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.ocr.processor import OCRProcessor
from modules.translation.processor import Translator
from modules.utils.text_item import OutlineInfo, OutlineType
from modules.utils.textblock import TextBlock
from modules.utils.file_handler import FileHandler
from modules.utils.pipeline_utils import font_selected, validate_settings, get_layout_direction, \
    validate_ocr, validate_translator, get_language_code, get_config, generate_mask, inpaint_map
from modules.utils.archives import make
from modules.utils.download import get_models, mandatory_models
from modules.utils.translator_utils import format_translations, is_there_text, get_raw_text, get_raw_translation
from modules.rendering.render import TextRenderingSettings, get_best_render_area, pyside_word_wrap
from pipeline import ComicTranslatePipeline

def get_image_path(folder_path: str):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw) -> List[str]:
    """
    Wrap `text` into lines that fit within `max_width` pixels given `font` and `draw`.
    """
    words = text.split()
    lines: List[str] = []
    if not words:
        return lines
    line = words[0]
    for word in words[1:]:
        test_line = f"{line} {word}"
        width, _ = draw.textsize(test_line, font=font)
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines

def load_image(file_path: str):
    cv2_image = cv2.imread(file_path)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA)

    if cv2_image is None:
        raise ValueError(f"Could not load image: {file_path}")
    return cv2_image

@dataclass
class ImageObj:
    def __init__(self, path):
        self.raw_data = None
        self.path = path
        self.cleaned_image = {}
        self.translated_image = {}
        self.blk_list = []
        self.status = "not loaded"

class Logger:
    def __init__(self, image_paths: List[str]):
        self.status = {}
        self.logs = []
        for image_path in image_paths:
            self.status[image_path] = "not loaded"

    def log(self, message):
        self.logs.append(message)
        print(message)

    def update_status(self, image_path, status):
        self.status[image_path] = status
        # current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"{current_time}: Updated status for {image_path}: {status}")

    def get_logs(self):
        return self.logs


class TranslatePipeline:
    def __init__(self, main_page):
        self.main_page = main_page
        self.block_detector_cache = None
        self.inpainter_cache = None
        self.cached_inpainter_key = None
        self.ocr = OCRProcessor()
        # TODO: try to update this to not need the settings page
        self.detector = TextBlockDetector(self.main_page.settings_page)
        self.rectangles_map = {}
        self.ocr.initialize(self.main_page, self.main_page.source_lang)
        self.cached_inpainter_key = None
        device = 'cuda' if self.main_page.settings_page.is_gpu_enabled() else 'cpu'
        inpainter_key = self.main_page.settings_page.get_tool_selection('inpainter')
        self.inpainter = LaMa(device)
        self.translator = Translator(self.main_page, self.main_page.source_lang, self.main_page.target_lang)

    def process_image(self, image: ImageObj):
        # detect text blocks
        self.detect_text_block(image)
        # ocr
        self.ocr_image(image)
        # clean image
        self.clean_image(image)
        # translate
        self.translate_blocks(image)
        # render
        self.render_image(image)

    def translate_blocks(self, image: ImageObj):
        extra_context = self.main_page.settings_page.get_llm_settings()['extra_context']
        image.blk_list = self.translator.translate(image.blk_list, image.raw_data, extra_context)

        entire_raw_text = get_raw_text(image.blk_list)
        entire_translated_text = get_raw_translation(image.blk_list)

        # Parse JSON strings and check if they're empty objects or invalid
        try:
            raw_text_obj = json.loads(entire_raw_text)
            translated_text_obj = json.loads(entire_translated_text)

        except json.JSONDecodeError as e:
            # Handle invalid JSON
            error_message = str(e)
            print(f"Error decoding JSON: {error_message}")

    def render_text_blocks(
            self,
            image_np: np.ndarray,
            blocks: List[TextBlock],
            settings: TextRenderingSettings
    ) -> np.ndarray:
        # Convert numpy array to PIL Image (assume BGR if OpenCV import used)
        if image_np.dtype != np.uint8:
            raise ValueError("Expected image_np with dtype=uint8")
        # Detect BGR vs RGB by channel ordering heuristic (optional)
        # Here we assume BGR if coming from cv2
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for block in blocks:
            # Merge per-block overrides
            min_fs = block.min_font_size if block.min_font_size > 0 else settings.min_font_size
            max_fs = block.max_font_size if block.max_font_size > 0 else settings.max_font_size
            if min_fs > max_fs:
                min_fs, max_fs = settings.min_font_size, settings.max_font_size
            fill_color = block.font_color or settings.color
            outline = settings.outline
            outline_color = settings.outline_color
            outline_width = settings.outline_width
            underline = settings.underline
            line_spacing = settings.line_spacing * block.line_spacing
            alignment = block.alignment.lower()
            direction = settings.direction.lower()

            # Inpaint erase: fill with white for each inpaint bbox
            if block.inpaint_bboxes is not None:
                for ib in block.inpaint_bboxes:
                    x, y, w, h = ib
                    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))

            # Block dimensions
            x1, y1, x2, y2 = block.xyxy
            box_w, box_h = x2 - x1, y2 - y1

            # Find the largest fitting font size
            chosen_font = None
            chosen_size = min_fs
            lines: List[str] = []
            for fs in range(max_fs, min_fs - 1, -1):
                try:
                    font = ImageFont.truetype(settings.font_family, fs)
                except IOError:
                    font = ImageFont.load_default()
                test_lines = wrap_text(block.translation, font, box_w, draw)
                # Total text height
                heights = [settings. for ln in test_lines]
                total_h = sum(heights) + int((len(test_lines) - 1) * fs * (line_spacing - 1))
                if total_h <= box_h:
                    chosen_font = font
                    chosen_size = fs
                    lines = test_lines
                    break
            if chosen_font is None:
                try:
                    chosen_font = ImageFont.truetype(settings.font_family, min_fs)
                except IOError:
                    chosen_font = ImageFont.load_default()
                lines = wrap_text(block.translation, chosen_font, box_w, draw)

            # Vertical centering
            line_heights = [draw.textsize(ln, font=chosen_font)[1] for ln in lines]
            total_text_height = sum(line_heights) + int((len(lines) - 1) * chosen_size * (line_spacing - 1))
            y_offset = y1 + max(0, (box_h - total_text_height) // 2)

            # Draw lines
            for i, ln in enumerate(lines):
                lw, lh = draw.textsize(ln, font=chosen_font)
                # Horizontal alignment
                align = alignment
                if direction == 'rtl':
                    if align == 'left':
                        align = 'right'
                    elif align == 'right':
                        align = 'left'
                if align == 'left':
                    x = x1
                elif align == 'center':
                    x = x1 + (box_w - lw) // 2
                else:
                    x = x2 - lw
                y = y_offset + int(i * chosen_size * line_spacing)

                # Outline or regular text
                if outline and outline_width > 0:
                    draw.text(
                        (x, y), ln,
                        font=chosen_font,
                        fill=fill_color,
                        stroke_width=outline_width,
                        stroke_fill=outline_color
                    )
                else:
                    draw.text((x, y), ln, font=chosen_font, fill=fill_color)

                # Underline
                if underline:
                    underline_y = y + lh + 1
                    draw.line(
                        [(x, underline_y), (x + lw, underline_y)],
                        fill=fill_color,
                        width=max(1, chosen_size // 15)
                    )

        # Convert back to numpy array (BGR)
        result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return result

    def render_image(self, image: Image):
        # TODO: make text rendering settings dynamic
        render_settings = TextRenderingSettings(
            alignment_id = 0,
            font_family = "Noto Sans",
            min_font_size = 25,
            max_font_size = 40,
            color = (0, 0, 0),
            upper_case = False,
            outline = True,
            outline_color = (255, 255, 255),
            outline_width = "1.5",
            bold = True,
            italic = False,
            underline = False,
            line_spacing = "1",
            direction = "direction"
        )
        upper_case = render_settings.upper_case
        outline = render_settings.outline
        trg_lng_cd = get_language_code(self.main_page.target_lang)
        format_translations(image.blk_list, trg_lng_cd, upper_case=upper_case)
        get_best_render_area(image.blk_list, image, image.cleaned_image)

        font = render_settings.font_family
        font_color = render_settings.color

        max_font_size = render_settings.max_font_size
        min_font_size = render_settings.min_font_size
        line_spacing = float(render_settings.line_spacing)
        outline_width = float(render_settings.outline_width)
        outline_color = render_settings.outline_color
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        # alignment_id = render_settings.alignment_id
        alignment = "center"
        direction = render_settings.direction
        text_items_state = []
        for blk in image.blk_list:
            x1, y1, width, height = blk.xywh

            translation = blk.translation
            if not translation or len(translation) == 1:
                continue

            translation, font_size = pyside_word_wrap(translation, font, width, height,
                                                      line_spacing, outline_width, bold, italic, underline,
                                                      alignment, direction, max_font_size, min_font_size)

            if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):
                translation = translation.replace(' ', '')

            text_items_state.append({
                'text': translation,
                'font_family': font,
                'font_size': font_size,
                'text_color': font_color,
                'alignment': alignment,
                'line_spacing': line_spacing,
                'outline_color': outline_color,
                'outline_width': outline_width,
                'bold': bold,
                'italic': italic,
                'underline': underline,
                'position': (x1, y1),
                'rotation': blk.angle,
                'scale': 1.0,
                'transform_origin': blk.tr_origin_point,
                'width': width,
                'direction': direction,
                'selection_outlines': [OutlineInfo(0, len(translation),
                                                   outline_color, outline_width,
                                                   OutlineType.Full_Document)] if outline else []
            })

        # self.main_page.progress_update.emit(index, total_images, 9, 10, False)
        # if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
        #     self.main_page.current_worker = None
        #     break


        image.translated_image = self.renderer.render_to_image(image.cleaned_image)
        # self.main_page.progress_update.emit(index, total_images, 10, 10, False)

    def clean_image(self, image: Image):
        config = get_config(self.main_page.settings_page)
        mask = generate_mask(image.raw_data, image.blk_list)

        inpaint_input_img = self.inpainter(cv2.cvtColor(image.raw_data, cv2.COLOR_BGR2RGB), mask, config)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGBA)
        image.cleaned_image = inpaint_input_img

    def save_image(self, image_obj, target_folder: str, overwrite: bool):
        os.makedirs(target_folder, exist_ok=True)

        filename = os.path.basename(getattr(image_obj, 'path', ''))
        if not filename:
            raise ValueError("Cannot derive filename; 'path' attribute is missing or empty on image object.")

        target_path = os.path.join(target_folder, filename)

        if os.path.exists(target_path) and not overwrite:
            raise FileExistsError(f"File '{target_path}' exists and overwrite=False.")

        img_data = None
        if hasattr(image_obj, 'translated_image') and image_obj.translated_image is not None:
            img_data = image_obj.translated_image
        else:
            raise ValueError(f"No image data on object for saving (path: {getattr(image_obj, 'path', '<unknown>')}).")

        if not cv2.imwrite(target_path, img_data):
            raise IOError(f"Failed to write image to '{target_path}'.")

    def start_batch(self):

        for image_path in self.main_page.image_paths:
            image = self.main_page.images[image_path]
            self.process_image(image)
            self.main_page.logger.update_status(image_path, "done")
            self.save_image(image, self.main_page.output_path, True)
            # save the translated image to a target path
            

    def detect_text_block(self, image):
        blk_list = self.detector.detect(image.raw_data)
        # TODO: add a check for empty blk_list
        image.blk_list = blk_list

    def ocr_image(self,image: Image):
        result = self.ocr.process(image.raw_data, image.blk_list)
        image.blk_list = result

class ComicTranslator:
    def __init__(self, comic_path, output_path, source_lang, target_lang):
        self.path = comic_path
        self.images = {}
        self.image_paths = get_image_path(comic_path)
        for image_path in self.image_paths:
            image = Image(image_path)
            image.raw_data = load_image(image_path)
            self.images[image_path] = image
        self.comic_path = comic_path
        self.output_path = output_path
        self.settings_page = SettingsPage()
        self.logger = Logger(self.image_paths)
        # TODO: make this dynamic
        self.num_of_threads = self.settings_page.get_max_threads()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.processor = TranslatePipeline(self)

    def on_initial_image_loaded(self, images):
        self.images = images

    def start_batch_process(self):
        self.processor.start_batch()

ct = ComicTranslator("./test-data", "./test-data2", "Japanese", "Korean")
ct.start_batch_process()


