import os
from typing import List

import cv2, shutil
import tempfile
import numpy as np
import copy
# from typing import Callable, Tuple, List
# from dataclasses import asdict, is_dataclass
#
from modules.detection.utils.general import do_rectangles_overlap, get_inpaint_bboxes
from modules.utils.textblock import TextBlock
from modules.rendering.render import manual_wrap
from modules.utils.file_handler import FileHandler
from modules.utils.pipeline_utils import font_selected, validate_settings, get_layout_direction, \
                                         validate_ocr, validate_translator, get_language_code
from modules.utils.archives import make
from modules.utils.download import get_models, mandatory_models
from modules.utils.translator_utils import format_translations, is_there_text
from modules.rendering.render import TextRenderingSettings
from pipeline import ComicTranslatePipeline


for model in mandatory_models:
    get_models(model)

class ComicTranslate:
    def __init__(self, parent=None):
        self.image_files = []
        self.curr_img_idx = -1
        self.image_states = {}

        self.blk_list = []
        self.image_data = {}  # Store the latest version of each image
        self.image_history = {}  # Store file path history for all images
        self.in_memory_history = {}  # Store cv2 image history for recent images
        self.current_history_index = {}  # Current position in the history for each image
        self.displayed_images = set()  # Set to track displayed images


        self.curr_tblock = None
        self.curr_tblock_item = None

        self.project_file = None

        self.pipeline = ComicTranslatePipeline(self)
        self.file_handler = FileHandler()
        self.current_worker = None

        self.image_cards = []
        self.current_highlighted_card = None

        self.max_images_in_memory = 10
        self.loaded_images = []

    def load_image(self, file_path: str):
        if file_path in self.image_data:
            return self.image_data[file_path]

        # Check if the image has been displayed before
        if file_path in self.image_history:
            # Get the current index from the history
            current_index = self.current_history_index[file_path]

            # Get the temp file path at the current index
            current_temp_path = self.image_history[file_path][current_index]

            # Load the image from the temp file
            cv2_image = cv2.imread(current_temp_path)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            if cv2_image is not None:
                return cv2_image

        # If not in memory and not in history (or failed to load from temp),
        # load from the original file path
        try:
            cv2_image = cv2.imread(file_path)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return cv2_image
        except Exception as e:
            print(f"Error loading image {file_path}: {str(e)}")
            return None

    def load_initial_image(self, file_path: str):
        # get the list of image under the file path
        file_paths = []
        if os.path.isdir(file_path):
            for root, _, files in os.walk(file_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            file_paths = [file_path]
        file_paths = self.file_handler.prepare_files(file_paths)
        self.image_files = file_paths

        if file_paths:
            return self.load_image(file_paths[0])
        return None

    # def save_image_state(self, file: str):
    #     self.image_states[file] = {
    #         'viewer_state': self.image_viewer.save_state(),
    #         'source_lang': self.s_combo.currentText(),
    #         'target_lang': self.t_combo.currentText(),
    #         'brush_strokes': self.image_viewer.save_brush_strokes(),
    #         'blk_list': self.blk_list.copy()  # Store a copy of the list, not a reference
    #     }

    def on_initial_image_loaded(self, cv2_image):
        if cv2_image is not None:
            self.image_data[self.image_files[0]] = cv2_image
            self.image_history[self.image_files[0]] = [self.image_files[0]]
            self.in_memory_history[self.image_files[0]] = [cv2_image.copy()]
            self.current_history_index[self.image_files[0]] = 0
            # self.save_image_state(self.image_files[0])

    def start_batch_process(self):
        for image_path in self.image_files:
            source_lang = "Japanese"# self.image_states[image_path]['source_lang']
            target_lang = "Korean" # self.image_states[image_path]['target_lang']

            # if not validate_settings(self, source_lang, target_lang):
            #     return
        self.pipeline.batch_process()


