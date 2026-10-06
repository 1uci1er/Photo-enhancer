#!/usr/bin/env python3
"""
AI-Powered Professional Color Grading Script
Applies professional color grading without photo enhancement or GUI dependency
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import argparse
from sklearn.cluster import KMeans
import json
import math
from tqdm import tqdm
import configparser
import logging
import time
import concurrent.futures

# Define maps at module level for global access
preset_map = {'1': 'cinematic', '2': 'portrait', '3': 'landscape', '4': 'vintage', '5': 'modern'}

class AIColorGrader:
    def __init__(self, config_path: str = 'config.ini'):
        self.presets = {
            'cinematic': {'warmth': 0.2, 'contrast': 0.3, 'saturation': 0.1, 'shadows': 0.15, 'highlights': -0.1},
            'portrait': {'warmth': 0.15, 'contrast': 0.2, 'saturation': 0.2, 'shadows': 0.2, 'highlights': -0.05},
            'landscape': {'warmth': 0.1, 'contrast': 0.25, 'saturation': 0.3, 'shadows': 0.1, 'highlights': -0.15},
            'vintage': {'warmth': 0.3, 'contrast': 0.15, 'saturation': -0.1, 'shadows': 0.25, 'highlights': -0.2},
            'modern': {'warmth': -0.1, 'contrast': 0.4, 'saturation': 0.15, 'shadows': 0.05, 'highlights': -0.25}
        }
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        if 'DEFAULT' in config:
            self.default_settings = config['DEFAULT']
        else:
            self.default_settings = {}
    
    def analyze_image_ai(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logging.debug(f"Image loaded, shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        
        colors = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(colors)
        dominant_colors = kmeans.cluster_centers_
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        color_temp = (avg_r - avg_b) / (avg_r + avg_b + 0.001)
        
        scene_type = self._detect_scene_type(dominant_colors, brightness, contrast)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'color_temp': color_temp,
            'dominant_colors': dominant_colors,
            'scene_type': scene_type,
            'recommended_preset': self._recommend_preset(scene_type, brightness, contrast)
        }
    
    def _detect_scene_type(self, dominant_colors: np.ndarray, brightness: float, contrast: float) -> str:
        avg_color = np.mean(dominant_colors, axis=0)
        skin_tone_ranges = [(200, 180, 140), (255, 220, 177), (139, 69, 19)]
        is_portrait = any(np.linalg.norm(avg_color - skin) < 50 for skin in skin_tone_ranges)
        green_dominance = avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]
        blue_dominance = avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]
        
        if is_portrait:
            return 'portrait'
        elif green_dominance or blue_dominance:
            return 'landscape'
        elif brightness < 100:
            return 'cinematic'
        else:
            return 'modern'
    
    def _recommend_preset(self, scene_type: str, brightness: float, contrast: float) -> str:
        if scene_type in self.presets:
            return scene_type
        if brightness < 80:
            return 'cinematic'
        elif contrast > 60:
            return 'modern'
        else:
            return 'portrait'
    
    def apply_color_grading(self, image_path: str, output_path: str = None, 
                          preset: str = None, custom_params: dict = None, 
                          quality: int = 100) -> str:
        try:
            start_time = time.time()
            img = Image.open(image_path)
            metadata = img.info
            logging.debug(f"Opened image {image_path}, mode: {img.mode}, size: {img.size}")
            
            if img.mode != 'RGB':
                logging.debug(f"Converting image from mode {img.mode} to RGB")
                img = img.convert('RGB')
            
            temp_path = f"temp_{os.path.basename(image_path)}"
            img.save(temp_path, quality=95)
            
            try:
                analysis = self.analyze_image_ai(temp_path)
                params = custom_params or self.presets.get(preset or analysis['recommended_preset'])
                logging.debug(f"Applying preset: {preset}, params: {params}")
                
                processed_img = self._process_image_hq(img, params, analysis)
                
                if output_path is None:
                    name, ext = os.path.splitext(image_path)
                    suffix = "_graded"
                    output_path = f"{name}{suffix}{ext}"
                
                save_kwargs = {'quality': quality, 'optimize': True}
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    save_kwargs.update({
                        'format': 'JPEG',
                        'subsampling': 0,
                        'qtables': 'web_high'
                    })
                elif image_path.lower().endswith('.png'):
                    save_kwargs.update({
                        'format': 'PNG',
                        'compress_level': 1
                    })
                
                # Merge metadata with save_kwargs, avoiding duplicate 'progressive'
                final_kwargs = save_kwargs.copy()
                for key, value in metadata.items():
                    if key not in save_kwargs:
                        final_kwargs[key] = value
                
                processed_img.save(output_path, **final_kwargs)
                logging.info(f"Processed image saved to {output_path} in {time.time() - start_time:.2f}s")
                return output_path
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            raise
    
    def _process_image_hq(self, img: Image.Image, params: dict, analysis: dict) -> Image.Image:
        img_array = np.array(img, dtype=np.float32)
        logging.debug(f"Processing HQ image, shape: {img_array.shape}, dtype: {img_array.dtype}")
        img_array = self._adjust_color_temperature_hq(img_array, params.get('warmth', 0))
        img_array = self._adjust_tone_curve_hq(img_array, params.get('shadows', 0), params.get('highlights', 0))
        img_array = np.clip(img_array, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
        
        if params.get('contrast', 0) != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 + params['contrast'])
        
        if params.get('saturation', 0) != 0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1 + params['saturation'])
        
        if analysis['scene_type'] == 'cinematic':
            img = self._add_film_grain_hq(img)
        
        return img
    
    def _adjust_color_temperature_hq(self, img_array: np.ndarray, warmth: float) -> np.ndarray:
        logging.debug(f"Adjusting color temperature, warmth: {warmth}, array shape: {img_array.shape}")
        if warmth == 0:
            return img_array
        
        if warmth > 0:
            red_curve = 1 + (warmth * 0.15 * np.sin(np.pi * img_array[:, :, 0] / 255))
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_curve, 0, 255)
            blue_curve = 1 - (warmth * 0.08 * np.sin(np.pi * img_array[:, :, 2] / 255))
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_curve, 0, 255)
        else:
            red_curve = 1 + (warmth * 0.08 * np.sin(np.pi * img_array[:, :, 0] / 255))
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * red_curve, 0, 255)
            blue_curve = 1 - (warmth * 0.15 * np.sin(np.pi * img_array[:, :, 2] / 255))
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * blue_curve, 0, 255)
        
        return img_array
    
    def _adjust_tone_curve_hq(self, img_array: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
        logging.debug(f"Adjusting tone curve, shadows: {shadows}, highlights: {highlights}, array shape: {img_array.shape}")
        if shadows == 0 and highlights == 0:
            return img_array
        
        x = np.linspace(0, 255, 256)
        curve = x.copy()
        
        if shadows != 0:
            shadow_strength = shadows * 0.8
            shadow_mask = 1 - np.power(x / 255.0, 2)
            curve += shadow_strength * 40 * shadow_mask
        
        if highlights != 0:
            highlight_strength = highlights * 0.8
            highlight_mask = np.power(x / 255.0, 2)
            curve += highlight_strength * 40 * highlight_mask
        
        curve = np.clip(curve, 0, 255)
        
        for i in range(3):
            img_array[:, :, i] = np.interp(img_array[:, :, i], x, curve)
        
        return img_array
    
    def _add_film_grain_hq(self, img: Image.Image, intensity: float = 0.05) -> Image.Image:
        img_array = np.array(img, dtype=np.float32)
        logging.debug(f"Adding film grain, intensity: {intensity}, array shape: {img_array.shape}")
        noise_r = np.random.normal(0, intensity * 15, img_array.shape[:2])
        noise_g = np.random.normal(0, intensity * 12, img_array.shape[:2])
        noise_b = np.random.normal(0, intensity * 18, img_array.shape[:2])
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        grain_intensity = (255 - luminance) / 255
        img_array[:, :, 0] += noise_r * grain_intensity
        img_array[:, :, 1] += noise_g * grain_intensity
        img_array[:, :, 2] += noise_b * grain_intensity
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def batch_process(self, input_paths: list[str], output_dir: str, preset: str = None, 
                     maintain_structure: bool = True, max_workers: int = 4):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        progress_lock = threading.Lock()
        completed = {'count': 0}
        total = len(input_paths)
        
        def process_single_image(input_path: str) -> dict:
            try:
                if maintain_structure and os.path.isdir(os.path.dirname(input_path)):
                    rel_path = os.path.relpath(input_path, os.path.commonpath(input_paths))
                    output_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                else:
                    filename = os.path.basename(input_path)
                    name, ext = os.path.splitext(filename)
                    suffix = "_graded"
                    output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")
                
                result_path = self.apply_color_grading(input_path, output_path, preset, None, 100)
                
                with progress_lock:
                    completed['count'] += 1
                    logging.info(f"[{completed['count']}/{total}] Completed: {os.path.basename(input_path)}")
                return {'status': 'success', 'input': input_path, 'output': result_path}
                
            except Exception as e:
                with progress_lock:
                    completed['count'] += 1
                    logging.error(f"[{completed['count']}/{total}] Error processing {os.path.basename(input_path)}: {e}")
                return {'status': 'error', 'input': input_path, 'error': str(e)}
        
        results = []
        with tqdm(total=total, desc="Processing images", unit="image") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_single_image, input_paths))
                pbar.update(total)
        
        return results
    
    def scan_directory_recursive(self, directory: str, include_subdirs: bool = True) -> list[str]:
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.raw'}
        image_files = []
        
        if include_subdirs:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_formats):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    image_files.append(os.path.join(directory, file))
        
        return sorted(image_files)
    
    def process_selected_photos(self, preset: str = None, output_dir: str = None, 
                              max_workers: int = 4) -> list[dict]:
        logging.error("GUI selection is not available. Use batch mode with a directory instead.")
        return []

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Professional Color Grading')
    parser.add_argument('input', nargs='?', help='Input image or directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('-p', '--preset', choices=['cinematic', 'portrait', 'landscape', 'vintage', 'modern'], 
                       help='Color grading preset')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    parser.add_argument('-s', '--select', action='store_true', help='Select multiple photos with GUI (disabled)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Include subdirectories in batch processing')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel workers for batch processing')
    parser.add_argument('-q', '--quality', type=int, default=100, help='Output quality (1-100)')
    parser.add_argument('--warmth', type=float, default=0, help='Color temperature adjustment (-1 to 1)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast adjustment (-1 to 1)')
    parser.add_argument('--saturation', type=float, default=0, help='Saturation adjustment (-1 to 1)')
    parser.add_argument('--shadows', type=float, default=0, help='Shadow adjustment (-1 to 1)')
    parser.add_argument('--highlights', type=float, default=0, help='Highlight adjustment (-1 to 1)')
    parser.add_argument('--save-preset', help='Save current settings as a custom preset')
    parser.add_argument('--load-preset', help='Load custom preset from file')
    parser.add_argument('--preview', action='store_true', help='Preview results before saving')
    parser.add_argument('--config', default='config.ini', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('color_grading.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    grader = AIColorGrader(config_path=args.config)

    if args.load_preset:
        try:
            grader.load_presets(args.load_preset)
            logger.info(f"Loaded custom preset from {args.load_preset}")
        except Exception as e:
            logger.error(f"Failed to load preset: {e}")
            return

    custom_params = None
    if any([args.warmth, args.contrast, args.saturation, args.shadows, args.highlights]):
        custom_params = {'warmth': args.warmth, 'contrast': args.contrast, 'saturation': args.saturation,
                        'shadows': args.shadows, 'highlights': args.highlights}
        if args.save_preset:
            try:
                grader.create_preset(args.save_preset, args.warmth, args.contrast, args.saturation,
                                  args.shadows, args.highlights)
                grader.save_presets(args.save_preset + '.json')
                logger.info(f"Saved custom preset as {args.save_preset}.json")
            except Exception as e:
                logger.error(f"Failed to save preset: {e}")

    if args.select:
        logger.warning("GUI selection is disabled. Use batch mode with a directory instead.")
        return

    if args.batch and args.input:
        if os.path.isdir(args.input):
            print("Select Color Grading Preset:")
            for key, value in preset_map.items():
                print(f"{key}. {value.capitalize()}")
            preset_choice = input("Enter choice (1-5): ").strip()
            preset = preset_map.get(preset_choice, None)
            if preset is None:
                logger.error("Invalid preset choice")
                return
            
            logger.info(f"Selected preset: {preset}")
            image_files = grader.scan_directory_recursive(args.input, args.recursive)
            if image_files:
                output_dir = args.output or os.path.join(args.input, 'color_graded_output')
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Starting batch processing of {len(image_files)} images")
                start_time = time.time()
                
                results = grader.batch_process(image_files, output_dir, preset,
                                             max_workers=args.workers)
                
                summary = {'total': len(results), 'successful': sum(1 for r in results if r['status'] == 'success'),
                          'failed': sum(1 for r in results if r['status'] == 'error'),
                          'duration': time.time() - start_time}
                with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Batch processing completed in {summary['duration']:.2f}s")
                logger.info(f"Summary: {summary['successful']} successful, {summary['failed']} failed")
            else:
                logger.warning("No image files found in directory")
        else:
            logger.error("Input path must be a directory for batch processing")
        return

    if args.input and os.path.isfile(args.input):
        logger.info(f"Processing single image: {args.input}")
        print("Select Color Grading Preset:")
        for key, value in preset_map.items():
            print(f"{key}. {value.capitalize()}")
        preset_choice = input("Enter choice (1-5): ").strip()
        preset = preset_map.get(preset_choice, None)
        if preset is None:
            logger.error("Invalid preset choice")
            return
        
        logger.info(f"Selected preset: {preset}")
        try:
            if args.preview:
                img = Image.open(args.input)
                processed_img = grader.apply_color_grading(args.input, None, preset, custom_params,
                                                        args.quality)
                processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                user_input = input("Save result? (y/n): ").lower()
                if user_input == 'y':
                    output_path = grader.apply_color_grading(args.input, args.output, preset, custom_params,
                                                          args.quality)
                    logger.info(f"Graded image saved to: {output_path}")
                else:
                    logger.info("Processing cancelled by user")
            else:
                output_path = grader.apply_color_grading(args.input, args.output, preset, custom_params,
                                                      args.quality)
                logger.info(f"Graded image saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {args.input}: {str(e)}")
        return

    logger.info("Starting interactive mode")
    print("🎨 AI Color Grading Tool")
    print("1. Process single photo")
    print("2. Batch process directory")
    print("3. Create custom preset")
    print("4. Generate sample config file")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        file_path = input("Enter image path: ").strip()
        if os.path.isfile(file_path):
            print("Select Color Grading Preset:")
            for key, value in preset_map.items():
                print(f"{key}. {value.capitalize()}")
            preset_choice = input("Enter choice (1-5): ").strip()
            preset = preset_map.get(preset_choice, None)
            if preset is None:
                logger.error("Invalid preset choice")
                return
            
            logger.info(f"Selected preset: {preset}")
            try:
                if args.preview:
                    img = Image.open(file_path)
                    processed_img = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                            args.quality)
                    processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                    user_input = input("Save result? (y/n): ").lower()
                    if user_input == 'y':
                        output_path = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                              args.quality)
                        logger.info(f"Graded image saved to: {output_path}")
                    else:
                        logger.info("Processing cancelled by user")
                else:
                    output_path = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                          args.quality)
                    logger.info(f"Graded image saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        else:
            logger.error("File not found")
    
    elif choice == '2':
        dir_path = input("Enter directory path: ").strip()
        if os.path.isdir(dir_path):
            print("Select Color Grading Preset:")
            for key, value in preset_map.items():
                print(f"{key}. {value.capitalize()}")
            preset_choice = input("Enter choice (1-5): ").strip()
            preset = preset_map.get(preset_choice, None)
            if preset is None:
                logger.error("Invalid preset choice")
                return
            
            logger.info(f"Selected preset: {preset}")
            image_files = grader.scan_directory_recursive(dir_path, True)
            if image_files:
                output_dir = args.output or os.path.join(dir_path, 'color_graded_output')
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Starting batch processing of {len(image_files)} images")
                start_time = time.time()
                
                results = grader.batch_process(image_files, output_dir, preset,
                                             max_workers=args.workers)
                
                summary = {'total': len(results), 'successful': sum(1 for r in results if r['status'] == 'success'),
                          'failed': sum(1 for r in results if r['status'] == 'error'),
                          'duration': time.time() - start_time}
                with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Batch processing completed in {summary['duration']:.2f}s")
                logger.info(f"Summary: {summary['successful']} successful, {summary['failed']} failed")
            else:
                logger.warning("No image files found in directory")
        else:
            logger.error("Directory not found")
    
    elif choice == '3':
        preset_name = input("Enter preset name: ").strip()
        try:
            warmth = float(input("Enter warmth (-1 to 1): ").strip())
            contrast = float(input("Enter contrast (-1 to 1): ").strip())
            saturation = float(input("Enter saturation (-1 to 1): ").strip())
            shadows = float(input("Enter shadows (-1 to 1): ").strip())
            highlights = float(input("Enter highlights (-1 to 1): ").strip())
            
            grader.create_preset(preset_name, warmth, contrast, saturation, shadows, highlights)
            save_path = input("Enter file path to save preset (or press Enter for default): ").strip()
            if not save_path:
                save_path = f"{preset_name}.json"
            grader.save_presets(save_path)
            logger.info(f"Custom preset '{preset_name}' saved to {save_path}")
        except ValueError as e:
            logger.error(f"Invalid input for preset parameters: {e}")
    
    elif choice == '4':
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'quality': '100', 'workers': '4', 'recursive': 'True', 'preset': 'modern'}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        logger.info("Sample config file generated as 'config.ini'")
    
    else:
        logger.error("Invalid choice")

if __name__ == "__main__":
    import threading
    main()