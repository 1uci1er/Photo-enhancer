#!/usr/bin/env python3
"""
AI-Powered Professional Color Grading Script
Automatically applies professional color grading to photos using AI analysis
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import argparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Tuple, List
import json


class AIColorGrader:
    def __init__(self):
        self.presets = {
            'cinematic': {'warmth': 0.2, 'contrast': 0.3, 'saturation': 0.1, 'shadows': 0.15, 'highlights': -0.1},
            'portrait': {'warmth': 0.15, 'contrast': 0.2, 'saturation': 0.2, 'shadows': 0.2, 'highlights': -0.05},
            'landscape': {'warmth': 0.1, 'contrast': 0.25, 'saturation': 0.3, 'shadows': 0.1, 'highlights': -0.15},
            'vintage': {'warmth': 0.3, 'contrast': 0.15, 'saturation': -0.1, 'shadows': 0.25, 'highlights': -0.2},
            'modern': {'warmth': -0.1, 'contrast': 0.4, 'saturation': 0.15, 'shadows': 0.05, 'highlights': -0.25}
        }

    def analyze_image_ai(self, image_path: str) -> dict:
        """AI-powered image analysis to determine optimal grading parameters"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Analyze color distribution
        colors = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(colors)
        dominant_colors = kmeans.cluster_centers_

        # Analyze brightness and contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Analyze color temperature
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        color_temp = (avg_r - avg_b) / (avg_r + avg_b + 0.001)

        # Detect scene type based on color analysis
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
        """Detect scene type using AI analysis"""
        # Analyze color distribution
        avg_color = np.mean(dominant_colors, axis=0)

        # Check for skin tones (portrait detection)
        skin_tone_ranges = [(200, 180, 140), (255, 220, 177), (139, 69, 19)]
        is_portrait = any(np.linalg.norm(avg_color - skin) < 50 for skin in skin_tone_ranges)

        # Check for landscape indicators (greens and blues)
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
        """Recommend the best preset based on AI analysis"""
        if scene_type in self.presets:
            return scene_type

        # Fallback recommendations
        if brightness < 80:
            return 'cinematic'
        elif contrast > 60:
            return 'modern'
        else:
            return 'portrait'

    def apply_color_grading(self, image_path: str, output_path: str = None,
                            preset: str = None, custom_params: dict = None) -> str:
        """Apply professional color grading to image"""

        # Analyze image with AI
        analysis = self.analyze_image_ai(image_path)
        print(f"AI Analysis - Scene: {analysis['scene_type']}, Recommended: {analysis['recommended_preset']}")

        # Use preset or custom parameters
        if preset and preset in self.presets:
            params = self.presets[preset]
        elif custom_params:
            params = custom_params
        else:
            params = self.presets[analysis['recommended_preset']]

        # Load and process image
        img = Image.open(image_path)
        processed_img = self._process_image(img, params, analysis)

        # Save result
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_graded{ext}"

        processed_img.save(output_path, quality=95)
        return output_path

    def _process_image(self, img: Image.Image, params: dict, analysis: dict) -> Image.Image:
        """Apply color grading transformations"""

        # Convert to numpy for advanced processing
        img_array = np.array(img)

        # Apply color temperature adjustment
        img_array = self._adjust_color_temperature(img_array, params.get('warmth', 0))

        # Apply tone curve adjustments
        img_array = self._adjust_tone_curve(img_array, params.get('shadows', 0), params.get('highlights', 0))

        # Convert back to PIL for final adjustments
        img = Image.fromarray(img_array.astype(np.uint8))

        # Apply contrast
        if params.get('contrast', 0) != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 + params['contrast'])

        # Apply saturation
        if params.get('saturation', 0) != 0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1 + params['saturation'])

        # Apply film grain for cinematic look
        if analysis['scene_type'] == 'cinematic':
            img = self._add_film_grain(img)

        # Apply sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        return img

    def _adjust_color_temperature(self, img_array: np.ndarray, warmth: float) -> np.ndarray:
        """Adjust color temperature (warmth/coolness)"""
        if warmth == 0:
            return img_array

        # Create color temperature adjustment matrix
        if warmth > 0:  # Warmer
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + warmth * 0.2), 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - warmth * 0.1), 0, 255)  # Blue
        else:  # Cooler
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + warmth * 0.1), 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - warmth * 0.2), 0, 255)  # Blue

        return img_array

    def _adjust_tone_curve(self, img_array: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
        """Adjust shadows and highlights using tone curve"""
        if shadows == 0 and highlights == 0:
            return img_array

        # Create tone curve
        curve = np.arange(256, dtype=np.float32)

        # Adjust shadows (0-85 range)
        if shadows != 0:
            shadow_curve = np.power(curve / 255.0, 1 - shadows) * 255
            mask = curve <= 85
            curve[mask] = shadow_curve[mask]

        # Adjust highlights (170-255 range)
        if highlights != 0:
            highlight_curve = np.power(curve / 255.0, 1 + highlights) * 255
            mask = curve >= 170
            curve[mask] = highlight_curve[mask]

        # Apply curve to each channel
        for i in range(3):
            img_array[:, :, i] = np.interp(img_array[:, :, i], np.arange(256), curve)

        return img_array

    def _add_film_grain(self, img: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add subtle film grain for cinematic look"""
        img_array = np.array(img)
        noise = np.random.normal(0, intensity * 25, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

    def batch_process(self, input_dir: str, output_dir: str, preset: str = None):
        """Process multiple images in batch"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"graded_{filename}")

                print(f"Processing: {filename}")
                try:
                    self.apply_color_grading(input_path, output_path, preset)
                    print(f"✓ Completed: {filename}")
                except Exception as e:
                    print(f"✗ Error processing {filename}: {e}")

    def create_preset(self, name: str, warmth: float, contrast: float,
                      saturation: float, shadows: float, highlights: float):
        """Create custom color grading preset"""
        self.presets[name] = {
            'warmth': warmth,
            'contrast': contrast,
            'saturation': saturation,
            'shadows': shadows,
            'highlights': highlights
        }

    def save_presets(self, filepath: str):
        """Save presets to file"""
        with open(filepath, 'w') as f:
            json.dump(self.presets, f, indent=2)

    def load_presets(self, filepath: str):
        """Load presets from file"""
        with open(filepath, 'r') as f:
            self.presets.update(json.load(f))


def main():
    parser = argparse.ArgumentParser(description='AI-Powered Professional Color Grading')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-p', '--preset', choices=['cinematic', 'portrait', 'landscape', 'vintage', 'modern'],
                        help='Color grading preset')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--warmth', type=float, default=0, help='Color temperature adjustment (-1 to 1)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast adjustment (-1 to 1)')
    parser.add_argument('--saturation', type=float, default=0, help='Saturation adjustment (-1 to 1)')
    parser.add_argument('--shadows', type=float, default=0, help='Shadow adjustment (-1 to 1)')
    parser.add_argument('--highlights', type=float, default=0, help='Highlight adjustment (-1 to 1)')

    args = parser.parse_args()

    grader = AIColorGrader()

    # Use custom parameters if provided
    custom_params = None
    if any([args.warmth, args.contrast, args.saturation, args.shadows, args.highlights]):
        custom_params = {
            'warmth': args.warmth,
            'contrast': args.contrast,
            'saturation': args.saturation,
            'shadows': args.shadows,
            'highlights': args.highlights
        }

    if args.batch:
        grader.batch_process(args.input, args.output or 'graded_output', args.preset)
    else:
        output_path = grader.apply_color_grading(args.input, args.output, args.preset, custom_params)
        print(f"Graded image saved to: {output_path}")


if __name__ == "__main__":
    main()