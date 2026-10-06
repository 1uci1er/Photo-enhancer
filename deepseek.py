#!/usr/bin/env python3
"""
AI-Powered Professional Color Grading & Enhancement Suite
Advanced image processing with AI-driven color grading, super-resolution, and enhancement
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import argparse
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Tuple, List, Dict, Optional, Union
import json
import math
from tqdm import tqdm
import configparser
import logging
import time
import psutil
import concurrent.futures
import hashlib
import threading
from pathlib import Path
import platform
from collections import defaultdict
import warnings

# Suppress scientific notation display
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore', category=UserWarning)

# GPU acceleration setup
try:
    import cupy as cp

    GPU_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except ImportError:
    cp = None
    GPU_AVAILABLE = False


class AIPhotoEnhancer:
    """Advanced AI-powered photo enhancement with multi-stage processing"""

    VERSION = "2.5"

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.enhancement_presets = {
            'light': {'sharpen': 0.3, 'denoise': 0.2, 'upscale': 1.5, 'detail': 0.4},
            'medium': {'sharpen': 0.5, 'denoise': 0.3, 'upscale': 2.0, 'detail': 0.6},
            'heavy': {'sharpen': 0.7, 'denoise': 0.4, 'upscale': 2.5, 'detail': 0.8},
            'ultra': {'sharpen': 0.9, 'denoise': 0.5, 'upscale': 3.0, 'detail': 1.0},
            'balanced': {'sharpen': 0.6, 'denoise': 0.3, 'upscale': 1.8, 'detail': 0.7},
            'portrait': {'sharpen': 0.4, 'denoise': 0.35, 'upscale': 1.5, 'detail': 0.55},
            'landscape': {'sharpen': 0.65, 'denoise': 0.25, 'upscale': 2.2, 'detail': 0.75}
        }
        self.cache = {}
        self.model_loaded = False
        self.init_models()

        if self.use_gpu:
            logging.info(
                f"GPU acceleration enabled (CUDA v{cv2.cuda.getCudaVersion() // 1000}.{cv2.cuda.getCudaVersion() % 100 // 10})")

    def init_models(self):
        """Initialize AI models for enhancement"""
        try:
            # Placeholder for model initialization
            # In a real implementation, we would load pre-trained models here
            self.model_loaded = True
        except Exception as e:
            logging.warning(f"Could not load AI models: {str(e)}")
            self.model_loaded = False

    def get_memory_status(self) -> dict:
        """Get current memory usage statistics"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'free': mem.free,
            'percent': mem.percent
        }

    def clear_cache(self):
        """Clear processing cache"""
        self.cache.clear()
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

    def enhance_image_ai(self, image: Image.Image, preset: str = 'medium',
                         progress_callback: callable = None) -> Image.Image:
        """Main AI enhancement function with progress reporting"""
        if preset not in self.enhancement_presets:
            preset = 'medium'
            logging.warning(f"Invalid preset '{preset}', using 'medium' instead")

        params = self.enhancement_presets[preset]
        cache_key = hashlib.md5(image.tobytes() + preset.encode()).hexdigest()

        if cache_key in self.cache:
            logging.debug("Using cached enhanced image")
            return self.cache[cache_key]

        img_array = np.array(image, dtype=np.float32)
        if self.use_gpu:
            img_array = cp.array(img_array)

        start_mem = self.get_memory_status()
        start_time = time.time()

        try:
            total_steps = 5
            current_step = 0

            def update_progress(step_name):
                nonlocal current_step
                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps, step_name)

            # Step 1: AI-powered denoising
            if params['denoise'] > 0:
                update_progress("Denoising")
                img_array = self._ai_denoise(img_array, params['denoise'])

            # Step 2: Super-resolution upscaling
            if params['upscale'] > 1.0:
                update_progress("Upscaling")
                img_array = self._ai_super_resolution(img_array, params['upscale'])

            # Step 3: Advanced deblurring
            if params['sharpen'] > 0:
                update_progress("Deblurring")
                img_array = self._ai_deblur(img_array, params['sharpen'])

            # Step 4: Detail enhancement
            if params['detail'] > 0:
                update_progress("Detail Enhancement")
                img_array = self._enhance_details(img_array, params['detail'])

            # Step 5: Final refinement
            update_progress("Final Refinement")
            img_array = self._final_refinement(img_array)

            # Convert back to CPU if using GPU
            if self.use_gpu:
                img_array = cp.asnumpy(img_array)

            # Convert back to PIL
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            result = Image.fromarray(img_array)
            self.cache[cache_key] = result

            end_mem = self.get_memory_status()
            logging.info(f"Enhancement completed in {time.time() - start_time:.2f}s | "
                         f"Memory Δ: {(end_mem['used'] - start_mem['used']) / 1024 ** 2:.1f}MB")

            return result

        except Exception as e:
            logging.error(f"Enhancement failed: {str(e)}")
            raise

    def _ai_denoise(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        """Advanced noise reduction with AI guidance"""
        if strength <= 0:
            return img_array

        if self.use_gpu:
            img_cuda = cv2.cuda_GpuMat()
            img_cuda.upload(img_array.astype(np.uint8))

            # Adaptive denoising based on strength
            h_val = max(1.0, strength * 15)
            h_color = max(1.0, strength * 12)

            denoised = cv2.cuda.fastNlMeansDenoisingColored(
                img_cuda,
                h=h_val,
                hColor=h_color,
                templateWindowSize=7,
                searchWindowSize=21
            )
            return cp.asnumpy(denoised).astype(np.float32)

        # CPU implementation
        img_cv = img_array.astype(np.uint8)
        h_val = max(1.0, strength * 15)

        if len(img_cv.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_cv,
                None,
                h=h_val,
                hColor=h_val * 0.8,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                img_cv,
                None,
                h=h_val,
                templateWindowSize=7,
                searchWindowSize=21
            )
        return 0.7 * denoised.astype(np.float32) + 0.3 * img_array

    def _ai_super_resolution(self, img_array: np.ndarray, scale_factor: float) -> np.ndarray:
        """AI-powered super-resolution with edge preservation"""
        if scale_factor <= 1.0:
            return img_array

        original_dtype = img_array.dtype
        h, w = img_array.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Preserve aspect ratio
        aspect_ratio = w / h
        if abs(aspect_ratio - (new_w / new_h)) > 0.01:
            new_w = int(new_h * aspect_ratio)

        # Convert to uint8 for OpenCV processing
        img_uint8 = img_array.astype(np.uint8)

        # Edge-preserving upscaling
        upscaled = cv2.resize(
            img_uint8,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Edge-aware filtering
        upscaled = cv2.edgePreservingFilter(
            upscaled,
            flags=cv2.RECURS_FILTER,
            sigma_s=60,
            sigma_r=0.35
        )

        # Smart sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp_strength = min(0.2, (scale_factor - 1.0) * 0.1)

        if len(upscaled.shape) == 3:
            for i in range(3):
                channel = upscaled[:, :, i]
                blurred = cv2.GaussianBlur(channel, (0, 0), 1.0)
                mask = cv2.subtract(channel, blurred)
                upscaled[:, :, i] = cv2.addWeighted(
                    channel, 1.0 + sharp_strength,
                    mask, sharp_strength,
                    0
                )
        else:
            blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
            mask = cv2.subtract(upscaled, blurred)
            upscaled = cv2.addWeighted(
                upscaled, 1.0 + sharp_strength,
                mask, sharp_strength,
                0
            )

        return upscaled.astype(original_dtype)

    def _ai_deblur(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        """Adaptive deblurring with blur estimation"""
        if strength <= 0:
            return img_array

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.astype(np.uint8)

        # Estimate blur amount
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_amount = laplacian.var()
        logging.debug(f"Blur amount detected: {blur_amount:.1f}")

        # Adaptive parameters based on blur estimation
        if blur_amount < 100:
            sigma = 2.5 - (strength * 1.5)
        elif blur_amount < 500:
            sigma = 1.8 - (strength * 0.8)
        else:
            sigma = 1.2 - (strength * 0.4)

        sigma = max(0.5, sigma)
        blurred = gaussian_filter(img_array, sigma=sigma)
        unsharp_mask = img_array - blurred
        sharpened = img_array + strength * unsharp_mask

        # Edge enhancement
        sharpened = self._enhance_edges(sharpened, strength)

        return np.clip(sharpened, 0, 255)

    def _enhance_edges(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        """Multi-directional edge enhancement"""
        kernels = {
            'horizontal': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            'vertical': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'diagonal1': np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
            'diagonal2': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        }

        enhanced = img_array.copy()
        edge_strength = strength * 0.15

        if len(img_array.shape) == 3:
            for i in range(3):
                channel = img_array[:, :, i]
                edge_response = np.zeros_like(channel, dtype=np.float32)

                for name, kernel in kernels.items():
                    edge = cv2.filter2D(channel, -1, kernel)
                    edge_response = np.maximum(edge_response, np.abs(edge))

                enhanced[:, :, i] = channel + edge_strength * edge_response
        else:
            edge_response = np.zeros_like(img_array, dtype=np.float32)
            for kernel in kernels.values():
                edge = cv2.filter2D(img_array, -1, kernel)
                edge_response = np.maximum(edge_response, np.abs(edge))
            enhanced = img_array + edge_strength * edge_response

        return enhanced

    def _enhance_details(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        """Detail enhancement using frequency separation"""
        gaussian_blur = gaussian_filter(img_array, sigma=1.2)
        high_pass = img_array - gaussian_blur
        enhanced = img_array + strength * high_pass * 0.6

        # Local contrast enhancement
        enhanced = self._enhance_texture(enhanced, strength)
        return enhanced

    def _enhance_texture(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        """Texture enhancement using local contrast"""
        if len(img_array.shape) == 3:
            img_uint8 = img_array.astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)

            # Adaptive local contrast
            mean_filtered = uniform_filter(l_channel, size=11)
            detail = l_channel - mean_filtered
            enhanced_l = l_channel + strength * 0.4 * detail

            # Apply sigmoid curve for natural contrast
            x = np.linspace(0, 255, 256)
            curve = 255 / (1 + np.exp(-0.05 * (x - 128)))
            enhanced_l = np.interp(enhanced_l, x, curve)

            lab[:, :, 0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced.astype(np.float32)
        else:
            mean_filtered = uniform_filter(img_array, size=11)
            detail = img_array - mean_filtered
            enhanced = img_array + strength * 0.4 * detail
            return enhanced

    def _final_refinement(self, img_array: np.ndarray) -> np.ndarray:
        """Final image refinement pipeline"""
        # Median filter for noise reduction
        refined = medfilt2d(img_array.astype(np.uint8), kernel_size=3)

        # Micro-contrast enhancement
        enhanced = cv2.convertScaleAbs(refined, alpha=1.02, beta=1)

        # Subtle S-curve contrast
        x = np.linspace(0, 255, 256)
        s_curve = 255 * (x / 255) ** (1 / 1.1)
        s_curve = s_curve * 0.7 + x * 0.3  # Blend with linear

        if len(enhanced.shape) == 3:
            for i in range(3):
                enhanced[:, :, i] = np.interp(enhanced[:, :, i], x, s_curve)
        else:
            enhanced = np.interp(enhanced, x, s_curve)

        return enhanced.astype(np.float32)

    def detect_blur_level(self, image: Image.Image) -> dict:
        """Advanced blur detection with multiple metrics"""
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # FFT-based blur detection
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # High-frequency content analysis
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        fft_shift[cy - 30:cy + 30, cx - 30:cx + 30] = 0
        high_freq = np.sum(magnitude > 50) / (h * w)

        # Combined blur score
        blur_score = (laplacian_var * 0.7) + (high_freq * 1000 * 0.3)

        # Classification
        if blur_score < 150:
            blur_level = "very_blurry"
            recommended_preset = "ultra"
        elif blur_score < 300:
            blur_level = "blurry"
            recommended_preset = "heavy"
        elif blur_score < 500:
            blur_level = "slightly_blurry"
            recommended_preset = "medium"
        else:
            blur_level = "sharp"
            recommended_preset = "light"

        return {
            'blur_score': blur_score,
            'laplacian_var': laplacian_var,
            'high_freq_content': high_freq,
            'blur_level': blur_level,
            'recommended_preset': recommended_preset
        }


class AIColorGrader:
    """AI-powered professional color grading system"""

    def __init__(self, config_path: str = 'config.ini'):
        self.presets = {
            'cinematic': {'warmth': 0.2, 'contrast': 0.3, 'saturation': 0.1, 'shadows': 0.15, 'highlights': -0.1,
                          'vibrance': 0.1},
            'portrait': {'warmth': 0.15, 'contrast': 0.2, 'saturation': 0.2, 'shadows': 0.2, 'highlights': -0.05,
                         'vibrance': 0.15},
            'landscape': {'warmth': 0.1, 'contrast': 0.25, 'saturation': 0.3, 'shadows': 0.1, 'highlights': -0.15,
                          'vibrance': 0.25},
            'vintage': {'warmth': 0.3, 'contrast': 0.15, 'saturation': -0.1, 'shadows': 0.25, 'highlights': -0.2,
                        'vibrance': -0.1},
            'modern': {'warmth': -0.1, 'contrast': 0.4, 'saturation': 0.15, 'shadows': 0.05, 'highlights': -0.25,
                       'vibrance': 0.2},
            'dramatic': {'warmth': 0.25, 'contrast': 0.5, 'saturation': 0.1, 'shadows': 0.3, 'highlights': -0.3,
                         'vibrance': 0.15},
            'pastel': {'warmth': 0.05, 'contrast': -0.1, 'saturation': -0.15, 'shadows': 0.1, 'highlights': 0.1,
                       'vibrance': 0.05},
            'moody': {'warmth': 0.15, 'contrast': 0.35, 'saturation': -0.05, 'shadows': 0.35, 'highlights': -0.15,
                      'vibrance': 0.05}
        }
        self.enhancer = AIPhotoEnhancer(use_gpu=True)
        self.config_path = config_path
        self.history = []
        self.load_config()

    def load_config(self):
        """Load configuration settings"""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path)
            if 'DEFAULT' in config:
                self.default_settings = dict(config['DEFAULT'])
            else:
                self.default_settings = {}

            # Load custom presets
            if 'PRESETS' in config:
                for name, values in config['PRESETS'].items():
                    try:
                        params = json.loads(values)
                        self.presets[name] = params
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse preset: {name}")
        else:
            self.default_settings = {}
            logging.info("No config file found, using default settings")

    def save_config(self):
        """Save current configuration to file"""
        config = configparser.ConfigParser()
        config['DEFAULT'] = self.default_settings

        # Save custom presets
        config['PRESETS'] = {}
        for name, params in self.presets.items():
            if name not in ['cinematic', 'portrait', 'landscape', 'vintage', 'modern']:
                config['PRESETS'][name] = json.dumps(params)

        with open(self.config_path, 'w') as configfile:
            config.write(configfile)
        logging.info(f"Configuration saved to {self.config_path}")

    def analyze_image_ai(self, image_path: str) -> dict:
        """AI-powered image analysis for optimal grading"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Downsample for faster processing
        scale_factor = max(1, min(h, w) // 1000)
        small_img = cv2.resize(img_rgb, (w // scale_factor, h // scale_factor))

        colors = small_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(colors)
        dominant_colors = kmeans.cluster_centers_

        # Calculate color variance
        color_variance = np.mean([np.std(colors[kmeans.labels_ == i], axis=0)
                                  for i in range(5)])

        # Brightness and contrast analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Color temperature analysis
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        color_temp = (avg_r - avg_b) / (avg_r + avg_b + 0.001) * 100

        # Scene type detection
        scene_type = self._detect_scene_type(dominant_colors, brightness, contrast)

        return {
            'dimensions': (w, h),
            'brightness': brightness,
            'contrast': contrast,
            'color_temp': color_temp,
            'color_variance': color_variance,
            'dominant_colors': dominant_colors.tolist(),
            'scene_type': scene_type,
            'recommended_preset': self._recommend_preset(scene_type, brightness, contrast)
        }

    def _detect_scene_type(self, dominant_colors: np.ndarray,
                           brightness: float, contrast: float) -> str:
        """Advanced scene type detection"""
        avg_color = np.mean(dominant_colors, axis=0)

        # Skin tone detection
        skin_tone_ranges = [
            (np.array([180, 130, 100]), np.array([255, 200, 170])),  # Light skin
            (np.array([150, 90, 70]), np.array([190, 140, 120])),  # Medium skin
            (np.array([80, 50, 40]), np.array([150, 100, 90]))  # Dark skin
        ]

        is_portrait = False
        for low, high in skin_tone_ranges:
            if np.all(avg_color > low) and np.all(avg_color < high):
                is_portrait = True
                break

        # Color dominance analysis
        r, g, b = avg_color
        green_dominance = g > r * 1.2 and g > b * 1.2
        blue_dominance = b > r * 1.2 and b > g * 1.2
        red_dominance = r > g * 1.2 and r > b * 1.2

        # Scene classification
        if is_portrait:
            return 'portrait'
        elif green_dominance:
            return 'landscape'
        elif blue_dominance:
            return 'seascape' if brightness > 120 else 'night'
        elif red_dominance and brightness < 100:
            return 'sunset'
        elif contrast > 80 and brightness < 120:
            return 'dramatic'
        elif brightness < 100:
            return 'cinematic'
        elif contrast < 30:
            return 'pastel'
        else:
            return 'general'

    def _recommend_preset(self, scene_type: str,
                          brightness: float, contrast: float) -> str:
        """Preset recommendation based on scene analysis"""
        scene_preset_map = {
            'portrait': 'portrait',
            'landscape': 'landscape',
            'seascape': 'landscape',
            'night': 'cinematic',
            'sunset': 'vintage',
            'dramatic': 'dramatic',
            'pastel': 'pastel',
            'cinematic': 'cinematic'
        }

        if scene_type in scene_preset_map:
            return scene_preset_map[scene_type]
        if brightness < 80:
            return 'cinematic'
        elif contrast > 70:
            return 'dramatic'
        elif contrast < 40:
            return 'pastel'
        else:
            return 'modern'

    def apply_color_grading(self, image_path: str, output_path: str = None,
                            preset: str = None, custom_params: dict = None,
                            quality: int = 98, enhance: bool = True,
                            enhancement_preset: str = None,
                            progress_callback: callable = None) -> str:
        """Apply color grading with enhanced processing pipeline"""
        try:
            start_time = time.time()
            img = Image.open(image_path)
            original_mode = img.mode
            metadata = img.info

            # Preserve transparency if exists
            has_alpha = 'A' in img.getbands()
            if has_alpha:
                alpha_channel = img.split()[-1]

            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Auto-detect enhancement preset if needed
            if enhance and enhancement_preset is None:
                blur_analysis = self.enhancer.detect_blur_level(img)
                enhancement_preset = blur_analysis['recommended_preset']
                logging.info(f"Auto-detected enhancement preset: {enhancement_preset} "
                             f"(blur score: {blur_analysis['blur_score']:.1f})")

            # Apply enhancement
            if enhance:
                def enhancement_progress(progress, step):
                    if progress_callback:
                        progress_callback(progress * 0.5, f"Enhancing: {step}")

                img = self.enhancer.enhance_image_ai(
                    img,
                    enhancement_preset,
                    progress_callback=enhancement_progress
                )

            # Create temp file for analysis
            temp_path = f"temp_{os.getpid()}_{os.path.basename(image_path)}"
            img.save(temp_path, quality=95)

            try:
                # Analyze image
                analysis = self.analyze_image_ai(temp_path)

                # Determine parameters
                if custom_params:
                    params = custom_params
                else:
                    preset_name = preset or analysis['recommended_preset']
                    params = self.presets.get(preset_name, self.presets['modern'])

                # Apply color grading
                def grading_progress(progress, step):
                    if progress_callback:
                        progress_callback(0.5 + progress * 0.5, f"Grading: {step}")

                processed_img = self._process_image_hq(
                    img,
                    params,
                    analysis,
                    progress_callback=grading_progress
                )

                # Restore alpha channel if existed
                if has_alpha:
                    processed_img.putalpha(alpha_channel)

                # Determine output path
                if output_path is None:
                    name, ext = os.path.splitext(image_path)
                    suffix = "_enhanced_graded" if enhance else "_graded"
                    output_path = f"{name}{suffix}{ext}"

                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save with appropriate settings
                save_kwargs = {'quality': quality, 'optimize': True}
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    save_kwargs.update({
                        'format': 'JPEG',
                        'subsampling': 0,  # 4:4:4 chroma subsampling
                        'qtables': 'web_high'
                    })
                elif image_path.lower().endswith('.png'):
                    save_kwargs.update({
                        'format': 'PNG',
                        'compress_level': 2,
                        'optimize': True
                    })
                elif image_path.lower().endswith('.webp'):
                    save_kwargs.update({
                        'format': 'WEBP',
                        'method': 6,  # Highest quality
                        'lossless': False
                    })

                # Preserve metadata
                processed_img.save(output_path, **save_kwargs, **metadata)

                # Add to history
                self.history.append({
                    'input': image_path,
                    'output': output_path,
                    'preset': preset,
                    'enhance': enhance,
                    'timestamp': time.time(),
                    'analysis': analysis
                })

                logging.info(f"Processed image saved to {output_path} in {time.time() - start_time:.2f}s")
                return output_path

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            raise

    def _process_image_hq(self, img: Image.Image, params: dict,
                          analysis: dict, progress_callback: callable = None) -> Image.Image:
        """High-quality color grading pipeline"""

        def update_progress(step, description):
            if progress_callback:
                progress_callback(step, description)

        update_progress(0.1, "Color Temperature")
        img_array = np.array(img, dtype=np.float32)
        img_array = self._adjust_color_temperature_hq(img_array, params.get('warmth', 0))

        update_progress(0.3, "Tone Curve")
        img_array = self._adjust_tone_curve_hq(
            img_array,
            params.get('shadows', 0),
            params.get('highlights', 0)
        )

        update_progress(0.5, "Vibrance")
        img_array = self._adjust_vibrance(img_array, params.get('vibrance', 0))

        img_array = np.clip(img_array, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

        update_progress(0.7, "Contrast")
        if params.get('contrast', 0) != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 + params['contrast'])

        update_progress(0.8, "Saturation")
        if params.get('saturation', 0) != 0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1 + params['saturation'])

        if analysis['scene_type'] == 'cinematic':
            update_progress(0.9, "Film Grain")
            img = self._add_film_grain_hq(img)

        update_progress(1.0, "Final Sharpening")
        img = self._apply_smart_sharpening(img)
        return img

    def _adjust_color_temperature_hq(self, img_array: np.ndarray, warmth: float) -> np.ndarray:
        """Perceptual color temperature adjustment"""
        if warmth == 0:
            return img_array

        # Create temperature adjustment curves
        r_curve = np.linspace(0, 255, 256)
        b_curve = np.linspace(0, 255, 256)

        if warmth > 0:  # Warmer (more yellow/red)
            r_curve = r_curve + warmth * 10 * (1 - r_curve / 255)
            b_curve = b_curve - warmth * 8 * (b_curve / 255)
        else:  # Cooler (more blue)
            warmth = abs(warmth)
            r_curve = r_curve - warmth * 8 * (r_curve / 255)
            b_curve = b_curve + warmth * 10 * (1 - b_curve / 255)

        # Apply curves
        img_array[:, :, 0] = np.interp(img_array[:, :, 0], np.arange(256), r_curve)
        img_array[:, :, 2] = np.interp(img_array[:, :, 2], np.arange(256), b_curve)

        return img_array

    def _adjust_tone_curve_hq(self, img_array: np.ndarray,
                              shadows: float, highlights: float) -> np.ndarray:
        """Advanced tone curve adjustment"""
        if shadows == 0 and highlights == 0:
            return img_array

        # Create base tone curve
        x = np.linspace(0, 255, 256)
        curve = x.copy()

        # Shadow adjustment (S-curve in shadows)
        if shadows != 0:
            shadow_strength = shadows * 0.8
            shadow_mask = 1 - np.power(np.clip(x / 128, 0, 1), 2)
            curve += shadow_strength * 50 * shadow_mask

        # Highlight adjustment (S-curve in highlights)
        if highlights != 0:
            highlight_strength = highlights * 0.8
            highlight_mask = np.power(np.clip((x - 128) / 127, 0, 1), 2)
            curve += highlight_strength * 50 * highlight_mask

        curve = np.clip(curve, 0, 255)

        # Apply per channel to preserve color relationships
        for i in range(3):
            img_array[:, :, i] = np.interp(img_array[:, :, i], x, curve)

        return img_array

    def _adjust_vibrance(self, img_array: np.ndarray, vibrance: float) -> np.ndarray:
        """Selective saturation boost for less saturated colors"""
        if vibrance == 0:
            return img_array

        # Convert to HSV for saturation manipulation
        hsv = cv2.cvtColor(img_array.ast(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = s.astype(np.float32)

        # Calculate saturation boost curve
        # Boost less saturated areas more than already saturated ones
        saturation_boost = vibrance * (1.0 - s / 255.0) * 100
        s = np.clip(s + saturation_boost, 0, 255)

        # Convert back to RGB
        hsv = cv2.merge([h, s.astype(np.uint8), v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    def _add_film_grain_hq(self, img: Image.Image, intensity: float = 0.03) -> Image.Image:
        """Realistic film grain simulation"""
        img_array = np.array(img, dtype=np.float32)
        h, w, c = img_array.shape

        # Generate Perlin-like noise for organic grain
        noise = np.zeros((h, w), dtype=np.float32)
        scale = 0.01
        octaves = 4
        persistence = 0.5

        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave

            # Create grid and vectors
            x = np.arange(w) * scale * freq
            y = np.arange(h) * scale * freq
            xv, yv = np.meshgrid(x, y)

            # Simple gradient noise
            noise += amp * np.sin(xv + np.sin(yv)) * np.cos(yv - np.cos(xv))

        # Normalize and adjust intensity
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * intensity * 255

        # Apply based on luminance
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        grain_intensity = (255 - luminance) / 255

        for i in range(3):
            channel_noise = noise * (0.8 + 0.4 * np.random.random())
            img_array[:, :, i] += channel_noise * grain_intensity

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _apply_smart_sharpening(self, img: Image.Image) -> Image.Image:
        """Edge-aware sharpening with halo control"""
        img_array = np.array(img, dtype=np.float32)
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Detect edges with Canny
        edges = cv2.Canny(gray, 50, 150)

        # Create edge mask with dilation
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1) / 255.0

        # Apply different sharpening based on edge presence
        sharp_strong = img.filter(ImageFilter.UnsharpMask(
            radius=1.2,
            percent=180,
            threshold=3
        ))
        sharp_mild = img.filter(ImageFilter.UnsharpMask(
            radius=0.7,
            percent=100,
            threshold=5
        ))

        # Blend based on edge mask
        result_array = np.array(img, dtype=np.float32)
        sharp_strong_array = np.array(sharp_strong, dtype=np.float32)
        sharp_mild_array = np.array(sharp_mild, dtype=np.float32)

        for i in range(3):
            result_array[:, :, i] = (
                    sharp_strong_array[:, :, i] * edge_mask +
                    sharp_mild_array[:, :, i] * (1 - edge_mask)
            )

        return Image.fromarray(result_array.astype(np.uint8))

    def batch_process(self, input_paths: List[str], output_dir: str, preset: str = None,
                      maintain_structure: bool = True, max_workers: int = 4,
                      enhance: bool = True, enhancement_preset: str = None,
                      progress_callback: callable = None) -> List[dict]:
        """Batch processing with resource management and progress reporting"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total = len(input_paths)
        processed = 0
        lock = threading.Lock()
        results = []
        errors = []

        def update_progress():
            nonlocal processed
            with lock:
                processed += 1
                if progress_callback:
                    progress_callback(processed / total, f"Processing {processed}/{total}")

        def process_file(input_path: str) -> dict:
            try:
                # Find common path for structure preservation
                common_path = os.path.commonpath(input_paths) if maintain_structure else ""
                rel_path = os.path.relpath(input_path, common_path) if maintain_structure else Path(input_path).name

                if maintain_structure:
                    output_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                else:
                    name, ext = os.path.splitext(Path(input_path).name)
                    suffix = "_enhanced_graded" if enhance else "_graded"
                    output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")

                def file_progress(progress, message):
                    update_progress()

                result_path = self.apply_color_grading(
                    input_path,
                    output_path,
                    preset,
                    None,
                    98,
                    enhance,
                    enhancement_preset,
                    progress_callback=file_progress
                )
                update_progress()
                return {'status': 'success', 'input': input_path, 'output': result_path}

            except Exception as e:
                update_progress()
                logging.error(f"Error processing {input_path}: {str(e)}")
                return {'status': 'error', 'input': input_path, 'error': str(e)}

        with tqdm(total=total, desc="Batch Processing", unit="image") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(process_file, path): path
                    for path in input_paths
                }

                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result['status'] == 'error':
                            errors.append(result)
                    except Exception as e:
                        err_result = {'status': 'error', 'input': path, 'error': str(e)}
                        results.append(err_result)
                        errors.append(err_result)
                    pbar.update(1)

        # Generate summary report
        summary = {
            'total': total,
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': len(errors),
            'errors': errors,
            'timestamp': time.time(),
            'preset': preset,
            'enhance': enhance,
            'enhancement_preset': enhancement_preset
        }

        # Save summary
        summary_path = os.path.join(output_dir, f"batch_summary_{int(time.time())}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(
            f"Batch processing complete. Summary: {summary['successful']} successful, {summary['failed']} failed")
        return results

    def get_history(self, max_entries: int = 20) -> list:
        """Get processing history"""
        return self.history[-max_entries:]

    def clear_history(self):
        """Clear processing history"""
        self.history = []

    def create_preset(self, name: str, **params):
        """Create custom color grading preset"""
        valid_params = ['warmth', 'contrast', 'saturation', 'shadows', 'highlights', 'vibrance']
        preset = {k: v for k, v in params.items() if k in valid_params}
        self.presets[name] = preset
        self.save_config()

    def delete_preset(self, name: str):
        """Delete custom preset"""
        if name in self.presets and name not in ['cinematic', 'portrait', 'landscape', 'vintage', 'modern']:
            del self.presets[name]
            self.save_config()
            return True
        return False

    def save_presets(self, filepath: str):
        """Save presets to external file"""
        with open(filepath, 'w') as f:
            json.dump(self.presets, f, indent=2)

    def load_presets(self, filepath: str):
        """Load presets from external file"""
        with open(filepath, 'r') as f:
            external_presets = json.load(f)

        # Merge without overwriting built-in presets
        for name, params in external_presets.items():
            if name not in ['cinematic', 'portrait', 'landscape', 'vintage', 'modern']:
                self.presets[name] = params


def main():
    # Setup argument parser first without grader-dependent choices
    parser = argparse.ArgumentParser(description='AI-Powered Professional Color Grading with Photo Enhancement')
    parser.add_argument('input', nargs='?', help='Input image or directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('-p', '--preset', help='Color grading preset')
    parser.add_argument('-e', '--enhance', help='Enhancement preset')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    parser.add_argument('-s', '--select', action='store_true', help='Select multiple photos with GUI')
    parser.add_argument('-r', '--recursive', action='store_true', help='Include subdirectories in batch processing')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel workers for batch processing')
    parser.add_argument('-q', '--quality', type=int, default=98, help='Output quality (1-100)')
    parser.add_argument('--warmth', type=float, default=0, help='Color temperature adjustment (-1 to 1)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast adjustment (-1 to 1)')
    parser.add_argument('--saturation', type=float, default=0, help='Saturation adjustment (-1 to 1)')
    parser.add_argument('--shadows', type=float, default=0, help='Shadow adjustment (-1 to 1)')
    parser.add_argument('--highlights', type=float, default=0, help='Highlight adjustment (-1 to 1)')
    parser.add_argument('--vibrance', type=float, default=0, help='Vibrance adjustment (-1 to 1)')
    parser.add_argument('--save-preset', help='Save current settings as a custom preset')
    parser.add_argument('--load-preset', help='Load custom preset from file')
    parser.add_argument('--preview', action='store_true', help='Preview results before saving')
    parser.add_argument('--config', default='config.ini', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration if available')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('color_grading.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize AIColorGrader with GPU support if requested
    grader = AIColorGrader(config_path=args.config)
    grader.enhancer = AIPhotoEnhancer(use_gpu=args.gpu)

    # Now validate presets
    if args.preset and args.preset not in grader.presets:
        logger.error(f"Invalid color preset: {args.preset}. Available presets: {', '.join(grader.presets.keys())}")
        return

    if args.enhance and args.enhance not in grader.enhancer.enhancement_presets:
        logger.error(
            f"Invalid enhancement preset: {args.enhance}. Available presets: {', '.join(grader.enhancer.enhancement_presets.keys())}")
        return

    # Load custom preset if specified
    if args.load_preset:
        try:
            grader.load_presets(args.load_preset)
            logger.info(f"Loaded custom preset from {args.load_preset}")
        except Exception as e:
            logger.error(f"Failed to load preset: {e}")
            return

    # Use custom parameters if provided
    custom_params = None
    if any([args.warmth, args.contrast, args.saturation, args.shadows, args.highlights, args.vibrance]):
        custom_params = {
            'warmth': args.warmth,
            'contrast': args.contrast,
            'saturation': args.saturation,
            'shadows': args.shadows,
            'highlights': args.highlights,
            'vibrance': args.vibrance
        }
        if args.save_preset:
            try:
                grader.create_preset(args.save_preset, **custom_params)
                grader.save_presets(args.save_preset + '.json')
                logger.info(f"Saved custom preset as {args.save_preset}.json")
            except Exception as e:
                logger.error(f"Failed to save preset: {e}")

    # GUI selection mode
    if args.select:
        logger.info("Starting photo selection mode")
        results = grader.process_selected_photos(
            preset=args.preset,
            output_dir=args.output,
            max_workers=args.workers,
            enhance=args.enhance is not None,
            enhancement_preset=args.enhance
        )
        if results:
            successful = sum(1 for r in results if r['status'] == 'success')
            logger.info(f"Successfully processed {successful}/{len(results)} photos")
        return

    # Batch processing mode
    if args.batch and args.input:
        if os.path.isdir(args.input):
            image_files = grader.scan_directory_recursive(args.input, args.recursive)
            if image_files:
                output_dir = args.output or os.path.join(args.input, 'color_graded_output')
                os.makedirs(output_dir, exist_ok=True)

                logger.info(f"Starting batch processing of {len(image_files)} images")
                start_time = time.time()

                results = grader.batch_process(
                    image_files,
                    output_dir,
                    args.preset,
                    max_workers=args.workers,
                    enhance=args.enhance is not None,
                    enhancement_preset=args.enhance
                )

                summary = {
                    'total': len(results),
                    'successful': sum(1 for r in results if r['status'] == 'success'),
                    'failed': sum(1 for r in results if r['status'] == 'error'),
                    'duration': time.time() - start_time
                }
                with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Batch processing completed in {summary['duration']:.2f}s")
                logger.info(f"Summary: {summary['successful']} successful, {summary['failed']} failed")
            else:
                logger.warning("No image files found in directory")
        else:
            logger.error("Input path must be a directory for batch processing")
        return

    # Single file processing
    if args.input and os.path.isfile(args.input):
        logger.info(f"Processing single image: {args.input}")
        try:
            if args.preview:
                img = Image.open(args.input)
                processed_img = grader.apply_color_grading(
                    args.input,
                    None,
                    args.preset,
                    custom_params,
                    args.quality,
                    enhance=args.enhance is not None,
                    enhancement_preset=args.enhance
                )
                processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                user_input = input("Save result? (y/n): ").lower()
                if user_input == 'y':
                    output_path = grader.apply_color_grading(
                        args.input,
                        args.output,
                        args.preset,
                        custom_params,
                        args.quality,
                        enhance=args.enhance is not None,
                        enhancement_preset=args.enhance
                    )
                    logger.info(f"Graded image saved to: {output_path}")
                else:
                    logger.info("Processing cancelled by user")
            else:
                output_path = grader.apply_color_grading(
                    args.input,
                    args.output,
                    args.preset,
                    custom_params,
                    args.quality,
                    enhance=args.enhance is not None,
                    enhancement_preset=args.enhance
                )
                logger.info(f"Graded image saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {args.input}: {e}")
        return

    # Interactive mode
    logger.info("Starting interactive mode")
    print("🎨 AI Color Grading Tool (Enhanced)")
    print("1. Select multiple photos (GUI)")
    print("2. Process single photo")
    print("3. Batch process directory")
    print("4. Create custom preset")
    print("5. Generate sample config file")

    choice = input("Enter choice (1-5): ").strip()

    if choice == '1':
        results = grader.process_selected_photos(
            preset=args.preset,
            output_dir=args.output,
            max_workers=args.workers,
            enhance=args.enhance is not None,
            enhancement_preset=args.enhance
        )
        if results:
            successful = sum(1 for r in results if r['status'] == 'success')
            logger.info(f"Successfully processed {successful}/{len(results)} photos")

    elif choice == '2':
        file_path = input("Enter image path: ").strip()
        if os.path.isfile(file_path):
            try:
                if args.preview:
                    img = Image.open(file_path)
                    processed_img = grader.apply_color_grading(
                        file_path,
                        None,
                        args.preset,
                        custom_params,
                        args.quality,
                        enhance=args.enhance is not None,
                        enhancement_preset=args.enhance
                    )
                    processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                    user_input = input("Save result? (y/n): ").lower()
                    if user_input == 'y':
                        output_path = grader.apply_color_grading(
                            file_path,
                            None,
                            args.preset,
                            custom_params,
                            args.quality,
                            enhance=args.enhance is not None,
                            enhancement_preset=args.enhance
                        )
                        logger.info(f"Graded image saved to: {output_path}")
                    else:
                        logger.info("Processing cancelled by user")
                else:
                    output_path = grader.apply_color_grading(
                        file_path,
                        None,
                        args.preset,
                        custom_params,
                        args.quality,
                        enhance=args.enhance is not None,
                        enhancement_preset=args.enhance
                    )
                    logger.info(f"Graded image saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        else:
            logger.error("File not found")

    elif choice == '3':
        dir_path = input("Enter directory path: ").strip()
        if os.path.isdir(dir_path):
            image_files = grader.scan_directory_recursive(dir_path, True)
            if image_files:
                output_dir = args.output or os.path.join(dir_path, 'color_graded_output')
                start_time = time.time()

                results = grader.batch_process(
                    image_files,
                    output_dir,
                    args.preset,
                    max_workers=args.workers,
                    enhance=args.enhance is not None,
                    enhancement_preset=args.enhance
                )

                summary = {
                    'total': len(results),
                    'successful': sum(1 for r in results if r['status'] == 'success'),
                    'failed': sum(1 for r in results if r['status'] == 'error'),
                    'duration': time.time() - start_time
                }
                with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Batch processing completed in {summary['duration']:.2f}s")
                logger.info(f"Summary: {summary['successful']} successful, {summary['failed']} failed")
            else:
                logger.warning("No image files found in directory")
        else:
            logger.error("Directory not found")

    elif choice == '4':
        preset_name = input("Enter preset name: ").strip()
        try:
            warmth = float(input("Enter warmth (-1 to 1): ").strip())
            contrast = float(input("Enter contrast (-1 to 1): ").strip())
            saturation = float(input("Enter saturation (-1 to 1): ").strip())
            shadows = float(input("Enter shadows (-1 to 1): ").strip())
            highlights = float(input("Enter highlights (-1 to 1): ").strip())
            vibrance = float(input("Enter vibrance (-1 to 1): ").strip())

            grader.create_preset(preset_name,
                                 warmth=warmth,
                                 contrast=contrast,
                                 saturation=saturation,
                                 shadows=shadows,
                                 highlights=highlights,
                                 vibrance=vibrance)
            save_path = input("Enter file path to save preset (or press Enter for default): ").strip()
            if not save_path:
                save_path = f"{preset_name}.json"
            grader.save_presets(save_path)
            logger.info(f"Custom preset '{preset_name}' saved to {save_path}")
        except ValueError as e:
            logger.error(f"Invalid input for preset parameters: {e}")

    elif choice == '5':
        config = configparser.ConfigParser()
        config['DEFAULT'] = {
            'quality': '98',
            'workers': '4',
            'recursive': 'True',
            'enhance': 'medium',
            'preset': 'modern',
            'gpu': 'True'
        }
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        logger.info("Sample config file generated as 'config.ini'")

    else:
        logger.error("Invalid choice")


if __name__ == "__main__":
    main()