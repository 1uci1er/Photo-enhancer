#!/usr/bin/env python3
"""
AI-Powered Professional Color Grading Script (Improved Quality)
Automatically applies professional color grading and photo enhancement using AI analysis
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import tempfile
import argparse
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Tuple, List, Dict, Optional
import json
import math
from tqdm import tqdm
import configparser
import logging
import time
import psutil
import concurrent.futures

# BUGFIX: `cv2.cuda` doesn't exist on most OpenCV builds (only CUDA-enabled
# builds have it), which raises AttributeError, not ImportError -- the
# original bare `except ImportError` let that crash the whole module on
# import for anyone without a CUDA build of opencv installed.
try:
    import cupy as cp
    GPU_AVAILABLE = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
except (ImportError, AttributeError, cv2.error):
    GPU_AVAILABLE = False

    def _cp_asnumpy(arr):
        return arr
else:
    def _cp_asnumpy(arr):
        # cp.asnumpy() is a no-op passthrough when given a plain numpy
        # array, so this is safe to call on arrays that were never
        # actually moved to the GPU.
        return cp.asnumpy(arr)

class AIPhotoEnhancer:
    """Advanced AI-powered photo enhancement for super-resolution and deblurring"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.enhancement_presets = {
            'light': {'sharpen': 0.3, 'denoise': 0.2, 'upscale': 1.5, 'detail': 0.4},
            'medium': {'sharpen': 0.5, 'denoise': 0.3, 'upscale': 1.5, 'detail': 0.6},  # Reduced upscale to 1.5
            'heavy': {'sharpen': 0.7, 'denoise': 0.4, 'upscale': 2.5, 'detail': 0.8},
            'ultra': {'sharpen': 0.9, 'denoise': 0.5, 'upscale': 3.0, 'detail': 1.0}
        }
        if self.use_gpu:
            logging.info("GPU acceleration enabled")
    
    def enhance_image_ai(self, image: Image.Image, preset: str = 'medium') -> Image.Image:
        """Main AI enhancement function with GPU support"""
        params = self.enhancement_presets.get(preset, self.enhancement_presets['medium'])
        
        if image.mode != 'RGB':
            logging.debug(f"Converting image from mode {image.mode} to RGB")
            image = image.convert('RGB')
        
        img_array = np.array(image, dtype=np.float32)
        logging.debug(f"Input image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # BUGFIX: no longer eagerly converting the whole image to a cupy
        # array here. Every downstream step (cv2.resize, cv2.filter2D,
        # scipy.ndimage, ...) requires a plain numpy array, so that
        # conversion broke the pipeline whenever _ai_denoise was skipped
        # (the only step that could actually consume a cupy array).
        # GPU use is now handled locally inside _ai_denoise instead.
        
        start_memory = psutil.virtual_memory().used
        try:
            if params['denoise'] > 0:
                img_array = self._ai_denoise(img_array, params['denoise'])
            
            if params['upscale'] > 1.0:
                img_array = self._ai_super_resolution(img_array, params['upscale'])
            
            if params['sharpen'] > 0:
                img_array = self._ai_deblur(img_array, params['sharpen'])
            
            if params['detail'] > 0:
                img_array = self._enhance_details(img_array, params['detail'])
            
            img_array = self._final_refinement(img_array)
            
            if img_array.ndim == 2:
                logging.debug("Converting grayscale to RGB")
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim > 3:
                logging.error(f"Unexpected array dimensions: {img_array.shape}")
                raise ValueError("Image array has too many dimensions")
            
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            logging.debug(f"Output image shape: {img_array.shape}, dtype: {img_array.dtype}")
            return Image.fromarray(img_array)
        
        except Exception as e:
            logging.error(f"Error in enhance_image_ai: {str(e)}")
            raise
        finally:
            end_memory = psutil.virtual_memory().used
            logging.debug(f"Memory usage: {(end_memory - start_memory) / 1024**2:.2f} MB")
    
    def _ai_denoise(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        logging.debug(f"Denoising with strength {strength}, array shape: {img_array.shape}")
        if self.use_gpu:
            try:
                # BUGFIX: cv2.cuda_GpuMat.upload() requires a plain numpy
                # array; _cp_asnumpy() is a safe no-op if img_array was
                # never actually a cupy array.
                cpu_array = _cp_asnumpy(img_array).astype(np.uint8)
                img_cuda = cv2.cuda_GpuMat()
                img_cuda.upload(cpu_array)
                denoised_gpu = cv2.cuda.fastNlMeansDenoisingColored(
                    img_cuda, h=strength*10, hColor=strength*10, templateWindowSize=7, searchWindowSize=21)
                # BUGFIX: the result is a cv2.cuda_GpuMat, not a cupy
                # array -- cp.asnumpy() on it was silently wrong (or
                # would raise). GpuMat needs .download() to come back
                # to the CPU.
                denoised = denoised_gpu.download()
                return 0.7 * denoised.astype(np.float32) + 0.3 * cpu_array.astype(np.float32)
            except Exception as e:
                logging.warning(f"GPU denoising failed: {e}. Falling back to CPU.")
                img_array = _cp_asnumpy(img_array)
        
        img_cv = _cp_asnumpy(img_array).astype(np.uint8)
        if len(img_cv.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, h=strength*10, hColor=strength*10,
                                                    templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.fastNlMeansDenoising(img_cv, None, h=strength*10, templateWindowSize=7, searchWindowSize=21)
        return 0.7 * denoised.astype(np.float32) + 0.3 * _cp_asnumpy(img_array).astype(np.float32)
    
    def _ai_super_resolution(self, img_array: np.ndarray, scale_factor: float) -> np.ndarray:
        logging.debug(f"Super-resolution with scale factor {scale_factor}, array shape: {img_array.shape}")
        if scale_factor <= 1.0:
            return img_array
        
        h, w = img_array.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        img_cv = img_array.astype(np.uint8)
        upscaled = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # Changed to LANCZOS4
        
        upscaled = cv2.edgePreservingFilter(upscaled, flags=1, sigma_s=50, sigma_r=0.4)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        if len(upscaled.shape) == 3:
            for i in range(3):
                upscaled[:,:,i] = cv2.filter2D(upscaled[:,:,i], -1, kernel * 0.1)
        else:
            upscaled = cv2.filter2D(upscaled, -1, kernel * 0.1)
        
        return upscaled.astype(np.float32)
    
    def _ai_deblur(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        logging.debug(f"Deblurring with strength {strength}, array shape: {img_array.shape}")
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.astype(np.uint8)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_amount = laplacian.var()
        
        if blur_amount < 100:
            kernel_size = 5
            sigma = 2.0
        elif blur_amount < 500:
            kernel_size = 3
            sigma = 1.5
        else:
            kernel_size = 3
            sigma = 1.0
        
        blurred = gaussian_filter(img_array, sigma=sigma)
        unsharp_mask = img_array - blurred
        sharpened = img_array + strength * unsharp_mask
        sharpened = self._enhance_edges(sharpened, strength)
        
        return np.clip(sharpened, 0, 255)
    
    def _enhance_edges(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        logging.debug(f"Enhancing edges with strength {strength}, array shape: {img_array.shape}")
        kernels = {
            'horizontal': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            'vertical': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'diagonal1': np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
            'diagonal2': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        }
        
        enhanced = img_array.copy()
        
        if len(img_array.shape) == 3:
            for i in range(3):
                channel = img_array[:, :, i]
                edge_response = np.zeros_like(channel)
                for kernel in kernels.values():
                    edge_response += np.abs(cv2.filter2D(channel, -1, kernel))
                enhanced[:, :, i] = channel + strength * 0.1 * edge_response
        else:
            edge_response = np.zeros_like(img_array)
            for kernel in kernels.values():
                edge_response += np.abs(cv2.filter2D(img_array, -1, kernel))
            enhanced = img_array + strength * 0.1 * edge_response
        
        return enhanced
    
    def _enhance_details(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        logging.debug(f"Enhancing details with strength {strength}, array shape: {img_array.shape}")
        gaussian_blur = gaussian_filter(img_array, sigma=1.0)
        high_pass = img_array - gaussian_blur
        enhanced = img_array + strength * high_pass * 0.5
        enhanced = self._enhance_texture(enhanced, strength)
        return enhanced
    
    def _enhance_texture(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        logging.debug(f"Enhancing texture with strength {strength}, array shape: {img_array.shape}")
        if len(img_array.shape) == 3:
            img_cv = img_array.astype(np.uint8)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            mean_filtered = uniform_filter(l_channel, size=9)
            enhanced_l = l_channel + strength * 0.3 * (l_channel - mean_filtered)
            lab[:, :, 0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced.astype(np.float32)
        else:
            mean_filtered = uniform_filter(img_array, size=9)
            enhanced = img_array + strength * 0.3 * (img_array - mean_filtered)
            return enhanced
    
    def _final_refinement(self, img_array: np.ndarray) -> np.ndarray:
        logging.debug(f"Final refinement, array shape: {img_array.shape}, dtype: {img_array.dtype}")
        img_uint8 = img_array.astype(np.uint8)
        
        if len(img_array.shape) == 3:
            refined = np.zeros_like(img_uint8)
            for i in range(3):
                refined[:, :, i] = medfilt2d(img_uint8[:, :, i], kernel_size=3)
        else:
            refined = medfilt2d(img_uint8, kernel_size=3)
        
        enhanced = cv2.convertScaleAbs(refined, alpha=1.05, beta=2)
        gamma = 1.1
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        return enhanced.astype(np.float32)
    
    def detect_blur_level(self, image: Image.Image) -> dict:
        img_array = np.array(image)
        logging.debug(f"Detecting blur level, image shape: {img_array.shape}")
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        
        blur_score = (laplacian_var + sobel_var) / 2
        
        if blur_score < 100:
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
            'blur_level': blur_level,
            'recommended_preset': recommended_preset
        }

class AIColorGrader:
    def __init__(self, config_path: str = 'config.ini'):
        self.presets = {
            'cinematic': {'warmth': 0.2, 'contrast': 0.3, 'saturation': 0.1, 'shadows': 0.15, 'highlights': -0.1},
            'portrait': {'warmth': 0.15, 'contrast': 0.2, 'saturation': 0.2, 'shadows': 0.2, 'highlights': -0.05},
            'landscape': {'warmth': 0.1, 'contrast': 0.25, 'saturation': 0.3, 'shadows': 0.1, 'highlights': -0.15},
            'vintage': {'warmth': 0.3, 'contrast': 0.15, 'saturation': -0.1, 'shadows': 0.25, 'highlights': -0.2},
            'modern': {'warmth': -0.1, 'contrast': 0.4, 'saturation': 0.15, 'shadows': 0.05, 'highlights': -0.25}
        }
        self.enhancer = AIPhotoEnhancer(use_gpu=True)
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
        
        # BUGFIX: fitting KMeans on every pixel of a full-resolution photo
        # (e.g. 24MP = ~24 million samples) is effectively a hang -- this
        # was never tested against a real camera photo, only small images.
        # A random subsample gives the same dominant-color estimate in a
        # fraction of the time.
        colors = img_rgb.reshape(-1, 3)
        if colors.shape[0] > 10000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(colors.shape[0], size=10000, replace=False)
            sample_colors = colors[sample_idx]
        else:
            sample_colors = colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(sample_colors)
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
                          quality: int = 98, enhance: bool = True, 
                          enhancement_preset: str = None) -> str:
        try:
            start_time = time.time()
            img = Image.open(image_path)
            metadata = img.info
            # BUGFIX: many camera/phone photos are stored with an EXIF
            # orientation tag rather than pre-rotated pixels. Without
            # applying it, a large share of real-world photos (portrait
            # phone shots especially) come out enhanced but sideways.
            img = ImageOps.exif_transpose(img)
            logging.debug(f"Opened image {image_path}, mode: {img.mode}, size: {img.size}, format: {img.format}")
            
            if img.mode != 'RGB':
                logging.debug(f"Converting image from mode {img.mode} to RGB")
                img = img.convert('RGB')
            
            if enhance and enhancement_preset is None:
                blur_analysis = self.enhancer.detect_blur_level(img)
                enhancement_preset = blur_analysis['recommended_preset']
                logging.info(f"Auto-detected enhancement preset: {enhancement_preset} "
                            f"(blur score: {blur_analysis['blur_score']:.1f})")
            
            if enhance:
                img = self.enhancer.enhance_image_ai(img, enhancement_preset)
            
            # BUGFIX: the old temp filename was just "temp_<basename>" in
            # the current working directory. batch_process() runs multiple
            # images concurrently via ThreadPoolExecutor, so two images
            # with the same filename (common with subfolders, e.g. two
            # "IMG_0001.jpg" from different folders) would stomp on each
            # other's temp file mid-processing. tempfile guarantees a
            # unique path per call.
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=os.path.splitext(image_path)[1] or '.jpg',
                prefix='color_grade_tmp_'
            )
            os.close(temp_fd)
            img.save(temp_path, quality=95)
            
            try:
                analysis = self.analyze_image_ai(temp_path)
                params = custom_params or self.presets.get(preset or analysis['recommended_preset'])
                logging.debug(f"Applying preset: {preset}, params: {params}")
                
                processed_img = self._process_image_hq(img, params, analysis)
                
                if output_path is None:
                    name, ext = os.path.splitext(image_path)
                    suffix = "_enhanced_graded" if enhance else "_graded"
                    output_path = f"{name}{suffix}{ext}"
                
                # BUGFIX: this was hardcoded to 100, silently ignoring the
                # `quality` parameter (and therefore the --quality CLI flag).
                save_kwargs = {'quality': quality, 'optimize': True, 'progressive': True}
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    # BUGFIX: 'subsampling': 0 (force 4:4:4, no chroma
                    # subsampling) combined with 'optimize'/'progressive'
                    # crashes libjpeg outright with "Suspension not
                    # allowed here" on this Pillow/libjpeg build -- 100%
                    # reproducible, not data-dependent. Dropping
                    # subsampling and letting Pillow choose it
                    # automatically (it uses little/no subsampling anyway
                    # at high quality settings) avoids the crash; quality
                    # is controlled by 'quality' as before.
                    save_kwargs.update({
                        'format': 'JPEG',
                    })
                elif image_path.lower().endswith('.png'):
                    save_kwargs.update({
                        'format': 'PNG',
                        'compress_level': 1,
                        'optimize': True
                    })
                
                # BUGFIX: blindly splatting the source image's raw `.info`
                # dict as save() kwargs can throw -- info keys are
                # format-specific (e.g. a PNG's 'transparency' index isn't
                # a valid JPEG save argument), and the source/output
                # formats can differ here. Only pass through keys that are
                # safe across PIL's JPEG/PNG writers.
                safe_metadata = {k: v for k, v in metadata.items()
                                  if k in ('exif', 'icc_profile', 'dpi')}
                processed_img.save(output_path, **save_kwargs, **safe_metadata)
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
        
        img = self._apply_smart_sharpening(img)
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
    
    def _apply_smart_sharpening(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img, dtype=np.float32)
        logging.debug(f"Applying smart sharpening, array shape: {img_array.shape}, dtype: {img_array.dtype}")
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges / 255.0
        sharp_strong = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=2))
        sharp_mild = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=5))
        result_array = np.array(img, dtype=np.float32)
        sharp_strong_array = np.array(sharp_strong, dtype=np.float32)
        sharp_mild_array = np.array(sharp_mild, dtype=np.float32)
        
        for i in range(3):
            result_array[:, :, i] = (sharp_strong_array[:, :, i] * edge_mask + 
                                   sharp_mild_array[:, :, i] * (1 - edge_mask))
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def batch_process(self, input_paths: List[str], output_dir: str, preset: str = None, 
                     maintain_structure: bool = True, max_workers: int = 4, 
                     enhance: bool = True, enhancement_preset: str = None,
                     quality: int = 98):
        import threading
        from pathlib import Path
        
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
                    suffix = "_enhanced_graded" if enhance else "_graded"
                    output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")
                
                # BUGFIX: this was hardcoded to 98 regardless of what
                # quality the caller (and ultimately --quality) requested.
                result_path = self.apply_color_grading(input_path, output_path, preset, 
                                                     None, quality, enhance, enhancement_preset)
                
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
    
    def select_multiple_photos(self, title: str = "Select Photos for Color Grading") -> List[str]:
        try:
            from tkinter import filedialog, Tk
            
            root = Tk()
            root.withdraw()
            
            file_types = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.raw"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
            
            selected_files = filedialog.askopenfilenames(
                title=title,
                filetypes=file_types
            )
            
            if selected_files:
                logging.info(f"Selected {len(selected_files)} images for processing")
                return list(selected_files)
            else:
                logging.info("No files selected")
                return []
                
        except ImportError:
            logging.warning("GUI not available. Please use command line mode.")
            return []
    
    def scan_directory_recursive(self, directory: str, include_subdirs: bool = True) -> List[str]:
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
                              max_workers: int = 4, enhance: bool = True, 
                              enhancement_preset: str = None, quality: int = 98) -> List[dict]:
        selected_files = self.select_multiple_photos()
        
        if not selected_files:
            logging.info("No photos selected for processing")
            return []
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(selected_files[0]), "ai_enhanced_output")
        
        results = self.batch_process(selected_files, output_dir, preset, 
                                   max_workers=max_workers, enhance=enhance,
                                   enhancement_preset=enhancement_preset, quality=quality)
        
        return results
    
    def create_preset(self, name: str, warmth: float, contrast: float, 
                     saturation: float, shadows: float, highlights: float):
        self.presets[name] = {
            'warmth': warmth,
            'contrast': contrast,
            'saturation': saturation,
            'shadows': shadows,
            'highlights': highlights
        }
    
    def save_presets(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.presets, f, indent=2)
    
    def load_presets(self, filepath: str):
        with open(filepath, 'r') as f:
            self.presets.update(json.load(f))

def resolve_preset_and_enhancement(args, preset_map, enhancement_map, logger):
    """Use -p/--preset and -e/--enhance from the CLI when they were given;
    only fall back to the interactive numbered menu when they weren't.

    BUGFIX: -p/--preset and -e/--enhance were defined in argparse and
    documented in --help, but no code path ever read args.preset or
    args.enhance -- every branch (single file, --batch, --select) always
    called input() for a numbered menu regardless. That means those flags
    silently did nothing, and running the script non-interactively (from
    another script, a cron job, a CI pipeline, etc.) with them set would
    still block forever waiting on stdin that was never coming.
    """
    if args.preset:
        preset = args.preset
    else:
        print("Select Color Grading Preset:")
        for key, value in preset_map.items():
            print(f"{key}. {value.capitalize()}")
        preset_choice = input("Enter choice (1-5): ").strip()
        preset = preset_map.get(preset_choice, None)
        if preset is None:
            logger.error("Invalid preset choice")
            return None, None

    if args.enhance:
        enhancement_preset = args.enhance
    else:
        print("Select Enhancement Level:")
        for key, value in enhancement_map.items():
            print(f"{key}. {value.capitalize()}")
        enhance_choice = input("Enter choice (1-4): ").strip()
        enhancement_preset = enhancement_map.get(enhance_choice, 'medium')

    return preset, enhancement_preset


def main():
    parser = argparse.ArgumentParser(description='AI-Powered Professional Color Grading with Photo Enhancement')
    parser.add_argument('input', nargs='?', help='Input image or directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('-p', '--preset', choices=['cinematic', 'portrait', 'landscape', 'vintage', 'modern'], 
                       help='Color grading preset')
    parser.add_argument('-e', '--enhance', choices=['light', 'medium', 'heavy', 'ultra'], 
                       help='Enhancement preset')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    parser.add_argument('-s', '--select', action='store_true', help='Select multiple photos with GUI')
    parser.add_argument('-r', '--recursive', action='store_true', help='Include subdirectories in batch processing')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel workers for batch processing')
    parser.add_argument('-q', '--quality', type=int, default=100, help='Output quality (1-100)')  # Default to 100
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
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration if available')
    
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
    grader.enhancer = AIPhotoEnhancer(use_gpu=args.gpu)

    preset_map = {'1': 'cinematic', '2': 'portrait', '3': 'landscape', '4': 'vintage', '5': 'modern'}
    enhancement_map = {'1': 'light', '2': 'medium', '3': 'heavy', '4': 'ultra'}

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
        logger.info("Starting photo selection mode")
        preset, enhancement_preset = resolve_preset_and_enhancement(args, preset_map, enhancement_map, logger)
        if preset is None:
            return
        
        logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
        results = grader.process_selected_photos(preset=preset, output_dir=args.output,
                                               max_workers=args.workers, enhance=True,
                                               enhancement_preset=enhancement_preset, quality=args.quality)
        if results:
            successful = sum(1 for r in results if r['status'] == 'success')
            logger.info(f"Successfully processed {successful}/{len(results)} photos")
        return

    if args.batch and args.input:
        if os.path.isdir(args.input):
            preset, enhancement_preset = resolve_preset_and_enhancement(args, preset_map, enhancement_map, logger)
            if preset is None:
                return
            
            logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
            image_files = grader.scan_directory_recursive(args.input, args.recursive)
            if image_files:
                output_dir = args.output or os.path.join(args.input, 'color_graded_output')
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Starting batch processing of {len(image_files)} images")
                start_time = time.time()
                
                results = grader.batch_process(image_files, output_dir, preset,
                                             max_workers=args.workers, enhance=True,
                                             enhancement_preset=enhancement_preset, quality=args.quality)
                
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
        preset, enhancement_preset = resolve_preset_and_enhancement(args, preset_map, enhancement_map, logger)
        if preset is None:
            return
        
        logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
        try:
            if args.preview:
                img = Image.open(args.input)
                processed_img = grader.apply_color_grading(args.input, None, preset, custom_params,
                                                        args.quality, enhance=True,
                                                        enhancement_preset=enhancement_preset)
                processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                user_input = input("Save result? (y/n): ").lower()
                if user_input == 'y':
                    output_path = grader.apply_color_grading(args.input, args.output, preset, custom_params,
                                                          args.quality, enhance=True,
                                                          enhancement_preset=enhancement_preset)
                    logger.info(f"Graded image saved to: {output_path}")
                else:
                    logger.info("Processing cancelled by user")
            else:
                output_path = grader.apply_color_grading(args.input, args.output, preset, custom_params,
                                                      args.quality, enhance=True,
                                                      enhancement_preset=enhancement_preset)
                logger.info(f"Graded image saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {args.input}: {str(e)}")
        return

    logger.info("Starting interactive mode")
    print("🎨 AI Color Grading Tool (Enhanced)")
    print("1. Select multiple photos (GUI)")
    print("2. Process single photo")
    print("3. Batch process directory")
    print("4. Create custom preset")
    print("5. Generate sample config file")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == '1':
        print("Select Color Grading Preset:")
        for key, value in preset_map.items():
            print(f"{key}. {value.capitalize()}")
        preset_choice = input("Enter choice (1-5): ").strip()
        preset = preset_map.get(preset_choice, None)
        if preset is None:
            logger.error("Invalid preset choice")
            return
        
        print("Select Enhancement Level:")
        for key, value in enhancement_map.items():
            print(f"{key}. {value.capitalize()}")
        enhance_choice = input("Enter choice (1-4): ").strip()
        enhancement_preset = enhancement_map.get(enhance_choice, 'medium')
        if enhancement_preset is None:
            logger.error("Invalid enhancement choice, defaulting to medium")
            enhancement_preset = 'medium'
        
        logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
        results = grader.process_selected_photos(preset=preset, output_dir=args.output,
                                               max_workers=args.workers, enhance=True,
                                               enhancement_preset=enhancement_preset, quality=args.quality)
        if results:
            successful = sum(1 for r in results if r['status'] == 'success')
            logger.info(f"Successfully processed {successful}/{len(results)} photos")
    
    elif choice == '2':
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
            
            print("Select Enhancement Level:")
            for key, value in enhancement_map.items():
                print(f"{key}. {value.capitalize()}")
            enhance_choice = input("Enter choice (1-4): ").strip()
            enhancement_preset = enhancement_map.get(enhance_choice, 'medium')
            if enhancement_preset is None:
                logger.error("Invalid enhancement choice, defaulting to medium")
                enhancement_preset = 'medium'
            
            logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
            try:
                if args.preview:
                    img = Image.open(file_path)
                    processed_img = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                            args.quality, enhance=True,
                                                            enhancement_preset=enhancement_preset)
                    processed_img.show(title="Preview - Press Enter to save, Esc to cancel")
                    user_input = input("Save result? (y/n): ").lower()
                    if user_input == 'y':
                        output_path = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                              args.quality, enhance=True,
                                                              enhancement_preset=enhancement_preset)
                        logger.info(f"Graded image saved to: {output_path}")
                    else:
                        logger.info("Processing cancelled by user")
                else:
                    output_path = grader.apply_color_grading(file_path, None, preset, custom_params,
                                                          args.quality, enhance=True,
                                                          enhancement_preset=enhancement_preset)
                    logger.info(f"Graded image saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        else:
            logger.error("File not found")
    
    elif choice == '3':
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
            
            print("Select Enhancement Level:")
            for key, value in enhancement_map.items():
                print(f"{key}. {value.capitalize()}")
            enhance_choice = input("Enter choice (1-4): ").strip()
            enhancement_preset = enhancement_map.get(enhance_choice, 'medium')
            if enhancement_preset is None:
                logger.error("Invalid enhancement choice, defaulting to medium")
                enhancement_preset = 'medium'
            
            logger.info(f"Selected preset: {preset}, enhancement: {enhancement_preset}")
            image_files = grader.scan_directory_recursive(dir_path, True)
            if image_files:
                output_dir = os.path.join(dir_path, 'color_graded_output')
                start_time = time.time()
                
                results = grader.batch_process(image_files, output_dir, preset,
                                             max_workers=args.workers, enhance=True,
                                             enhancement_preset=enhancement_preset, quality=args.quality)
                
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
    
    elif choice == '4':
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
    
    elif choice == '5':
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'quality': '100', 'workers': '4', 'recursive': 'True', 'enhance': 'medium', 'preset': 'modern'}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        logger.info("Sample config file generated as 'config.ini'")
    
    else:
        logger.error("Invalid choice")

if __name__ == "__main__":
    main()
