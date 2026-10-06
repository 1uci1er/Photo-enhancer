#!/usr/bin/env python3
"""
High-Performance AI-Powered Color Grading & Photo Enhancement Pipeline
Optimized using OpenCV vectorized operations, 8-bit LUT transforms, and thread-pooled I/O.
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import configparser
import concurrent.futures
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except ImportError:
    GPU_AVAILABLE = False


class AIPhotoEnhancer:
    """Optimized AI-assisted photo enhancement for denoising, scaling, and sharpness."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.enhancement_presets = {
            'light': {'sharpen': 0.3, 'denoise': 2.0, 'upscale': 1.0, 'detail': 0.3},
            'medium': {'sharpen': 0.5, 'denoise': 3.0, 'upscale': 1.5, 'detail': 0.5},
            'heavy': {'sharpen': 0.7, 'denoise': 5.0, 'upscale': 2.0, 'detail': 0.7},
            'ultra': {'sharpen': 0.9, 'denoise': 7.0, 'upscale': 2.0, 'detail': 0.9}
        }
        if self.use_gpu:
            logging.info("Hardware acceleration via CUDA enabled.")

    def enhance_image(self, img_bgr: np.ndarray, preset: str = 'medium') -> np.ndarray:
        """Executes full enhancement pipeline directly on BGR NumPy arrays."""
        params = self.enhancement_presets.get(preset, self.enhancement_presets['medium'])

        # 1. Denoise
        if params['denoise'] > 0:
            img_bgr = self._denoise(img_bgr, params['denoise'])

        # 2. Super-Resolution / Upscale
        if params['upscale'] > 1.0:
            img_bgr = self._super_resolution(img_bgr, params['upscale'])

        # 3. Dynamic Range & Local Contrast (CLAHE via LAB)
        img_bgr = self._enhance_clahe(img_bgr)

        # 4. Detail Enhancement (OpenCV native computational photography)
        if params['detail'] > 0:
            img_bgr = cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15 * params['detail'])

        # 5. Smart Deblur / Unsharp Mask
        if params['sharpen'] > 0:
            img_bgr = self._deblur(img_bgr, params['sharpen'])

        # 6. Final Refinement
        img_bgr = self._final_refinement(img_bgr)

        return img_bgr

    def _denoise(self, img_bgr: np.ndarray, h_param: float) -> np.ndarray:
        if self.use_gpu:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(img_bgr)
                denoised_gpu = cv2.cuda.fastNlMeansDenoisingColored(
                    gpu_frame, h=h_param, hColor=h_param, templateWindowSize=7, searchWindowSize=21
                )
                return cv2.addWeighted(denoised_gpu.download(), 0.75, img_bgr, 0.25, 0)
            except Exception as e:
                logging.warning(f"CUDA Denoising failed ({e}). Falling back to CPU.")

        denoised = cv2.fastNlMeansDenoisingColored(
            img_bgr, None, h=h_param, hColor=h_param, templateWindowSize=7, searchWindowSize=21
        )
        return cv2.addWeighted(denoised, 0.75, img_bgr, 0.25, 0)

    def _super_resolution(self, img_bgr: np.ndarray, scale_factor: float) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        upscaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Lightweight edge sharpen kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(upscaled, -1, kernel * 0.2 + np.eye(3)[1, 1] * 0.8)

    def _enhance_clahe(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _deblur(self, img_bgr: np.ndarray, strength: float) -> np.ndarray:
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(img_bgr, 1.0 + strength, gaussian, -strength, 0)

    def _final_refinement(self, img_bgr: np.ndarray) -> np.ndarray:
        # Subtle median blur on chromatic planes via LAB to remove color artifacts
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = cv2.medianBlur(a, 3)
        b = cv2.medianBlur(b, 3)
        refined = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Gamma tuning via fast 256-byte LUT table
        inv_gamma = 1.0 / 1.05
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(refined, lut)

    def detect_blur_level(self, img_bgr: np.ndarray) -> Dict[str, any]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 80:
            return {'score': laplacian_var, 'preset': 'ultra'}
        elif laplacian_var < 200:
            return {'score': laplacian_var, 'preset': 'heavy'}
        elif laplacian_var < 500:
            return {'score': laplacian_var, 'preset': 'medium'}
        return {'score': laplacian_var, 'preset': 'light'}


class AIColorGrader:
    """Hardware-efficient Color Grading pipeline using Look-Up Tables (LUTs)."""

    def __init__(self, config_path: str = 'config.ini'):
        self.presets = {
            'cinematic': {'warmth': 0.18, 'contrast': 1.25, 'saturation': 1.1, 'shadows': 0.15, 'highlights': -0.12},
            'portrait': {'warmth': 0.12, 'contrast': 1.15, 'saturation': 1.18, 'shadows': 0.2, 'highlights': -0.05},
            'landscape': {'warmth': 0.05, 'contrast': 1.22, 'saturation': 1.3, 'shadows': 0.1, 'highlights': -0.15},
            'vintage': {'warmth': 0.28, 'contrast': 1.1, 'saturation': 0.85, 'shadows': 0.25, 'highlights': -0.2},
            'modern': {'warmth': -0.08, 'contrast': 1.3, 'saturation': 1.15, 'shadows': 0.05, 'highlights': -0.22}
        }
        self.enhancer = AIPhotoEnhancer(use_gpu=True)
        self.config_path = config_path

    def analyze_image(self, img_bgr: np.ndarray) -> Dict[str, any]:
        """Performs downscaled fast palette and exposure analysis."""
        # Downscale for immediate statistical evaluation
        small_frame = cv2.resize(img_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        pixels = small_frame.reshape(-1, 3)
        kmeans = MiniBatchKMeans(n_clusters=4, batch_size=256, random_state=42, n_init=1)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_

        avg_color = np.mean(dominant_colors, axis=0)  # B, G, R
        b, g, r = avg_color[0], avg_color[1], avg_color[2]

        # Scene heuristic
        if g > b and g > r:
            scene_type = 'landscape'
        elif brightness < 95:
            scene_type = 'cinematic'
        elif contrast > 55:
            scene_type = 'modern'
        else:
            scene_type = 'portrait'

        return {
            'brightness': brightness,
            'contrast': contrast,
            'scene_type': scene_type,
            'recommended_preset': scene_type
        }

    def _build_lut(self, warmth: float, shadows: float, highlights: float) -> np.ndarray:
        """Constructs a 3-channel 256-element LUT for one-pass color mapping."""
        x = np.arange(256, dtype=np.float32)

        # Baseline Tone Curve (Highlights & Shadows)
        curve = x.copy()
        if shadows != 0:
            curve += (shadows * 35.0) * (1.0 - np.power(x / 255.0, 2))
        if highlights != 0:
            curve += (highlights * 35.0) * np.power(x / 255.0, 2)
        curve = np.clip(curve, 0, 255)

        lut_3ch = np.zeros((256, 1, 3), dtype=np.uint8)

        # Warmth shift on Blue (channel 0) and Red (channel 2)
        # BGR Ordering in OpenCV:
        b_curve = curve * (1.0 - (warmth * 0.12 * np.sin(np.pi * x / 255.0)))
        g_curve = curve
        r_curve = curve * (1.0 + (warmth * 0.15 * np.sin(np.pi * x / 255.0)))

        lut_3ch[:, 0, 0] = np.clip(b_curve, 0, 255).astype(np.uint8)
        lut_3ch[:, 0, 1] = np.clip(g_curve, 0, 255).astype(np.uint8)
        lut_3ch[:, 0, 2] = np.clip(r_curve, 0, 255).astype(np.uint8)

        return lut_3ch

    def process_image(self, img_bgr: np.ndarray, params: dict, scene_type: str = '') -> np.ndarray:
        """Applies LUT grading, saturation, contrast, and film emulation."""
        # 1. Apply single-pass color + tone mapping via LUT
        lut = self._build_lut(
            warmth=params.get('warmth', 0.0),
            shadows=params.get('shadows', 0.0),
            highlights=params.get('highlights', 0.0)
        )
        graded = cv2.LUT(img_bgr, lut)

        # 2. Contrast adjustment
        contrast = params.get('contrast', 1.0)
        if contrast != 1.0:
            graded = cv2.convertScaleAbs(graded, alpha=contrast, beta=128 * (1 - contrast))

        # 3. Saturation adjustment in HSV color space
        sat_mult = params.get('saturation', 1.0)
        if sat_mult != 1.0:
            hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
            graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 4. Film Grain for cinematic profiles
        if scene_type == 'cinematic':
            noise = np.random.normal(0, 3.5, graded.shape).astype(np.float32)
            graded = np.clip(graded.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return graded

    def process_pipeline(self, input_path: str, output_path: str, preset: Optional[str] = None,
                         enhance: bool = True, enhancement_preset: Optional[str] = None) -> str:
        """Streamlined end-to-end reading, enhancement, grading, and writing."""
        img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image could not be decoded: {input_path}")

        # Auto-detect enhancement preset if needed
        if enhance and enhancement_preset is None:
            detection = self.enhancer.detect_blur_level(img_bgr)
            enhancement_preset = detection['preset']

        # Enhancement Pass
        if enhance:
            img_bgr = self.enhancer.enhance_image(img_bgr, preset=enhancement_preset)

        # Fast Analysis
        analysis = self.analyze_image(img_bgr)
        selected_preset = preset or analysis['recommended_preset']
        params = self.presets.get(selected_preset, self.presets['modern'])

        # Grading Pass
        result_bgr = self.process_image(img_bgr, params, analysis['scene_type'])

        # Output directory structure creation
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Optimized encode and write
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        elif ext == '.png':
            cv2.imwrite(output_path, result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(output_path, result_bgr)

        return output_path

    def batch_process(self, input_files: List[str], output_dir: str, preset: Optional[str] = None,
                      max_workers: int = 4, enhance: bool = True,
                      enhancement_preset: Optional[str] = None) -> List[Dict[str, any]]:
        """Concurrent processing across multiple worker threads."""
        results = []

        def worker(path: str) -> Dict[str, any]:
            try:
                base_name = os.path.basename(path)
                name, ext = os.path.splitext(base_name)
                suffix = "_enhanced_graded" if enhance else "_graded"
                dest_path = os.path.join(output_dir, f"{name}{suffix}{ext}")

                out = self.process_pipeline(path, dest_path, preset, enhance, enhancement_preset)
                return {'status': 'success', 'input': path, 'output': out}
            except Exception as exc:
                return {'status': 'error', 'input': path, 'error': str(exc)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, p): p for p in input_files}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(input_files), desc="Processing"):
                results.append(fut.result())

        return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized Batch Image Enhancement & Color Grading")
    parser.add_argument('input', nargs='?', help='Path to target image or folder')
    parser.add_argument('-o', '--output', help='Output directory or file path')
    parser.add_argument('-p', '--preset', choices=['cinematic', 'portrait', 'landscape', 'vintage', 'modern'])
    parser.add_argument('-e', '--enhance', choices=['light', 'medium', 'heavy', 'ultra'])
    parser.add_argument('-w', '--workers', type=int, default=4, help='Concurrency worker count')
    parser.add_argument('--no-enhance', action='store_true', help='Skip enhancement, run color grading only')
    parser.add_argument('--gpu', action='store_true', help='Enable CUDA optimizations if available')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose debug logging')
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    grader = AIColorGrader()
    grader.enhancer = AIPhotoEnhancer(use_gpu=args.gpu)

    if not args.input:
        print("Please provide an input image or directory. Run with --help for CLI arguments.")
        sys.exit(1)

    enhance_active = not args.no_enhance

    # Directory Batch Mode
    if os.path.isdir(args.input):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if not image_files:
            logging.error("No valid images found in directory.")
            return

        out_dir = args.output or os.path.join(args.input, "graded_output")
        logging.info(f"Processing {len(image_files)} files to: {out_dir}")
        t0 = time.time()
        results = grader.batch_process(
            image_files, out_dir, preset=args.preset,
            max_workers=args.workers, enhance=enhance_active,
            enhancement_preset=args.enhance
        )
        elapsed = time.time() - t0
        success_count = sum(1 for r in results if r['status'] == 'success')
        logging.info(f"Completed {success_count}/{len(image_files)} images in {elapsed:.2f}s")

    # Single File Mode
    elif os.path.isfile(args.input):
        out_path = args.output
        if not out_path:
            name, ext = os.path.splitext(args.input)
            suffix = "_enhanced_graded" if enhance_active else "_graded"
            out_path = f"{name}{suffix}{ext}"

        t0 = time.time()
        try:
            res = grader.process_pipeline(
                args.input, out_path, preset=args.preset,
                enhance=enhance_active, enhancement_preset=args.enhance
            )
            logging.info(f"Saved: {res} (in {time.time() - t0:.2f}s)")
        except Exception as e:
            logging.error(f"Failed to process image: {e}")


if __name__ == '__main__':
    main()