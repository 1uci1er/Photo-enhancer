import os
import cv2
import numpy as np
import face_recognition
import subprocess
import shutil
import argparse

# Requirements:
# - Python 3.x
# - Install dependencies:
#   pip install opencv-python face_recognition basicsr facexlib gfpgan realesrgan
# - Clone repositories and download pre-trained models:
#   - Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
#     - Download RealESRGAN_x4plus.pth and place in Real-ESRGAN/weights
#   - GFPGAN: https://github.com/TencentARC/GFPGAN
#     - Download GFPGANv1.3.pth and place in GFPGAN/experiments/pretrained_models
# - Ensure inference scripts (inference_realesrgan.py, inference_gfpgan.py) are accessible

# Paths to inference scripts (adjust according to your setup)
GFPGAN_INFERENCE = 'inference_gfpgan.py'
REALESRGAN_INFERENCE = 'inference_realesrgan.py'

def has_faces(image_path):
    """Detect if the image contains faces using face_recognition."""
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return len(face_locations) > 0
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return False

def enhance_with_gfpgan(input_path, output_dir, version='1.3', scale=2):
    """Enhance image with GFPGAN, using Real-ESRGAN for background."""
    try:
        cmd = [
            'python', GFPGAN_INFERENCE,
            '-i', input_path,
            '-o', output_dir,
            '-v', version,
            '-s', str(scale),
            '-bg_upsampler', 'realesrgan'
        ]
        subprocess.run(cmd, check=True)
        input_filename = os.path.basename(input_path)
        restored_img_path = os.path.join(output_dir, 'restored_imgs', input_filename)
        return restored_img_path
    except Exception as e:
        print(f"Error in GFPGAN enhancement for {input_path}: {e}")
        raise

def enhance_with_realesrgan(input_path, output_dir, model_name='RealESRGAN_x4plus', scale=2):
    """Enhance image with Real-ESRGAN."""
    try:
        cmd = [
            'python', REALESRGAN_INFERENCE,
            '-n', model_name,
            '-i', input_path,
            '-o', output_dir,
            '--outscale', str(scale)
        ]
        subprocess.run(cmd, check=True)
        input_filename = os.path.basename(input_path)
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_out{ext}"
        output_path = os.path.join(output_dir, output_filename)
        return output_path
    except Exception as e:
        print(f"Error in Real-ESRGAN enhancement for {input_path}: {e}")
        raise

def post_process(image_path, output_path):
    """Apply post-processing: contrast enhancement, sharpening, and saturation boost."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_sharp = cv2.filter2D(img_clahe, -1, kernel)
        # Increase saturation
        hsv = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 10)
        s = np.clip(s, 0, 255)
        hsv_enhanced = cv2.merge((h, s, v))
        img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, img_enhanced)
    except Exception as e:
        print(f"Error in post-processing {image_path}: {e}")
        raise

def enhance_image(input_path, output_path, scale=2):
    """Enhance a single image."""
    temp_dir = 'temp_enhance'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    try:
        if has_faces(input_path):
            enhanced_path = enhance_with_gfpgan(input_path, temp_dir, scale=scale)
        else:
            enhanced_path = enhance_with_realesrgan(input_path, temp_dir, scale=scale)
        post_process(enhanced_path, output_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def enhance_directory(input_dir, output_dir, scale=2):
    """Enhance all images in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {input_path}...")
            try:
                enhance_image(input_path, output_path, scale)
                print(f"Saved enhanced image to {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI-powered photo enhancement script')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output image or directory')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor')
    args = parser.parse_args()
    if os.path.isdir(args.input):
        enhance_directory(args.input, args.output, args.scale)
    else:
        enhance_image(args.input, args.output, args.scale)