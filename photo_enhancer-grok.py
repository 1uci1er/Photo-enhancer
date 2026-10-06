import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from realesrgan import RealESRGAN
from gfpgan import GFPGANer
import shutil

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
model = RealESRGAN(device, scale=4)
model.load_weights("weights/RealESRGAN_x4plus.pth")
gfpgan = GFPGANer(device, model_path="weights/GFPGANv1.3.pth", upscale=4)

def has_faces(image_path):
    """Detect faces using OpenCV's Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def detect_photo_type(image_path):
    """Detect photo type based on image statistics."""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)
    if mean_v < 50 and mean_s < 30:
        return "night"  # Low brightness and saturation
    elif has_faces(image_path):
        return "portrait"
    else:
        return "landscape"

def enhance_with_ai(image_path, output_path, is_face=False):
    """Enhance image using Real-ESRGAN or GFPGAN based on face detection."""
    img = Image.open(image_path).convert("RGB")
    if is_face:
        _, enhanced_img = gfpgan.enhance(img)
    else:
        enhanced_img = model.predict(img)
    enhanced_img.save(output_path)
    return enhanced_img

def post_process(image_path, output_path, photo_type):
    """Apply automatic adjustments based on photo type."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Color correction and white balance (simple gray world assumption)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    mean = cv2.mean(img)[:3]
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=-mean[0]+128)

    # Contrast and brightness optimization with CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpness enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    # HDR effect simulation and vibrance
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20, dtype=cv2.CV_8U)
    s = np.clip(s, 0, 255)
    v = cv2.add(v, 10, dtype=cv2.CV_8U)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Apply photo type-specific filters
    if photo_type == "night":
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # Brighten night images
    elif photo_type == "portrait":
        img = cv2.GaussianBlur(img, (5,5), 0)  # Soft focus for portraits
    elif photo_type == "landscape":
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)  # Enhance landscape details

    cv2.imwrite(output_path, img)

def create_comparison(original_path, enhanced_path, output_path):
    """Create a side-by-side before-and-after comparison."""
    orig = cv2.imread(original_path)
    enh = cv2.imread(enhanced_path)
    comparison = np.hstack((orig, enh))
    cv2.imwrite(output_path, comparison)

def enhance_image(input_path, output_dir, scale=4):
    """Enhance a single image."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    enhanced_path = os.path.join(output_dir, f"enhanced_{filename}")
    comparison_path = os.path.join(output_dir, f"comparison_{filename}")

    # Detect photo type and faces
    photo_type = detect_photo_type(input_path)
    is_face = has_faces(input_path)

    # Enhance with AI
    enhance_with_ai(input_path, enhanced_path, is_face)

    # Post-process
    post_process(enhanced_path, enhanced_path, photo_type)

    # Create comparison
    create_comparison(input_path, enhanced_path, comparison_path)

def enhance_directory(input_dir, output_dir, scale=4):
    """Enhance all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing {input_path}...")
            enhance_image(input_path, output_dir, scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered photo enhancement script")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor")
    args = parser.parse_args()
    if os.path.isdir(args.input):
        enhance_directory(args.input, args.output, args.scale)
    else:
        enhance_image(args.input, args.output, args.scale)