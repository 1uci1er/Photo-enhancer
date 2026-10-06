from PIL import Image
img = Image.open("'C:\\Users\1uci1er\Desktop\AI Photo Editing\edit.jpg'")
img.verify()  # Raises exception if invalid
print(f"Mode: {img.mode}, Size: {img.size}, Format: {img.format}, Info: {img.info}")