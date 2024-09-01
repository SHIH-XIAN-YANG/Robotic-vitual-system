import os
from PIL import Image

# Define the folder containing the images
folder_path = './'

# List all files in the folder and filter by image extensions
image_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# Load images
images = [Image.open(os.path.join(folder_path, img)) for img in image_files]

# Save as GIF
gif_path = 'output.gif'
images[0].save(
    gif_path, 
    save_all=True, 
    append_images=images[1:], 
    duration=500,  # Duration between frames in milliseconds
    loop=0  # 0 means infinite loop
)

print(f"GIF saved at {gif_path}")