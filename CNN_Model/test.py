import os
from PIL import Image

input_folder = r'D:\SplitDataSet\train\Abnormal\CC'
output_folder = r'C:\Users\Woody\Desktop\results'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all PNG files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        # Resize the image to 512x512 pixels
        img_resized = img.resize((512, 512), Image.ANTIALIAS)

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

print("Resizing and saving completed.")
