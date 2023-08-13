__author__ = "JasonLuo"
import zipfile

# Path to the zip file you want to extract
zip_file_path = '/home/hluo/Dataset/Abdominal_trauma/rsna-2023-abdominal-trauma-detection.zip'

# Directory where you want to extract the contents
extracted_folder_path = '/home/hluo/Dataset/Abdominal_trauma'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all contents to the specified directory
    zip_ref.extractall(extracted_folder_path)

print("Extraction complete.")
