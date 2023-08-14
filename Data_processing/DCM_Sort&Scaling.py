__author__ = "JasonLuo"
# the purpose of this one is to sort all the files based on if it's injured or not
import pydicom
import os
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ImageOps
import cv2
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
source_folder = r'D:\Abdominal_Trauma\rsna-2023-abdominal-trauma-detection\train_images'
train_csv = r'D:\Abdominal_Trauma\rsna-2023-abdominal-trauma-detection\train.csv'
outdir = r'D:\Abdominal_Trauma\RSNA'
def multi_scale_analysis(image):
    image = np.array(image)
    scales = [ 5 ,7, 9,11,13 , 15]
    results = []
    for scale in scales:
        # Apply Laplacian operator with current scale
        filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        results.append(filtered)
    return results
def dicom_to_png(dicom_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Get the pixel data
    pixel_array = ds.pixel_array

    # Normalize the pixel values
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_range = pixel_max - pixel_min

    normalized_array = (pixel_array - pixel_min) / pixel_range * 255.0

    # Convert the pixel array to a PIL image
    image = Image.fromarray(normalized_array.astype('uint8'))
    return multi_scale_analysis(image)

def if_injured(patientID):
    df = pd.read_csv(train_csv, delimiter=',')
    filtered_df = df[df['patient_id'] == int(patient_ID)]
    if not filtered_df.empty:
        injury_index = str(int(filtered_df.iloc[0]['any_injury']))
        return True if injury_index == str(1) else False

def generate_save_path(patient_ID, viewPosition,if_injured):
    folder = os.path.join(outdir,'Normal') if if_injured == False else os.path.join(outdir,'Abnormal')
    subfolder = os.path.join(folder,viewPosition)
    out_path = os.path.join(subfolder,patient_ID)
    return out_path

def main():
    for subpath, dirs, files in tqdm(os.walk(source_folder)):
        for file in tqdm(files):
            file_path = os.path.join(subpath, file)
            try:
                dcm = pydicom.dcmread(file_path)
                patient_ID = dcm[0x0010, 0x0020].value
                viewPosition = dcm[0x0018, 0x5100].value
                PNG_images = dicom_to_png(file_path)
                save_path = generate_save_path(patient_ID=patient_ID, viewPosition=viewPosition,
                                               if_injured=if_injured(patient_ID))
                os.makedirs(save_path, exist_ok=True)
                basename = (os.path.basename(file).split('.')[0])
                for i, result in enumerate(PNG_images):
                    output_filename = f"{basename}_scale{i + 1}.png"
                    if os.path.exists(os.path.join(save_path, output_filename)):
                        unique_id = 1
                        while os.path.exists(os.path.join(save_path, output_filename)):
                            output_filename = f"{basename}_scale{i + 1}_{unique_id}.png"
                            unique_id += 1
                    output_path = os.path.join(save_path, output_filename)

                    cv2.imwrite(output_path, result)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()
