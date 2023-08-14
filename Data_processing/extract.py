__author__ = "JasonLuo"
import zipfile

def unzipfile(zip_file_path,dirout):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all contents to the specified directory
        zip_ref.extractall(dirout)

def main():
    zip_file_path = '/home/hluo/Dataset/Abdominal_trauma/rsna-2023-abdominal-trauma-detection.zip'
    extracted_folder_path = '/home/hluo/Dataset/Abdominal_trauma'
    unzipfile(zip_file_path=zip_file_path, dirout=extracted_folder_path)

if __name__ == '__main__':
    main()

