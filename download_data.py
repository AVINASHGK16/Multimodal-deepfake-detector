import os
import zipfile
import subprocess


# os.environ["KAGGLE_API_TOKEN"] = "YOUR_KAGGLE_TOKEN_HERE"

def download_dataset():
    # Using a live, balanced public dataset instead of the broken 'sorokin' one
    dataset_name = "rohingarg12/faceforensics-1000" 
    download_path = "./deepfake_dataset"
    
    # Safely create the folder if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading {dataset_name} into {download_path}...")
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Error downloading dataset. Make sure your internet connection is stable.")
        return
    
    
    zip_file = f"{download_path}/{dataset_name.split('/')[1]}.zip"
    if os.path.exists(zip_file):
        print("Extracting files... This might take a few minutes!")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_file) 
        print("Extraction complete! Your dataset is ready.")
    else:
        print("Download failed or zip file not found.")

if __name__ == "__main__":
    download_dataset()
