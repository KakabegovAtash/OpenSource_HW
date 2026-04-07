import os
import requests
import sys

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

urls = {
    # Face detector
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    # Age predictor
    "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
    "age_net.caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel"
}

def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {out_path} ...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {out_path}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        sys.exit(1)

def main():
    print("Starting download of pre-trained models...")
    for filename, url in urls.items():
        out_path = os.path.join(MODELS_DIR, filename)
        download_file(url, out_path)
    print("All models are ready.")

if __name__ == "__main__":
    main()
