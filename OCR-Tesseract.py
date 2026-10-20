import os
import easyocr
import cv2
from pathlib import Path

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    return binary

reader = easyocr.Reader(['cs'], gpu=True)

BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "datasets" / "dataset_1"

image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
output_file_path = os.path.join(dataset_path, 'ResultsOCRgreedyAlt.txt')

with open(output_file_path, 'w', encoding='utf-8') as file:
    for image_file in image_files:
        file_path = os.path.join(dataset_path, image_file)
        print(f"Processing {image_file}...")

        # Předzpracování 
        preprocessed_image = preprocess_image(file_path)

        # Dočasné uložení 
        temp_path = os.path.join(dataset_path, "temp_preprocessed.png")
        cv2.imwrite(temp_path, preprocessed_image)

        # Provedení OCR 
        result = reader.readtext(temp_path, text_threshold=0.75, width_ths=0.7)

        extracted_texts = [entry[1] for entry in result]
        row_text = " ".join(extracted_texts)
        
        print(f"Result for {image_file}: {row_text}")
        file.write(row_text + "\n")
        print("-" * 50)

print(f"OCR results saved to {output_file_path}")
