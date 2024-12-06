import os
import io
from pdf2image import convert_from_path
from google.cloud import vision
from PIL import Image, ImageDraw

# Load your Google Vision API key from the text file and set it as an environment variable
with open('API.txt', 'r') as f:
    api_key_path = f.read().strip().split('=')[-1].strip('"')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Function to convert PDF to images
def convert_pdf_to_images(pdf_path):
    # Convert PDF pages to images (300 DPI for good quality)
    pages = convert_from_path(pdf_path, 600)
    image_paths = []
    for i, page in enumerate(pages):
        image_path = f'page_{i}.png'
        page.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

# Function to detect text and remove it using Google Vision API
def remove_text_from_image(image_path):
    # Load image and send it to Google Vision API for text detection
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")

    texts = response.text_annotations

    # Open the image using Pillow
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Reduce the threshold to ensure even smaller bounding boxes are captured
    max_area_threshold = 10000000  # Lower the threshold to catch all text, adjust if needed

    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        width = abs(vertices[1][0] - vertices[0][0])
        height = abs(vertices[2][1] - vertices[0][1])
        area = width * height

        # Remove all text (numbers and letters), including small and large bounding boxes
        if area < max_area_threshold:
            draw.polygon(vertices, fill=(255, 255, 255))  # White to cover the text

    # Save the cleaned image
    cleaned_image_path = f'cleaned_{image_path}'
    img.save(cleaned_image_path)
    return cleaned_image_path

# Function to process all pages in a PDF
def process_pdf(pdf_path):
    # Convert PDF pages to PNG images
    image_paths = convert_pdf_to_images(pdf_path)
    
    # For each image, detect and remove text
    cleaned_images = []
    for image_path in image_paths:
        cleaned_image_path = remove_text_from_image(image_path)
        cleaned_images.append(cleaned_image_path)
    
    return cleaned_images

# Example usage:
pdf_path = 'BZ_50D2AA (9-15-2023).pdf'
cleaned_images = process_pdf(pdf_path)

# Optionally: Combine cleaned images back into a PDF (if needed)
import img2pdf
with open("cleaned_output.pdf", "wb") as f:
    f.write(img2pdf.convert(cleaned_images))

print("Process completed. Cleaned images saved.")
