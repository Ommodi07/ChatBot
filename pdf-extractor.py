import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image, ImageOps
import os
from flask import Flask, request, render_template, send_file
import io
from fpdf import FPDF

model = load_model("unet_3d_model.h5")

def create_overlay(original_image, mask, color=(255, 255, 0, 128)):  # Yellow with 50% transparency
    # Convert original image to RGBA if it isn't already
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    # Create a yellow overlay mask
    overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y, x] > 0:
                overlay.putpixel((x, y), color)
    
    # Combine the original image with the overlay
    return Image.alpha_composite(original_image, overlay)

def preprocess_image(image):
    # Resize to match model input shape
    image = image.resize((128, 128))
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Convert RGB to grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)
    # Reshape to match model's expected input shape (None, 128, 128, 64, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.repeat(image, 64, axis=3)    # Repeat 64 times for the required depth
    image = np.expand_dims(image, axis=-1)  # Add final singleton dimension
    return image

def postprocess_mask(mask):
    # Take maximum across the 64 channels
    mask = np.max(mask, axis=3)
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0-255
    mask = np.squeeze(mask)  # Remove extra dimensions
    return mask

def segment_image(image_file):
    try:
        # Read image from uploaded file
        image_stream = io.BytesIO(image_file.read())
        original_image = Image.open(image_stream).convert('RGB')
        
        # Preserve original size for later
        original_size = original_image.size
        
        processed_image = preprocess_image(original_image)
        print(f"Preprocessed image shape: {processed_image.shape}")
        
        # Get segmentation mask from model
        prediction = model.predict(processed_image, verbose=0)
        
        print(f"Model prediction shape: {prediction.shape}")
        
        mask = postprocess_mask(prediction)
        print(f"Final mask shape: {mask.shape}")
        
        # Resize mask to match original image size
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
        mask = np.array(mask_image)
        
        # Create overlay
        result_image = create_overlay(original_image, mask)
        
        # Save to BytesIO object
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        return img_io
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

from datetime import date

def generate_pdf(name, age, contact, address, image_path):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        dt = date.today()
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, "ENVISION-MRI REPORT", ln=True, align='C')
        pdf.ln(10)
        
        # Add a border box
        pdf.set_line_width(0.5)
        pdf.rect(5, 5, 200, 287)
        
        # Patient Details
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Patient Information:", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Name: {name}", ln=True)
        pdf.cell(0, 8, f"Age: {age}", ln=True)
        pdf.cell(0, 8, f"Contact: {contact}", ln=True)
        pdf.multi_cell(0, 8, f"Address: {address}")
        pdf.ln(5)
        
        # Radiologist Details
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Radiologist Information:", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, "Radiologist's Name: OM MODI", ln=True)
        pdf.cell(0, 8, "Signature: OM", ln=True)
        pdf.cell(0, 8, f"Date: {dt}", ln=True)
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Thank you for using ENVISION-MRI. We hope you find this report helpful.")
        pdf.ln(10)
        
        # Process the image and get segmentation result
        with open(image_path, 'rb') as img_file:
            result_io = segment_image(img_file)
            
            if result_io:
                pdf.add_page()
                pdf.set_font("Arial", style='B', size=14)
                pdf.cell(0, 10, "Segmentation Result:", ln=True, align='C')
                pdf.ln(5)
                
                # Save the segmented image temporarily
                temp_image_path = "temp_segmented.png"
                with open(temp_image_path, "wb") as f:
                    f.write(result_io.getvalue())
                
                # Add the segmented image to PDF
                pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=190)
                
                # Remove temporary image
                os.remove(temp_image_path)
            else:
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(0, 10, "Error: Could not process the X-ray image", ln=True, align='C')
        
        # Save the PDF
        output_pdf = f"Report_{name.replace(' ', '_')}.pdf"
        pdf.output(output_pdf)
        print(f"PDF report generated successfully as {output_pdf}")
        return output_pdf
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

# Update the main block to include image path
if __name__ == "__main__":
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    contact = input("Enter your contact number: ")
    address = input("Enter your address: ")
    image_path = "Flask-App\\testing images\\42.nii_slice_035.jpg"
    generate_pdf(name, age, contact, address, image_path)