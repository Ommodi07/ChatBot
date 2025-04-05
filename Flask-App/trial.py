import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image, ImageOps
import os
from flask import Flask, request, render_template, send_file
import io
from fpdf import FPDF

app = Flask(__name__)

# Create templates and static directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

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

@app.route('/predict_mask', methods=['POST'])
def predict_mask():
    try:
        if 'image' not in request.files:
            return {'message': 'No file uploaded'}, 400
        
        file = request.files['image']
        result_io = segment_image(file)
        if result_io:
            return send_file(
                result_io,
                mimetype='image/png',
                as_attachment=True,
                download_name='segmented_image.png'
            )
        else:
            return {'message': 'processing failed'}, 500
    except Exception as e:
        return {'message': str(e)}, 500


def string_to_pdf(input_string, output_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times New Romen", size=12)
    pdf.multi_cell(0, 10, input_string)
    pdf.output(output_filename)

@app.route('/admin')
def get_mask():
    try:
        return {"message": "hello"}
    except Exception as e:
        return {'message': str(e)}, 500
    
@app.route('/generatepdf', methods=['GET', 'POST'])
def generate_pdf():
    name = request.args.get('name')
    age = request.args.get('age')
    contact = request.args.get('contact')
    address = request.args.get('address')
    data = """
        Name : {name}
        age : {age}
        contact : {contact}
        address : {address}

        Patient is having below results.
        
        """

    output_pdf = "Report.pdf"
    string_to_pdf(data, output_pdf)
    print(f"PDF generated and saved as {output_pdf}")
    return send_file(output_pdf, as_attachment=True)
        
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
            
        if file:
            try:
                result_io = segment_image(file)
                if result_io:
                    # Convert to base64 for displaying in HTML
                    import base64
                    result_io.seek(0)
                    img_base64 = base64.b64encode(result_io.getvalue()).decode()
                    
                    # Get original image in base64
                    file.seek(0)
                    orig_image = base64.b64encode(file.read()).decode()
                    
                    return render_template('upload.html', 
                                        result_image=img_base64,
                                        original_image=orig_image,
                                        show_images=True)
                else:
                    return render_template('upload.html', error='Error processing image')
            except Exception as e:
                return render_template('upload.html', error=str(e))
                
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)