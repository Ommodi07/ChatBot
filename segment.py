import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image, ImageOps
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

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

def segment_image(image_path, output_path):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load and process image
        original_image = Image.open(image_path).convert('RGB')
        # Preserve original size for later
        original_size = original_image.size
        
        processed_image = preprocess_image(original_image)
        print(f"Preprocessed image shape: {processed_image.shape}")
        
        print("Model loaded successfully.")
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
        
        # Save the overlaid image
        result_image.save(output_path)
        print(f"Overlaid image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False

@app.post("/predict_mask")
async def predict_mask(image: UploadFile = File(...)):
    try:
        # Create directories if they don't exist
        os.makedirs("api/inputs", exist_ok=True)
        os.makedirs("api/outputs", exist_ok=True)
        
        # Save uploaded file
        input_path = "api/inputs/input.png"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        output_path = 'api/outputs/output.png'
        success = segment_image(input_path, output_path)
        
        if success:
            # return JSONResponse(content={"message": "Report generated successfully"})
            with open(output_path, "rb") as image_file:
                import base64
                encoded_image = base64.b64encode(image_file.read()).decode()
                return JSONResponse(content={
                    "message": encoded_image
                })
        else:
            return JSONResponse(content={"message": "processing failed"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

@app.get("/admin")
async def get_mask():
    try:
        return {"message": "hello"}
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='FastAPI Backend Service')
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                       help='Host IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Port number (default: 8000)')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "segment:app",
        host=args.host,
        port=args.port,
        reload=True,
        workers=4
    )