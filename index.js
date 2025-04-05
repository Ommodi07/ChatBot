const form = document.getElementById('upload-form');
const loading = document.getElementById('loading');
const originalImage = document.getElementById('originalImage');
const maskImage = document.getElementById('maskImage');
const fileInput = document.getElementById('imageInput');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image file.");
        return;
    }

    // Display the original image
    originalImage.src = URL.createObjectURL(file);
    originalImage.style.display = 'block';

    // Prepare UI for loading
    loading.style.display = 'block';
    maskImage.style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);
    console.log('Form data prepared:', formData.get('image'));
    try {
        console.log('Sending request to API...');

        const response = await fetch('http://127.0.0.1:8000/predict_mask', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // console.log('Received response:', data);

        if (data.message) {
            maskImage.src = `data:image/png;base64,${data.message}`;
            maskImage.style.display = 'block';
        } else {
            throw new Error('The response does not contain a image');
        }

    } catch (error) {
        console.error('Error during request:', error);
        alert(`Error: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
});
