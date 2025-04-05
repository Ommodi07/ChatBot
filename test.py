import tensorflow as tf
from tensorflow.keras import layers, models

def load_and_fix_model(model_path):
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")

        # Check layers for missing weights and reinitialize
        for layer in model.layers:
            if isinstance(layer, (layers.Conv3D, layers.Dense)):
                weights = layer.get_weights()
                if not weights:  # If no weights are found
                    print(f"⚠️ Missing weights in layer '{layer.name}', initializing randomly.")
                    # Initialize weights with correct shape
                    shape = layer.get_weights()[0].shape if layer.get_weights() else layer.kernel.shape
                    new_weights = [tf.random.normal(shape), tf.zeros(shape[-1])]
                    layer.set_weights(new_weights)

        return model

    except Exception as e:
        print(f"❌ Could not load .keras model: {e}")
        return None

# Provide your .keras file path
model = load_and_fix_model("t1_unet_model.keras")
