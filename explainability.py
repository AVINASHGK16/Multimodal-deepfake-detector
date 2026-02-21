import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam_heatmap(model, inputs, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for a dual-stream (Video + Audio) architecture.
    """
    # Map the multi-input to the activations of the target conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # 1. Unpack the dual-stream inputs (Video and Audio)
        v_input, a_input = inputs
        
        # 2. Cast both to float32 for safe gradient computation
        v_tensor = tf.cast(v_input, tf.float32)
        a_tensor = tf.cast(a_input, tf.float32)
        
        # 3. Forward pass with BOTH inputs so the model doesn't crash
        last_conv_layer_output, preds = grad_model([v_tensor, a_tensor])
        
        # 4. Extract the prediction channel
        class_channel = preds[:, 0]

    # Compute gradients of the class with respect to the spatial feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Average the gradients spatially (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Broadcast multiplication 
    weighted_feature_map = tf.multiply(last_conv_layer_output[0], pooled_grads)
    
    # Sum along the channel axis to compress into 2D heatmap
    heatmap = tf.reduce_sum(weighted_feature_map, axis=-1)

    # Apply ReLU (tf.maximum) and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()