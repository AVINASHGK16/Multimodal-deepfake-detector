import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np # Moved to the top for clean global scope

def build_fusion_model():
    """
    Builds the Multimodal Fusion Architecture.
    Combines Xception (Visual) and CNN-LSTM (Audio) streams.
    """
    # 1. Visual Stream: Processing 299x299 face crops [cite: 86]
    video_input = layers.Input(shape=(299, 299, 3), name="video_input")
    
    # FIX: Use pre-trained ImageNet weights for Transfer Learning
    base_model = tf.keras.applications.Xception(include_top=False, pooling='avg', weights='imagenet')
    
    # Optional: Freeze the base model to only train the fusion/classification layers at first
    # base_model.trainable = False 
    
    x_v = base_model(video_input)
    
    # 2. Audio Stream: Processing Mel-spectrograms [cite: 52]
    audio_input = layers.Input(shape=(128, 100, 1), name="audio_input") 
    x_a = layers.Conv2D(32, (3, 3), activation='relu')(audio_input)
    x_a = layers.MaxPooling2D((2, 2))(x_a)
    x_a = layers.Reshape((-1, x_a.shape[-1] * x_a.shape[-2]))(x_a)
    x_a = layers.LSTM(64)(x_a) # Captures temporal patterns 

    # 3. Cross-Modal Fusion [cite: 62]
    # Merging streams to catch synchronization inconsistencies [cite: 23]
    merged = layers.Concatenate()([x_v, x_a])
    
    # 4. Final Classification
    dense = layers.Dense(128, activation='relu')(merged)
    # Dropout layer added to prevent overfitting during training
    dropout = layers.Dropout(0.5)(dense) 
    output = layers.Dense(1, activation='sigmoid', name="detection_output")(dropout)

    model = models.Model(inputs=[video_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def pseudo_fake_augmentation(audio_features, shift_limit=5):
    """
    Shifts the audio spectrogram to create timing inconsistencies.
    Forces the model to learn lip-sync errors.
    """
    shift = np.random.randint(-shift_limit, shift_limit)
    return np.roll(audio_features, shift, axis=1)

def train_step(model, video_batch, audio_batch, labels, optimizer):
    """
    Custom training step applying gradients.
    """
    with tf.GradientTape() as tape:
        predictions = model([video_batch, audio_batch], training=True)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

if __name__ == "__main__":
    # Test the architecture build
    detector = build_fusion_model()
    detector.summary()
    print("Fusion model compiled successfully!")
# detector = build_fusion_model()
# detector.summary()