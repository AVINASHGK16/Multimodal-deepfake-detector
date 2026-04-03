import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate, Conv2D, MaxPooling2D, Reshape, LSTM
from tensorflow.keras import Model

def build_fusion_model():
    """
    Builds a Dual-Stream Multimodal Late-Fusion Network with Auxiliary Outputs.
    This architecture outputs three distinct predictions to catch asymmetric attacks:
    1. Visual-Only Probability
    2. Audio-Only Probability
    3. Final Fused Probability
    """
    # ==========================================
    # 1. THE VISUAL STREAM (The "Eyes")
    # ==========================================
    v_input = Input(shape=(299, 299, 3), name="visual_input")
    
    # Load Xception base (pre-trained on ImageNet)
    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=v_input)
    
    # Freeze the Xception base for Phase 1 to prevent OOM errors and catastrophic forgetting
    xception_base.trainable = False 
    
    # Extract spatial features
    x_vis = xception_base.output
    v_features = GlobalAveragePooling2D(name="visual_global_pool")(x_vis)
    
    # AUXILIARY OUTPUT 1: What does the visual stream think?
    visual_prediction = Dense(1, activation='sigmoid', name="visual_only_pred")(v_features)


    # ==========================================
    # 2. THE AUDIO STREAM (The "Ears")
    # ==========================================
    # Assuming Mel-spectrogram input shape of (128, 128, 1)
    a_input = Input(shape=(128, 128, 1), name="audio_input")
    
    # 2D CNN to extract frequency patterns from the spectrogram
    x_aud = Conv2D(32, (3, 3), activation='relu', padding='same')(a_input)
    x_aud = MaxPooling2D((2, 2))(x_aud)
    x_aud = Conv2D(64, (3, 3), activation='relu', padding='same')(x_aud)
    x_aud = MaxPooling2D((2, 2))(x_aud)
    
    # Flatten spatial dimensions to create a temporal sequence for the LSTM
    # Shape becomes (32, 32 * 64) -> 32 time steps, 2048 features per step
    x_aud = Reshape((32, 32 * 64))(x_aud)
    
    # LSTM to track temporal inconsistencies (e.g., robotic pauses)
    a_features = LSTM(128, return_sequences=False, name="audio_lstm")(x_aud)
    
    # AUXILIARY OUTPUT 2: What does the audio stream think?
    audio_prediction = Dense(1, activation='sigmoid', name="audio_only_pred")(a_features)


    # ==========================================
    # 3. THE LATE FUSION ENGINE (The "Brain")
    # ==========================================
    # Concatenate the high-level features from both streams
    merged = Concatenate(name="fusion_concat")([v_features, a_features])
    
    # Dense reasoning layers
    x_fuse = Dense(256, activation='relu', name="fusion_dense_1")(merged)
    x_fuse = Dropout(0.5, name="fusion_dropout")(x_fuse)
    
    # MAIN OUTPUT 3: The final fused decision
    final_prediction = Dense(1, activation='sigmoid', name="fused_pred")(x_fuse)


    # ==========================================
    # 4. MODEL COMPILATION
    # ==========================================
    # Create the model mapping the 2 inputs to the 3 outputs
    model = Model(
        inputs=[v_input, a_input], 
        outputs=[visual_prediction, audio_prediction, final_prediction],
        name="Multimodal_Auxiliary_Deepfake_Detector"
    )
    
    # Compile the model with 3 separate loss functions
    # We assign higher weight (loss_weights) to the final fused prediction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'visual_only_pred': 'binary_crossentropy',
            'audio_only_pred': 'binary_crossentropy',
            'fused_pred': 'binary_crossentropy'
        },
        loss_weights={
            'visual_only_pred': 0.2,
            'audio_only_pred': 0.2,
            'fused_pred': 1.0
        },
        metrics=['accuracy']
    )
    
    return model