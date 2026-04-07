import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate, Conv2D, MaxPooling2D, Reshape, LSTM
from tensorflow.keras import Model

def build_fusion_model():

    # ================= VISUAL =================
    v_input = Input(shape=(299, 299, 3), name="visual_input")

    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=v_input)
    xception_base.trainable = True

    x_vis = xception_base.output
    v_features = GlobalAveragePooling2D()(x_vis)

    visual_prediction = Dense(1, activation='sigmoid', name="visual_only_pred")(v_features)

    # ================= AUDIO =================
    a_input = Input(shape=(148, 128, 1), name="audio_input")

    x_aud = Conv2D(32, (3, 3), activation='relu', padding='same')(a_input)
    x_aud = MaxPooling2D((2, 2))(x_aud)
    x_aud = Conv2D(64, (3, 3), activation='relu', padding='same')(x_aud)
    x_aud = MaxPooling2D((2, 2))(x_aud)

    x_aud = GlobalAveragePooling2D()(x_aud)
    a_features = Dense(128, activation='relu')(x_aud)

    audio_prediction = Dense(1, activation='sigmoid', name="audio_only_pred")(a_features)

    # ================= FUSION =================
    merged = Concatenate()([v_features, a_features])

    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)

    final_prediction = Dense(1, activation='sigmoid', name="fused_pred")(x)

    model = Model(
        inputs=[v_input, a_input],
        outputs=[visual_prediction, audio_prediction, final_prediction]
    )

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={
        'visual_only_pred': 'binary_crossentropy',
        'audio_only_pred': 'binary_crossentropy',
        'fused_pred': 'binary_crossentropy'
    },
    metrics=['accuracy']
)

    return model