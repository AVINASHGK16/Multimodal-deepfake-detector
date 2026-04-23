import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout,
    Concatenate, Conv2D, MaxPooling2D, BatchNormalization,
    Multiply, Lambda, Activation
)
from tensorflow.keras import Model
import tensorflow.keras.backend as K


# ─────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────
# Designed for heavily imbalanced datasets.
# Down-weights easy examples (abundant fakes the model already
# classifies correctly) and focuses training on hard examples
# (rare real samples and ambiguous cases).
# alpha=0.75 gives extra weight to the minority class (real).
# gamma=2.0 is the standard focusing parameter.

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_pred  = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce     = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal   = alpha_t * K.pow(1 - p_t, gamma) * bce
        return K.mean(focal)
    return loss_fn


# ─────────────────────────────────────────
# Cross-modal attention gate
# ─────────────────────────────────────────
def cross_modal_attention(v_features, a_features, projected_dim=256):
    # 1. Project to common dimension
    v_proj = Dense(projected_dim, activation='relu', name='v_proj')(v_features)
    v_proj = BatchNormalization(name='v_proj_bn')(v_proj)

    a_proj = Dense(projected_dim, activation='relu', name='a_proj')(a_features)
    a_proj = BatchNormalization(name='a_proj_bn')(a_proj)

    # 2. Compute attention gate from both signals
    combined = Concatenate(name='gate_input')([v_proj, a_proj])
    gate = Dense(projected_dim * 2, activation='relu', name='gate_hidden')(combined)
    gate = Dropout(0.3)(gate)
    gate = Dense(projected_dim * 2, activation='sigmoid', name='gate_output')(gate)

    # 3. Split gate and apply
    v_gate, a_gate = tf.keras.layers.Lambda(
        lambda x: tf.split(x, 2, axis=-1), name='split_gate'
    )(gate)

    v_gated = Multiply(name='v_gated')([v_proj, v_gate])
    a_gated = Multiply(name='a_gated')([a_proj, a_gate])

    # 4. Concatenate gated projections
    attended = Concatenate(name='attended_features')([v_gated, a_gated])
    return attended


# ─────────────────────────────────────────
# Audio CNN
# ─────────────────────────────────────────
def build_audio_branch(a_input):
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(a_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3 — separable for efficiency
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 4 — separable, captures higher-level speech patterns
    x = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    a_features = Dense(256, activation='relu')(x)
    a_features = BatchNormalization()(a_features)
    a_features = Dropout(0.3)(a_features)

    return a_features


# ─────────────────────────────────────────
# Main model builder
# ─────────────────────────────────────────
def build_fusion_model(freeze_xception=True):
    """
    Args:
        freeze_xception (bool):
            True  → freeze all but last 30 layers (phase 1)
            False → unfreeze everything (phase 2)
    """
    fl = focal_loss(gamma=2.0, alpha=0.75)

    # ── VISUAL BRANCH ──────────────────────────────
    v_input = Input(shape=(299, 299, 3), name="visual_input")

    xception_base = Xception(
        weights='imagenet', include_top=False, input_tensor=v_input
    )

    if freeze_xception:
        for layer in xception_base.layers[:-30]:
            layer.trainable = False
        for layer in xception_base.layers[-30:]:
            layer.trainable = True
    else:
        xception_base.trainable = True

    x_vis      = xception_base.output
    v_features = GlobalAveragePooling2D()(x_vis)
    v_features = Dropout(0.5)(v_features)

    visual_prediction = Dense(
        1, activation='sigmoid', name="visual_only_pred"
    )(v_features)

    # ── AUDIO BRANCH ───────────────────────────────
    a_input    = Input(shape=(148, 128, 1), name="audio_input")
    a_features = build_audio_branch(a_input)

    audio_prediction = Dense(
        1, activation='sigmoid', name="audio_only_pred"
    )(a_features)

    # ── CROSS-MODAL ATTENTION + FUSION ─────────────
    attended = cross_modal_attention(v_features, a_features, projected_dim=256)

    x = Dense(512, activation='relu')(attended)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    final_prediction = Dense(
        1, activation='sigmoid', name="fused_pred"
    )(x)

    model = Model(
        inputs=[v_input, a_input],
        outputs=[visual_prediction, audio_prediction, final_prediction]
    )

    # ── COMPILE with focal loss ────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "visual_only_pred": fl,
            "audio_only_pred":  fl,
            "fused_pred":       fl,
        },
        loss_weights={
            "visual_only_pred": 1.0,
            "audio_only_pred":  0.3,
            "fused_pred":       3.0,
        },
        metrics=["accuracy"]
    )

    return model


# ─────────────────────────────────────────
# Helper: recompile with focal loss for phase 2
# ─────────────────────────────────────────
def recompile_for_phase2(model):
    """Call this after unfreezing all layers for phase 2."""
    fl = focal_loss(gamma=2.0, alpha=0.75)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss={
            "visual_only_pred": fl,
            "audio_only_pred":  fl,
            "fused_pred":       fl,
        },
        loss_weights={
            "visual_only_pred": 1.0,
            "audio_only_pred":  0.3,
            "fused_pred":       3.0,
        },
        metrics=["accuracy"],
    )
    return model
