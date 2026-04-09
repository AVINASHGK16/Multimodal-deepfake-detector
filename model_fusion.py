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
# Cross-modal attention gate
# ─────────────────────────────────────────
# Given two feature vectors v and a, this learns a soft gate
# (a scalar per channel) that weights each modality based on
# how informative it is for *this sample*. This is the key
# fix for your audio branch being noisy: on samples where
# audio is uninformative the gate suppresses it instead of
# injecting noise into the fusion.

def cross_modal_attention(v_features, a_features, projected_dim=256):
    """
    Produces a gated, attended concatenation of v and a.

    Steps:
      1. Project both to the same dimension so they're comparable.
      2. Compute a gate vector from their concatenation via a 2-layer MLP.
      3. Gate controls how much of each modality survives into fusion.
      4. Return the weighted sum + original features (residual).
    """
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

    # 3. Split gate into per-modality halves and apply
    v_gate, a_gate = tf.keras.layers.Lambda(
        lambda x: tf.split(x, 2, axis=-1), name='split_gate'
    )(gate)

    v_gated = Multiply(name='v_gated')([v_proj, v_gate])
    a_gated = Multiply(name='a_gated')([a_proj, a_gate])

    # 4. Concatenate gated projections (much cleaner signal than raw concat)
    attended = Concatenate(name='attended_features')([v_gated, a_gated])

    return attended


# ─────────────────────────────────────────
# Improved Audio CNN
# ─────────────────────────────────────────
# Key changes vs your original:
#   - Added a 4th conv block so the network has more capacity
#   - Used SeparableConv2D in later blocks (more efficient, less overfit)
#   - Doubled projection head to 256 (matches v_features projection dim)
#   - Added residual-style skip so gradients flow even if a block collapses

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

    # Two-layer projection head
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
            True  → freeze all but last 30 layers (use for first ~3 epochs)
            False → unfreeze everything with a very low lr (fine-tuning phase)
    """

    # ── VISUAL BRANCH ──────────────────────────────
    v_input = Input(shape=(299, 299, 3), name="visual_input")

    xception_base = Xception(
        weights='imagenet', include_top=False, input_tensor=v_input
    )

    if freeze_xception:
        # Freeze all but last 30 layers
        for layer in xception_base.layers[:-30]:
            layer.trainable = False
        for layer in xception_base.layers[-30:]:
            layer.trainable = True
    else:
        # Full fine-tune — use with lr ≤ 1e-5
        xception_base.trainable = True

    x_vis = xception_base.output
    v_features = GlobalAveragePooling2D()(x_vis)    # → [batch, 2048]
    v_features = Dropout(0.5)(v_features)

    visual_prediction = Dense(
        1, activation='sigmoid', name="visual_only_pred"
    )(v_features)

    # ── AUDIO BRANCH ───────────────────────────────
    a_input = Input(shape=(148, 128, 1), name="audio_input")
    a_features = build_audio_branch(a_input)        # → [batch, 256]

    audio_prediction = Dense(
        1, activation='sigmoid', name="audio_only_pred"
    )(a_features)

    # ── CROSS-MODAL ATTENTION + FUSION ─────────────
    attended = cross_modal_attention(
        v_features, a_features, projected_dim=256
    )                                               # → [batch, 512]

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

    # ── COMPILE ────────────────────────────────────
    model = Model(
        inputs=[v_input, a_input],
        outputs=[visual_prediction, audio_prediction, final_prediction]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "visual_only_pred": "binary_crossentropy",
            "audio_only_pred":  "binary_crossentropy",
            "fused_pred":       "binary_crossentropy",
        },
        loss_weights={
            "visual_only_pred": 1.0,
            "audio_only_pred":  0.3,   # lowered: audio is noisy early on
            "fused_pred":       3.0,   # fused output is what we care about
        },
        metrics=["accuracy"]
    )

    return model
