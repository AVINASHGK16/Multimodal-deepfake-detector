import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Silences the info logs
from model_fusion import build_fusion_model

print("🧠 Building the new Phase 2 Auxiliary Architecture...")
model = build_fusion_model()

print("💾 Saving dummy weights for UI testing...")
# Save the initialized weights to a new file
model.save_weights("dummy_phase2.weights.h5")

print("✅ Success! 'dummy_phase2.weights.h5' has been generated in your folder.")