import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import random

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="GAN-Based IDS",
    page_icon="🔐",
    layout="centered"
)

# -------------------------------
# Discriminator Model
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Load data
# -------------------------------
BASE_PATH = "."

X_test = np.load(f"{BASE_PATH}/X_test.npy")
y_test = np.load(f"{BASE_PATH}/y_test.npy")
X_val_benign = np.load(f"{BASE_PATH}/X_benign_val.npy")

input_dim = X_test.shape[1]

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_benign, dtype=torch.float32)

# -------------------------------
# Load trained discriminator
# -------------------------------
D = Discriminator(input_dim)
D.load_state_dict(torch.load(f"{BASE_PATH}/discriminator.pth", map_location="cpu"))
D.eval()

# -------------------------------
# UI Header
# -------------------------------
st.title("🔐 Live GAN-Based Intrusion Detection System")
st.write(
    "Real-time anomaly detection using a GAN discriminator trained on **normal network traffic**."
)

# -------------------------------
# Threshold calibration
# -------------------------------
st.subheader("⚙ Threshold Calibration")

percentile = st.slider(
    "Select anomaly threshold percentile (lower = stricter detection)",
    min_value=1,
    max_value=20,
    value=10
)

with torch.no_grad():
    val_scores = D(X_val_tensor).squeeze().numpy()

threshold = np.percentile(val_scores, percentile)

st.success(
    f"Dynamic Threshold ({percentile}th percentile of benign scores): **{threshold:.4f}**"
)
st.info("✅ Balanced threshold → Good precision–recall tradeoff")

# -------------------------------
# Sample selection
# -------------------------------
num_samples = st.slider(
    "Number of live samples to stream",
    min_value=10,
    max_value=100,
    value=40
)

indices = random.sample(range(len(X_test)), num_samples)

# -------------------------------
# Score distribution visualization
# -------------------------------
with torch.no_grad():
    scores = D(X_test_tensor[indices]).squeeze().numpy()

benign_scores = scores[y_test[indices] == 0]
attack_scores = scores[y_test[indices] == 1]

fig, ax = plt.subplots()
ax.hist(benign_scores, bins=40, alpha=0.6, label="Benign")
ax.hist(attack_scores, bins=40, alpha=0.6, label="Attack")
ax.axvline(threshold, color="red", linestyle="--", label="Threshold")

ax.set_title("Discriminator Score Distribution")
ax.set_xlabel("Discriminator Score")
ax.set_ylabel("Frequency")
ax.legend()

st.pyplot(fig)

st.caption(
    "ℹ️ Overlap between benign and attack scores is expected in anomaly-based IDS. "
    "Samples near the threshold represent ambiguous traffic patterns."
)

# -------------------------------
# Live Detection
# -------------------------------
if st.button("▶ Start Live Detection"):
    TP = FP = FN = TN = 0
    progress = st.progress(0)

    margin = 0.005  # Low-risk buffer zone

    for i, sample_idx in enumerate(indices):
        score = scores[i]
        true_label = y_test[sample_idx]

        # Prediction logic
        if score < threshold:
            if score > threshold - margin:
                pred_label = "Low-Risk Attack"
            else:
                pred_label = "Attack"
        else:
            pred_label = "Benign"

        actual = "Attack" if true_label == 1 else "Benign"

        # Metrics calculation
        if pred_label in ["Attack", "Low-Risk Attack"] and actual == "Attack":
            TP += 1
        elif pred_label in ["Attack", "Low-Risk Attack"] and actual == "Benign":
            FP += 1
        elif pred_label == "Benign" and actual == "Attack":
            FN += 1
        else:
            TN += 1

        emoji = "⚠️" if pred_label != "Benign" else "✅"
        mark = "✔" if (pred_label == actual or
                        (pred_label == "Low-Risk Attack" and actual == "Attack")) else "✖"

        st.write(
            f"Sample **{sample_idx}** → "
            f"Score: `{score:.4f}` → "
            f"{emoji} **{pred_label}** → "
            f"True: {actual} {mark}"
        )

        progress.progress((i + 1) / num_samples)
        time.sleep(0.25)

    # -------------------------------
    # Metrics
    # -------------------------------
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    st.markdown("## 📊 Detection Summary")
    st.write(f"✅ **True Positives:** {TP}")
    st.write(f"❌ **False Negatives:** {FN}")
    st.write(f"⚠ **False Positives:** {FP}")
    st.write(f"✅ **True Negatives:** {TN}")

    st.markdown(f"**Precision:** `{precision:.4f}`")
    st.markdown(f"**Recall:** `{recall:.4f}`")
    st.markdown(f"**F1-Score:** `{f1:.4f}`")
