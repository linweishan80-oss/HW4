# app.py
# CIFAR-10 Streamlit Demo (aligned with your train.py)
# - Model: SimpleCNN (same architecture)
# - Weights: cifar10_model.pth (state_dict)
# - Transform: Resize(32,32) + ToTensor + Normalize(0.5,0.5,0.5)

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# =========================
# 0) Page / Config
# =========================
st.set_page_config(page_title="CIFAR-10 Image Classification Demo", layout="wide")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "cifar10_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1) Model (EXACT SAME as your train.py)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32->16

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16->8

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8->4
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),  # 256 * 4 * 4 = 4096
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# =========================
# 2) Transform (aligned with your training normalize)
# =========================
def build_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# =========================
# 3) Model load (state_dict)
# =========================
def _load_state_dict(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Your cifar10_model.pth is not a state_dict (dict).")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@st.cache_resource
def load_model_cached(mtime: float):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = SimpleCNN()
    model = _load_state_dict(model, MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return model


def load_model():
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else -1.0
    return load_model_cached(mtime)


# =========================
# 4) Agent (Explain + Chat)
# =========================
TIPS = {
    "airplane": "飛機若很小或背景太雜，容易誤判。建議裁切讓主體更大、飛機更清楚。",
    "automobile": "汽車與卡車可能混淆；拍到整台車身比例、輪子與車頭更穩。",
    "bird": "鳥常因主體太小而誤判；建議裁切或換更近的照片。",
    "cat": "貓與狗有時會混淆；如果臉部/耳朵輪廓更清楚，通常更準。",
    "deer": "鹿與馬可能混淆；拍到角或身形輪廓更完整會更好。",
    "dog": "狗通常好辨識；若只拍局部或光線太暗，信心度可能下降。",
    "frog": "青蛙在草地/水邊背景複雜時較難；建議裁切主體、背景乾淨。",
    "horse": "馬與鹿可能混淆；拍到四肢與身形輪廓更清楚會更穩。",
    "ship": "船若在遠景可能被誤判；換更近、更清楚的船體更好。",
    "truck": "卡車與汽車可能混淆；拍到貨斗或車身比例更容易判對。",
}


def agent_explain(pred_label: str, pred_prob: float, topk: list[tuple[str, float]]) -> str:
    topk_str = "\n".join([f"- {c}: {p:.1%}" for c, p in topk])
    return (
        f"我判斷是 **{pred_label}**（信心度約 **{pred_prob:.1%}**）。\n\n"
        f"Top-3 機率：\n{topk_str}\n\n"
        f"建議：{TIPS.get(pred_label, '建議換更清楚、主體更大、背景更乾淨的照片再試一次。')}"
    )


def chat_agent(user_msg: str, pred_label: str | None, pred_prob: float | None, topk: list[tuple[str, float]] | None) -> str:
    msg = (user_msg or "").strip().lower()

    if any(k in msg for k in ["為什麼", "why", "解釋"]):
        if pred_label is None or pred_prob is None or topk is None:
            return "你先上傳一張圖片跑出預測後，我才能針對這張圖解釋。"
        return agent_explain(pred_label, pred_prob, topk)

    if any(k in msg for k in ["模型", "pth", "not found", "找不到", "權重", "state_dict"]):
        return (
            "如果出現「Model file not found」，代表雲端找不到 `cifar10_model.pth`。\n"
            "解法：\n"
            "1) 把 `cifar10_model.pth` 跟 `app.py` 放同一層並 push 到 GitHub；或\n"
            "2) 用左側欄 Upload 直接上傳 `cifar10_model.pth`（我會存到同層）。"
        )

    if any(k in msg for k in ["部署", "streamlit", "requirements", "cloud"]):
        return (
            "Streamlit 會執行 `app.py`，並自動讀 `requirements.txt` 安裝套件。\n"
            "重點：模型檔（pth）一定要在雲端可讀到（同層或自動下載）。"
        )

    if any(k in msg for k in ["類別", "classes", "cifar"]):
        return "CIFAR-10 共有 10 類：\n" + ", ".join(CIFAR10_CLASSES)

    return "你可以問我：為什麼判成這類？部署時模型檔找不到怎麼辦？CIFAR-10 有哪些類別？"


# =========================
# 5) UI
# =========================
st.title("CIFAR-10 Image Classification Demo (SimpleCNN + Agent)")
st.write("Upload an image (JPG/PNG). The model outputs Prediction + Top-3 + Agent explanation.")

with st.sidebar:
    st.header("Model / Deployment")
    st.caption(f"Device: {DEVICE.upper()}")

    if MODEL_PATH.exists():
        st.success(f"Found: {MODEL_PATH.name}")
    else:
        st.warning("Missing: cifar10_model.pth")

    up_model = st.file_uploader("Upload cifar10_model.pth (optional)", type=["pth"])
    if up_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(up_model.read())
        st.success("Saved cifar10_model.pth. Please refresh / rerun the app.")

    st.divider()
    st.caption("Note: This model expects CIFAR-10 style input; app will resize to 32×32.")

# load model
model = None
model_error = None
try:
    model = load_model()
except Exception as e:
    model_error = str(e)

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Upload Image")
    img_file = st.file_uploader("Choose an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

    pred_label = None
    pred_prob = None
    topk = None

    if model_error:
        st.error("Model not ready. Please provide `cifar10_model.pth`.")
        st.code(model_error)
    else:
        if img_file is not None:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)

            tfm = build_transform()
            x = tfm(img).unsqueeze(0).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            dt_ms = (time.time() - t0) * 1000

            idx = int(np.argmax(probs))
            pred_label = CIFAR10_CLASSES[idx]
            pred_prob = float(probs[idx])

            top_idx = np.argsort(-probs)[:3]
            topk = [(CIFAR10_CLASSES[int(i)], float(probs[int(i)])) for i in top_idx]

            st.success(f"Prediction: {pred_label}")
            st.write(f"Inference time: **{dt_ms:.1f} ms**")

            st.markdown("**Top-3 probabilities**")
            for c, p in topk:
                st.write(f"- {c}: {p:.1%}")

            st.divider()
            st.subheader("Agent 解釋（自動）")
            st.info(agent_explain(pred_label, pred_prob, topk))
        else:
            st.info("請先上傳一張圖片。")

with right:
    st.subheader("Chat Agent")
    st.caption("可問：為什麼判成這類？模型檔找不到怎麼辦？CIFAR-10 有哪些類別？")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("輸入你的問題…")
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        reply = chat_agent(user_msg, pred_label, pred_prob, topk)
        st.session_state.chat.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
