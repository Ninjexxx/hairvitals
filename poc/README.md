<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-orange?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/flask-web_app-black?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/pytorch-U²Net-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/opencv-vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

<h1 align="center">💈 HairVitals</h1>

<p align="center">
  <strong>Web-based scalp diagnosis prototype powered by computer vision</strong><br>
  <em>Built on top of <a href="https://arxiv.org/abs/2406.17254">ScalpVision (MICCAI 2025)</a></em>
</p>

---

## 📸 What is this?

Upload or snap a photo of the scalp and get **instant automated analysis** — no cloud, no accounts, everything runs locally.

---

## ✨ Features

| Module | What it does | Method |
|--------|-------------|--------|
| 🧠 **Hair Segmentation** | Separates hair from scalp | U²-Net + adaptive threshold fallback |
| 📏 **Strand Thickness** | Estimates hair diameter | Erosion separation + Distance Transform |
| 🔴 **Redness Analysis** | Measures scalp inflammation | Erythema Index + Grey World normalization |
| 💧 **Oiliness Detection** | Assesses sebum levels | Specular reflection (HSV) |
| ⚠️ **Alopecia Risk** | Predicts hair loss risk | Thickness + scalp exposure |
| 🧴 **Seborrhea Risk** | Predicts seborrheic dermatitis | Oiliness + redness + texture (Laplacian) |

---

## 🚀 Quick Start

```bash
cd poc
pip install -r requirements_poc.txt
python app.py
```

Open **http://localhost:8080** in your browser.

### 📱 Mobile access

1. Find your IP: `ipconfig` (Windows)
2. Same WiFi → open `http://<YOUR_IP>:8080` on phone

---

## 📂 Structure

```
poc/
├── app.py                 # Flask server + analysis pipeline
├── templates/
│   └── index.html         # Mobile-friendly UI
├── uploads/               # User images (gitignored)
├── results/               # Generated masks (gitignored)
├── requirements_poc.txt
└── README.md
```

---

## ⚠️ Limitations

- Research prototype — not a medical device
- Thickness values are in pixels (uncalibrated)
- Results depend on lighting, distance, and focus
- Runs on CPU (~3-5s per image)

---

## 📄 References

> **Scalp Diagnostic System With Label-Free Segmentation and Training-Free Image Translation**
> Kim, Y., Kim, S., Moon, H., Yu, Y., Noh, J.
> *MICCAI 2025* — [arXiv:2406.17254](https://arxiv.org/abs/2406.17254)
