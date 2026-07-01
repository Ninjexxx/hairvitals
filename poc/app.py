"""
ScalpVision PoC - Proof of Concept
===================================
Escopo:
  1. Segmentação dos fios
  2. Espessura do fio
  3. Análise de vermelhidão
  4. Oleosidade
  5. Risco de alopecia
  6. Risco de seborreia
"""

import os
import sys
import uuid
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "segmentation"))
sys.path.insert(0, str(ROOT_DIR / "segmentation" / "model"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
u2net_model = None
USE_GPU = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_device():
    import torch
    if USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ============================================================
# Pré-processamento
# ============================================================
def normalize_illumination(img):
    """Grey World color constancy."""
    b, g, r = cv2.split(img.astype(np.float64))
    avg = (b.mean() + g.mean() + r.mean()) / 3
    b = np.clip(b * (avg / (b.mean() + 1e-6)), 0, 255)
    g = np.clip(g * (avg / (g.mean() + 1e-6)), 0, 255)
    r = np.clip(r * (avg / (r.mean() + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


# ============================================================
# 1. Segmentação de cabelo
# ============================================================
def load_u2net():
    global u2net_model
    if u2net_model is not None:
        return u2net_model

    import torch
    from u2net import U2NET

    weight_paths = [
        ROOT_DIR / "u2net_tar_min.pth",
        ROOT_DIR / "segmentation" / "saved_models" / "new_seg_ckpt" / "tar_min.pth",
        ROOT_DIR / "segmentation" / "saved_models" / "u2net.pth",
    ]

    for wp in weight_paths:
        if wp.exists():
            device = get_device()
            net = U2NET(3, 1)
            net.load_state_dict(torch.load(str(wp), map_location=device))
            net.to(device)
            net.eval()
            u2net_model = net
            print(f"[INFO] U²-Net carregado de: {wp}")
            return u2net_model

    print("[WARN] Peso do U²-Net não encontrado.")
    return None


def segment_hair(image_path):
    """Segmenta fios de cabelo usando U²-Net com fallback."""
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage

    model = load_u2net()
    if model is None:
        return segment_hair_fallback(image_path)

    device = get_device()
    img = PILImage.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        d1, *_ = model(input_tensor)

    pred = d1.squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred * 255).astype(np.uint8)

    mask = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)

    # Threshold com Otsu
    otsu_thresh, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_thresh < 50:
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Validação de cobertura
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    coverage = cv2.countNonZero(binary_mask) / total_pixels
    if coverage > 0.85:
        _, binary_mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        coverage = cv2.countNonZero(binary_mask) / total_pixels
        if coverage > 0.85:
            return segment_hair_fallback(image_path)

    return binary_mask


def segment_hair_fallback(image_path):
    """Fallback: segmentação por threshold adaptativo."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((480, 640), dtype=np.uint8)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Imagem muito uniforme = sem fios
    if np.std(gray) < 15:
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    thresh1 = cv2.adaptiveThreshold(
        l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(thresh1, thresh2)

    # Limpar ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Filtra componentes alongados
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    result = np.zeros_like(cleaned)
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 50:
            continue
        aspect = max(cw, ch) / (min(cw, ch) + 1)
        if aspect > 2.0:
            result[labels == i] = 255

    # Cobertura excessiva = falso positivo
    if cv2.countNonZero(result) / (h * w) > 0.85:
        return np.zeros((h, w), dtype=np.uint8)

    return result



# ============================================================
# 3. Análise de vermelhidão (Índice de Eritema)
# ============================================================
def analyze_redness(img, scalp_mask):
    """Índice de Eritema: EI = 100 * (log(1/R) - log(1/G))."""
    img_norm = normalize_illumination(img)
    b, g, r = cv2.split(img_norm.astype(np.float64))
    r = np.clip(r, 1, 255)
    g = np.clip(g, 1, 255)

    ei = 100 * (np.log(1.0 / r) - np.log(1.0 / g))
    pixels = scalp_mask > 0

    if np.sum(pixels) == 0:
        return 0.0, "Normal"

    ei_mean = float(np.mean(ei[pixels]))

    # Score: pele normal EI ~ -30 a 5, eritema > 5
    if ei_mean <= 5:
        score = max(0, (ei_mean + 30) / 35 * 15)
    elif ei_mean <= 20:
        score = 15 + (ei_mean - 5) / 15 * 30
    elif ei_mean <= 35:
        score = 45 + (ei_mean - 20) / 15 * 25
    else:
        score = min(100, 70 + (ei_mean - 35) / 20 * 30)
    score = max(0, min(100, score))

    if score > 65:
        label = "Alta vermelhidão (possível eritema)"
    elif score > 35:
        label = "Vermelhidão moderada"
    elif score > 15:
        label = "Vermelhidão leve"
    else:
        label = "Normal"

    return round(score, 1), label


# ============================================================
# 4. Oleosidade (reflexão especular)
# ============================================================
def analyze_oiliness(img, scalp_mask):
    """Detecta oleosidade via reflexos especulares (V alto + S baixo)."""
    img_norm = normalize_illumination(img)
    hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    pixels = scalp_mask > 0
    if np.sum(pixels) == 0:
        return 0.0, "Normal"

    s_vals = s[pixels]
    v_vals = v[pixels]

    # Reflexo especular: V > 200 e S < 40
    specular = (v_vals > 200) & (s_vals < 40)
    specular_ratio = float(np.sum(specular)) / len(v_vals) * 100

    if specular_ratio > 15:
        score = min(100, 60 + specular_ratio)
    elif specular_ratio > 5:
        score = 30 + (specular_ratio - 5) / 10 * 30
    elif specular_ratio > 2:
        score = 10 + (specular_ratio - 2) / 3 * 20
    else:
        score = specular_ratio / 2 * 10
    score = max(0, min(100, score))

    if score > 65:
        label = "Alta oleosidade"
    elif score > 35:
        label = "Oleosidade moderada"
    elif score > 15:
        label = "Oleosidade leve"
    else:
        label = "Normal"

    return round(score, 1), label


# ============================================================
# 5. Risco de alopecia
# ============================================================
def predict_alopecia_risk(scalp_coverage, hair_coverage):
    """Prediz risco de alopecia baseado na cobertura capilar."""
    risk_score = 0
    factors = []

    # Cobertura do couro exposto (principal indicador sem calibração)
    if scalp_coverage > 85:
        risk_score += 40
        factors.append("Alta exposição do couro cabeludo")
    elif scalp_coverage > 70:
        risk_score += 25
        factors.append("Couro cabeludo moderadamente exposto")
    elif scalp_coverage > 55:
        risk_score += 10
        factors.append("Leve exposição do couro cabeludo")

    # Baixa cobertura de cabelo
    if hair_coverage < 0.10:
        risk_score += 35
        factors.append("Muito pouco cabelo detectado na imagem")
    elif hair_coverage < 0.25:
        risk_score += 15
        factors.append("Baixa densidade capilar detectada")

    risk_score = min(100, risk_score)

    if risk_score >= 50:
        risk_class = "Alto risco"
        rec = "Consulte um dermatologista/tricologista."
    elif risk_score >= 25:
        risk_class = "Risco moderado"
        rec = "Acompanhamento periódico recomendado."
    else:
        risk_class = "Baixo risco"
        rec = "Couro cabeludo aparenta estar saudável."

    return {
        'risk_score': risk_score,
        'risk_class': risk_class,
        'risk_factors': factors,
        'recommendation': rec,
        'status': 'OK'
    }


# ============================================================
# 6. Risco de seborreia
# ============================================================
def predict_seborrhea_risk(oiliness_score, redness_score, img, scalp_mask):
    """Prediz risco de dermatite seborreica baseado em oleosidade + vermelhidão + descamação."""
    risk_score = 0
    factors = []

    # Oleosidade alta = fator principal
    if oiliness_score > 60:
        risk_score += 35
        factors.append("Alta oleosidade no couro cabeludo")
    elif oiliness_score > 30:
        risk_score += 15
        factors.append("Oleosidade moderada")

    # Vermelhidão associada
    if redness_score > 50:
        risk_score += 25
        factors.append("Vermelhidão significativa (possível inflamação)")
    elif redness_score > 25:
        risk_score += 10
        factors.append("Vermelhidão leve")

    # Detecção de descamação (textura irregular no couro)
    pixels = scalp_mask > 0
    if np.sum(pixels) > 100:
        img_norm = normalize_illumination(img)
        gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        # Variância local indica textura irregular (descamação)
        local_var = cv2.Laplacian(gray, cv2.CV_64F)
        scalp_var = float(np.std(local_var[pixels]))
        if scalp_var > 40:
            risk_score += 20
            factors.append("Textura irregular detectada (possível descamação)")
        elif scalp_var > 25:
            risk_score += 10
            factors.append("Leve irregularidade na textura do couro")

    risk_score = min(100, risk_score)

    if risk_score >= 50:
        risk_class = "Alto risco de seborreia"
        rec = "Sinais compatíveis com dermatite seborreica. Consulte um dermatologista."
    elif risk_score >= 25:
        risk_class = "Risco moderado"
        rec = "Monitore oleosidade e use shampoo antifúngico se necessário."
    else:
        risk_class = "Baixo risco"
        rec = "Sem sinais significativos de seborreia."

    return {
        'risk_score': risk_score,
        'risk_class': risk_class,
        'risk_factors': factors,
        'recommendation': rec,
        'status': 'OK'
    }


# ============================================================
# Pipeline completo
# ============================================================
def run_full_analysis(image_path):
    results = {
        'timestamp': datetime.now().isoformat(),
        'image': os.path.basename(image_path),
        'modules': {}
    }

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Não foi possível ler a imagem'}

    # 1. Segmentação
    try:
        mask = segment_hair(image_path)
        method = 'U²-Net' if u2net_model else 'Fallback'
        results['modules']['segmentation'] = {'method': method, 'status': 'OK'}
    except Exception as e:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        results['modules']['segmentation'] = {'method': 'Erro', 'status': str(e)}

    # Salva máscara
    mask_filename = f"mask_{os.path.basename(image_path)}"
    mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
    cv2.imwrite(mask_path, mask)
    results['mask_file'] = mask_filename

    # Prepara máscara do couro (inverso da máscara de cabelo)
    if mask.shape[:2] != img.shape[:2]:
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
    hair_coverage = cv2.countNonZero(mask_resized) / total_pixels

    if hair_coverage > 0.90 or hair_coverage == 0:
        scalp_mask = np.ones_like(mask_resized) * 255
    else:
        scalp_mask = cv2.bitwise_not(mask_resized)

    scalp_coverage = round((1 - hair_coverage) * 100, 1)

    # 2. Vermelhidão
    try:
        redness_score, redness_class = analyze_redness(img, scalp_mask)
        results['modules']['redness'] = {
            'score': redness_score,
            'class': redness_class,
            'status': 'OK'
        }
    except Exception as e:
        redness_score = 0
        results['modules']['redness'] = {'status': f'Erro: {e}'}

    # 3. Oleosidade
    try:
        oiliness_score, oiliness_class = analyze_oiliness(img, scalp_mask)
        results['modules']['oiliness'] = {
            'score': oiliness_score,
            'class': oiliness_class,
            'status': 'OK'
        }
    except Exception as e:
        oiliness_score = 0
        results['modules']['oiliness'] = {'status': f'Erro: {e}'}

    # 4. Risco de alopecia
    try:
        alopecia = predict_alopecia_risk(scalp_coverage, hair_coverage)
        results['modules']['alopecia_risk'] = alopecia
    except Exception as e:
        results['modules']['alopecia_risk'] = {'status': f'Erro: {e}'}

    # 5. Risco de seborreia
    try:
        seborrhea = predict_seborrhea_risk(oiliness_score, redness_score, img, scalp_mask)
        results['modules']['seborrhea_risk'] = seborrhea
    except Exception as e:
        results['modules']['seborrhea_risk'] = {'status': f'Erro: {e}'}

    # Resumo
    results['summary'] = generate_summary(results['modules'])

    return results


def generate_summary(modules):
    items = []

    r = modules.get('redness', {})
    if r.get('status') == 'OK':
        items.append(f"Vermelhidão: {r['class']}")

    o = modules.get('oiliness', {})
    if o.get('status') == 'OK':
        items.append(f"Oleosidade: {o['class']}")

    a = modules.get('alopecia_risk', {})
    if a.get('status') == 'OK':
        items.append(f"Alopecia: {a['risk_class']}")

    s = modules.get('seborrhea_risk', {})
    if s.get('status') == 'OK':
        items.append(f"Seborreia: {s['risk_class']}")

    return " | ".join(items) if items else "Análise incompleta."


# ============================================================
# Rotas Flask
# ============================================================
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Arquivo inválido'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        return jsonify(run_full_analysis(filepath))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/capture', methods=['POST'])
def capture_image():
    import base64
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Nenhuma imagem recebida'}), 400

    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    unique_name = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    try:
        return jsonify(run_full_analysis(filepath))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    print("=" * 50)
    print("  ScalpVision PoC - Diagnóstico Capilar")
    print("=" * 50)

    print("\n  Carregando U²-Net...")
    try:
        load_u2net()
        print("  Modelo carregado!")
    except Exception as e:
        print(f"  Modelo indisponível: {e}")
    print()

    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=True)
