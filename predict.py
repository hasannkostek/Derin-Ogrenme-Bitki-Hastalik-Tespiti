import os
import random
import pathlib
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
#  AYARLAR VE SABİTLER
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "raw" / "plantvillage" / "plantvillage dataset" / "color"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pth"

# PlantVillage sınıfları ve Türkçe çevirileri
CLASS_NAMES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Potato___Early_blight", 
    "Potato___Late_blight", "Potato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", 
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
])

TURKISH_LABELS = {
    "Apple___Apple_scab": "ELMA - KARA LEKE",
    "Apple___Black_rot": "ELMA - KARA ÇÜRÜK",
    "Apple___Cedar_apple_rust": "ELMA - SEDİR PASI",
    "Apple___healthy": "ELMA - SAĞLIKLI",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "MISIR - GRİ YAPRAK LEKESİ",
    "Corn_(maize)___Common_rust_": "MISIR - PAS HASTALIĞI",
    "Corn_(maize)___Northern_Leaf_Blight": "MISIR - KUZEY YAPRAK YANIKLIĞI",
    "Corn_(maize)___healthy": "MISIR - SAĞLIKLI",
    "Potato___Early_blight": "PATATES - ERKEN YANIKLIK",
    "Potato___Late_blight": "PATATES - GEÇ YANIKLIK",
    "Potato___healthy": "PATATES - SAĞLIKLI",
    "Tomato___Bacterial_spot": "DOMATES - BAKTERİYEL LEKE",
    "Tomato___Early_blight": "DOMATES - ERKEN YANIKLIK",
    "Tomato___Late_blight": "DOMATES - GEÇ YANIKLIK",
    "Tomato___Leaf_Mold": "DOMATES - YAPRAK KÜFÜ",
    "Tomato___Septoria_leaf_spot": "DOMATES - SEPTORIA YAPRAK LEKESİ",
    "Tomato___Spider_mites Two-spotted_spider_mite": "DOMATES - ÖRÜMCEK AKARI",
    "Tomato___Target_Spot": "DOMATES - HEDEF LEKE",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "DOMATES - SARI YAPRAK KIVIRCIK VİRÜSÜ",
    "Tomato___Tomato_mosaic_virus": "DOMATES - MOZAİK VİRÜSÜ",
    "Tomato___healthy": "DOMATES - SAĞLIKLI"
}

def load_model(device, num_classes):
    """Eğitilmiş EfficientNet-B0 modelini yükler."""
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    if not MODEL_PATH.exists():
        print(f"[HATA] Model ağırlık dosyası bulunamadı: {MODEL_PATH}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_random_image():
    """Veri setinden rastgele bir resim seçer."""
    if not DATA_DIR.exists():
        print(f"[HATA] Veri klasörü bulunamadı: {DATA_DIR}")
        sys.exit(1)
        
    classes = os.listdir(DATA_DIR)
    random_class = random.choice(classes)
    class_dir = DATA_DIR / random_class
    
    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"[HATA] {random_class} klasöründe resim bulunamadı.")
        sys.exit(1)
        
    random_image = random.choice(images)
    return class_dir / random_image, random_class

def predict_image(image_path, model, device):
    """Verilen resim için tahmin yapar."""
    # Test dönüştürücüleri (Validation ile aynı olmalı)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[HATA] Resim yüklenemedi: {e}")
        sys.exit(1)
        
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_idx = predicted.item()
    conf_score = confidence.item() * 100
    
    english_label = CLASS_NAMES[class_idx]
    turkish_label = TURKISH_LABELS.get(english_label, english_label)
    
    return turkish_label, conf_score, english_label

def main():
    parser = argparse.ArgumentParser(description="Bitki Hastalık Sınıflandırma Tahmini")
    parser.add_argument("--image", type=str, help="Tahmin edilecek resmin dosya yolu", default=None)
    args = parser.parse_args()
    
    print("="*70)
    print("  🌱 TARIM YAPAY ZEKASI - MODEL TEST VE TAHMİN")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BİLGİ] Kullanılan Donanım: {device.type.upper()}")
    
    print("\n[BİLGİ] Model yükleniyor...")
    model = load_model(device, num_classes=len(CLASS_NAMES))
    
    if args.image:
        image_path = pathlib.Path(args.image)
        if not image_path.exists():
            print(f"[HATA] Belirtilen resim bulunamadı: {image_path}")
            sys.exit(1)
        print(f"[BİLGİ] Kullanıcı tarafından belirtilen resim test ediliyor: {image_path.name}")
        true_class = "Bilinmiyor (Dışarıdan yüklendi)"
    else:
        print("[BİLGİ] Resim belirtilmedi. Veri setinden rastgele bir resim seçiliyor...")
        image_path, true_class = get_random_image()
        print(f"[BİLGİ] Seçilen Resim: {image_path.name}")
        print(f"[BİLGİ] Gerçek Sınıfı: {TURKISH_LABELS.get(true_class, true_class)}")
        
    turkish_label, conf_score, english_label = predict_image(image_path, model, device)
    
    print("-" * 70)
    print(f"🎯 Tahmin: {turkish_label} (Güven Oranı: %{conf_score:.2f})")
    print("-" * 70)

if __name__ == "__main__":
    main()
