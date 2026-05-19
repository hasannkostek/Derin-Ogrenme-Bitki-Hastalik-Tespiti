import os
import time
import copy
import pathlib

# PyTorch Local Env Workaround (OpenMP Hatası İçin)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "raw" / "plantvillage" / "plantvillage dataset" / "color"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def main():
    print("="*70)
    print("  🌱 TARIM YAPAY ZEKASI - MODEL EĞİTİMİ (Transfer Learning)")
    print("="*70)

    # 1. Cihaz (Device) Ayarı - GPU varsa GPU, yoksa CPU kullan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BİLGİ] Kullanılan Donanım (Device): {device.type.upper()}")
    if device.type == 'cuda':
        print(f"[BİLGİ] Ekran Kartı: {torch.cuda.get_device_name(0)}")

    if not DATA_DIR.exists():
        print(f"[HATA] Veri klasörü bulunamadı: {DATA_DIR}")
        print("Lütfen önce data_preparation.py betiğini başarıyla çalıştırdığınızdan emin olun.")
        return

    # 2. Data Augmentation (Veri Artırma) ve Dönüşümler
    print("\n[BİLGİ] Veri setleri yükleniyor ve Augmentation kuralları uygulanıyor...")
    
    # Eğitim Seti: Kesme, Çevirme, Döndürme, Renk Oynamaları
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Doğrulama (Validation) Seti: Sadece boyutlandırma ve kırpma (Augmentation yok)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # PyTorch ile aynı klasörden farklı transformlar çekmek için veri setini iki kez tanımlıyoruz
    full_train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transforms)

    num_classes = len(full_train_dataset.classes)
    total_images = len(full_train_dataset)
    print(f"[BİLGİ] Toplam Sınıf Sayısı: {num_classes}")
    print(f"[BİLGİ] Toplam Görüntü Sayısı: {total_images}")

    # Veriyi %80 Eğitim, %20 Doğrulama (Validation) olarak ayır
    # İki veri seti için de aynı rastgele indeksleri kullanmak önemli
    torch.manual_seed(42)
    indices = torch.randperm(total_images).tolist()
    train_size = int(0.8 * total_images)
    
    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_val_dataset, indices[train_size:])

    print(f"[BİLGİ] Eğitim Seti (Train): {len(train_dataset)} görüntü")
    print(f"[BİLGİ] Doğrulama Seti (Val): {len(val_dataset)} görüntü")

    # Dataloader'ları oluştur (num_workers Windows üzerinde hata vermemesi için 0 tutuldu)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Model Mimarisi: EfficientNet-B0
    print("\n[BİLGİ] EfficientNet-B0 modeli önceden eğitilmiş (pretrained) ağırlıklarla yükleniyor...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Son katmanı (Classifier) kendi sınıf sayımıza (21) göre ve Dropout ekleyerek değiştiriyoruz
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True), # Aşırı öğrenmeyi önlemek için %50 Dropout
        nn.Linear(num_ftrs, num_classes)
    )

    # Modeli seçili donanıma (GPU/CPU) gönder
    model = model.to(device)

    # 4. Optimizasyon ve Kayıp Fonksiyonu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Eğitim Döngüsü (Training Loop)
    print(f"\n🚀 Eğitim Başlıyor... (Toplam Epoch: {EPOCHS})")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 15)

        # Her epoch için önce eğitim (train), sonra doğrulama (val) aşaması çalışır
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modeli eğitim moduna al
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()   # Modeli değerlendirme moduna al (Dropout vb. devre dışı kalır)
                dataloader = val_loader
                dataset_size = len(val_dataset)

            running_loss = 0.0
            running_corrects = 0

            # Batchler halinde veri üzerinde dön
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Gradientleri sıfırla

                # Sadece eğitimdeyken gradient hesapla
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Geri yayılım (backprop) ve ağırlık güncellemesi (sadece train)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize():>5} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

            # Geçmişi kaydet
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Modeli yedekle (Eğer şu ana kadarki en iyi doğruluksa)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n✅ Eğitim Tamamlandı!')
    print(f'⏱️ Toplam Süre: {time_elapsed // 60:.0f} dakika {time_elapsed % 60:.0f} saniye')
    print(f'🏆 En İyi Doğruluk (Validation Accuracy): {best_acc:.4f}')

    # En iyi model ağırlıklarını yükle ve diske kaydet
    model.load_state_dict(best_model_wts)
    model_save_path = MODELS_DIR / "best_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 En iyi model başarıyla kaydedildi: {model_save_path}")

    # 6. Başarı Grafiklerini Çizdir
    plot_training_history(history)

def plot_training_history(history):
    print("\n[BİLGİ] Eğitim grafikleri hazırlanıyor...")
    epochs_range = range(1, EPOCHS + 1)

    # Grafik arka plan renkleri (Koyu Tema)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')
    
    # 1. Doğruluk (Accuracy) Grafiği
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#161b22')
    ax1.plot(epochs_range, history['train_acc'], label='Eğitim (Train)', marker='o', color='#00ff88', linewidth=2)
    ax1.plot(epochs_range, history['val_acc'], label='Doğrulama (Val)', marker='o', color='#00aaff', linewidth=2)
    ax1.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax1.set_title('Eğitim ve Doğrulama Doğruluğu (Accuracy)', color='white', pad=15)
    ax1.set_xlabel('Epoch', color='#c9d1d9')
    ax1.set_ylabel('Accuracy', color='#c9d1d9')
    ax1.grid(color='#30363d', linestyle='--', linewidth=0.5)

    # 2. Kayıp (Loss) Grafiği
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#161b22')
    ax2.plot(epochs_range, history['train_loss'], label='Eğitim (Train)', marker='o', color='#ff4455', linewidth=2)
    ax2.plot(epochs_range, history['val_loss'], label='Doğrulama (Val)', marker='o', color='#ffaa00', linewidth=2)
    ax2.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    ax2.set_title('Eğitim ve Doğrulama Kaybı (Loss)', color='white', pad=15)
    ax2.set_xlabel('Epoch', color='#c9d1d9')
    ax2.set_ylabel('Loss', color='#c9d1d9')
    ax2.grid(color='#30363d', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"📈 Grafik kaydedildi: {plot_path}")

if __name__ == "__main__":
    main()
