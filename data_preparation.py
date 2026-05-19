"""
=============================================================================
  TARIM YAPAY ZEKASI PROJESİ - VERİ HAZIRLAMA AŞAMASI
  data_preparation.py
=============================================================================
  Görev: PlantVillage veri setini Kaggle'dan indir, çıkar ve analiz et.
  Dataset: "plantvillage-dataset" (38 sınıf, ~54.000 görüntü)
=============================================================================
"""

import os
import sys
import json

# Windows konsolunu UTF-8'e zorla (Türkçe karakter ve emoji desteği)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import zipfile
import shutil
import pathlib
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlarda çalışır; grafikleri dosyaya kaydeder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
REPORTS_DIR = BASE_DIR / "reports"

KAGGLE_DATASET  = "abdallahalidev/plantvillage-dataset"   # Kaggle slug
KAGGLE_JSON_PATH = pathlib.Path.home() / ".kaggle" / "kaggle.json"

# LOW-RAM MODE AYARLARI
LOW_RAM_MODE = True
TARGET_PLANTS = ["Apple", "Corn", "Potato", "Tomato"]
MAX_IMAGES_PER_CLASS = 500

# Renk paleti (sınıf grafiklerinde kullanılır)
PALETTE = sns.color_palette("husl", 38)

# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "=", width: int = 70) -> None:
    """Terminalde bölüm başlığı yazdırır."""
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def check_kaggle_credentials() -> bool:
    """~/.kaggle/kaggle.json dosyasının varlığını ve geçerliliğini kontrol eder."""
    banner("1️⃣  KAGGLE API KONTROLÜ")
    if not KAGGLE_JSON_PATH.exists():
        print(f"""
  ❌  kaggle.json BULUNAMADI!

  Lütfen aşağıdaki adımları izleyin:

  1. https://www.kaggle.com adresine giriş yapın.
  2. Profil > Settings > API bölümüne gidin.
  3. "Create New Token" butonuna tıklayın.
  4. İndirilen kaggle.json dosyasını şu konuma kopyalayın:
       {KAGGLE_JSON_PATH}

  Klasörün yoksa önce oluşturun:
    mkdir "%USERPROFILE%\\.kaggle"
    copy "%USERPROFILE%\\Downloads\\kaggle.json" "%USERPROFILE%\\.kaggle\\kaggle.json"
""")
        return False

    try:
        creds = json.loads(KAGGLE_JSON_PATH.read_text())
        username = creds.get("username", "???")
        print(f"  ✅  kaggle.json bulundu → Kullanıcı: {username}")
        return True
    except Exception as e:
        print(f"  ❌  kaggle.json okunamadı: {e}")
        return False


def download_dataset() -> pathlib.Path:
    """Kaggle CLI ile veri setini indirir. Zaten varsa atlar."""
    banner("2️⃣  VERİ SETİ İNDİRME")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Daha önce indirilmiş zip var mı?
    existing_zips = list(RAW_DIR.glob("*.zip"))
    if existing_zips:
        print(f"  ℹ️   Mevcut zip bulundu: {existing_zips[0].name}  (indirme atlandı)")
        return existing_zips[0]

    print(f"  📥  İndiriliyor: {KAGGLE_DATASET}")
    print(f"  📂  Hedef klasör: {RAW_DIR}\n")

    # Kaggle 2.x API — singleton 'kaggle.api' nesnesini kullan
    try:
        import kaggle
        api = kaggle.api          # KaggleApi singleton (authenticate() otomatik çalışır)
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DIR),
            quiet=False,
            unzip=False,
        )
    except Exception as e:
        print(f"\n  ❌  İndirme hatası: {e}")
        sys.exit(1)

    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        print("  ❌  Zip dosyası bulunamadı! İndirme başarısız olmuş olabilir.")
        sys.exit(1)

    print(f"\n  ✅  İndirme tamamlandı: {zips[0].name}")
    return zips[0]


def extract_dataset(zip_path: pathlib.Path) -> pathlib.Path:
    """Zip dosyasını açar. Zaten açılmışsa atlar."""
    banner("3️⃣  ZIP DOSYASI AÇMA")

    extract_dir = RAW_DIR / "plantvillage"
    success_marker = extract_dir / ".extraction_success"

    # Eğer zaten başarıyla açılmışsa atla
    if success_marker.exists():
        print(f"  ℹ️   Veri zaten çıkarılmış: {extract_dir}")
        return extract_dir

    # Eğer yarım kalan işlem varsa temizle
    if extract_dir.exists():
        print(f"  🧹  Önceki yarım kalan işlem temizleniyor: {extract_dir}")
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  📦  Açılıyor: {zip_path.name}")
    print(f"  📂  Hedef: {extract_dir}\n")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        
        if LOW_RAM_MODE:
            print(f"  ⚠️  LOW-RAM MODU AKTİF: Sadece hedef bitkiler (Max {MAX_IMAGES_PER_CLASS} görsel) çıkarılacak.")
            filtered_members = []
            class_counts = collections.defaultdict(int)
            
            for m in members:
                m_lower = m.lower()
                # Orijinal zip'in gereksiz alt klasörlerini veya grayscale olanlarını atla
                if "color/" not in m_lower and "color\\" not in m_lower:
                    continue
                
                # Hedef bitki mi?
                if not any(plant.lower() in m_lower for plant in TARGET_PLANTS):
                    continue
                
                # Sınıf adını bul
                class_name = ""
                for part in m.replace("\\", "/").split("/"):
                    if "___" in part:
                        class_name = part
                        break
                
                if class_name:
                    if m_lower.endswith((".jpg", ".jpeg", ".png")):
                        if class_counts[class_name] < MAX_IMAGES_PER_CLASS:
                            class_counts[class_name] += 1
                            filtered_members.append(m)
                    else:
                        filtered_members.append(m)
            
            members_to_extract = filtered_members
        else:
            members_to_extract = members

        for member in tqdm(members_to_extract, desc="  Dosyalar çıkarılıyor", unit="dosya"):
            zf.extract(member, extract_dir)

    success_marker.touch()
    print(f"\n  ✅  Çıkarma tamamlandı → {extract_dir}")
    return extract_dir


def find_image_root(extract_dir: pathlib.Path) -> pathlib.Path:
    """
    Veri seti içindeki 'color' klasörünü bulur.
    PlantVillage zip'i 3 alt klasör içerir: color / grayscale / segmented.
    Biz 'color' klasörünü kullanıyoruz (en iyi sonuçlar için).
    """
    # Olası yollar
    candidates = [
        extract_dir / "color",
        extract_dir / "plantvillage dataset" / "color",
        extract_dir / "PlantVillage" / "color",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Bulamazsak 'color' adında ilk dizini ara
    for root, dirs, _ in os.walk(extract_dir):
        for d in dirs:
            if d.lower() == "color":
                return pathlib.Path(root) / d

    # Hiçbiri bulunamazsa kök dizini döndür
    print("  ⚠️  'color' klasörü bulunamadı, kök dizin kullanılıyor.")
    return extract_dir


# ─────────────────────────────────────────────────────────────────────────────
#  ANALİZ FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def collect_dataset_info(image_root: pathlib.Path) -> pd.DataFrame:
    """
    Her sınıf klasöründeki görüntü sayısını, bitki türünü ve
    hastalık adını çıkararak bir DataFrame oluşturur.
    """
    banner("4️⃣  VERİ SETİ TARAMA")

    records = []
    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"  ❌  Hiç sınıf klasörü bulunamadı: {image_root}")
        sys.exit(1)

    print(f"  🔍  {len(class_dirs)} sınıf klasörü taranıyor...\n")

    for class_dir in tqdm(class_dirs, desc="  Sınıflar", unit="sınıf"):
        class_name = class_dir.name          # örn: "Tomato___Early_blight"
        images = list(class_dir.glob("*.jpg")) + \
                 list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.PNG")) + \
                 list(class_dir.glob("*.jpeg"))

        # Bitki ve hastalık adlarını ayır
        parts = class_name.replace("___", "|||").split("|||")
        plant   = parts[0].replace("_", " ") if len(parts) >= 1 else "Bilinmeyen"
        disease = parts[1].replace("_", " ") if len(parts) >= 2 else "Bilinmeyen"

        is_healthy = "healthy" in disease.lower()

        records.append({
            "class_name"  : class_name,
            "plant"       : plant,
            "disease"     : disease,
            "is_healthy"  : is_healthy,
            "image_count" : len(images),
            "class_path"  : str(class_dir),
        })

    df = pd.DataFrame(records)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Veri seti özetini terminale yazdırır."""
    banner("5️⃣  VERİ SETİ ÖZETİ")

    total_images   = df["image_count"].sum()
    total_classes  = len(df)
    total_plants   = df["plant"].nunique()
    healthy_count  = df[df["is_healthy"]]["image_count"].sum()
    diseased_count = total_images - healthy_count

    print(f"""
  📊  GENEL İSTATİSTİKLER
  ─────────────────────────────────────────────────
  Toplam Görüntü Sayısı   : {total_images:>10,}
  Toplam Sınıf Sayısı     : {total_classes:>10}
  Toplam Bitki Türü       : {total_plants:>10}
  Sağlıklı Görüntüler     : {healthy_count:>10,}  ({healthy_count/total_images*100:.1f}%)
  Hastalıklı Görüntüler   : {diseased_count:>10,}  ({diseased_count/total_images*100:.1f}%)
  ─────────────────────────────────────────────────
  Sınıf Başına Ort. Görüntü: {total_images/total_classes:>9.0f}
  En Az Görüntülü Sınıf  : {df['image_count'].min():>10,}
  En Çok Görüntülü Sınıf : {df['image_count'].max():>10,}
""")

    print("  🌿  BİTKİ TÜRLERİ VE SINIF DAĞILIMI")
    print("  " + "─" * 62)
    print(f"  {'Bitki Türü':<25} {'Sınıf Sayısı':>12} {'Toplam Görüntü':>14}")
    print("  " + "─" * 62)

    plant_summary = df.groupby("plant").agg(
        sinif_sayisi=("class_name", "count"),
        toplam_goruntu=("image_count", "sum")
    ).sort_values("toplam_goruntu", ascending=False)

    for plant, row in plant_summary.iterrows():
        print(f"  {plant:<25} {row['sinif_sayisi']:>12} {row['toplam_goruntu']:>14,}")

    print(f"\n  📋  TÜM SINIFLAR (sıralı):")
    print("  " + "─" * 62)
    print(f"  {'Sınıf Adı':<45} {'Görüntü':>8}")
    print("  " + "─" * 62)
    for _, row in df.sort_values("image_count", ascending=False).iterrows():
        icon = "✅" if row["is_healthy"] else "🔴"
        print(f"  {icon} {row['class_name']:<43} {row['image_count']:>8,}")


def sample_images_preview(df: pd.DataFrame, n_classes: int = 9) -> pathlib.Path:
    """Her sınıftan örnek görüntüler alarak bir önizleme grid'i oluşturur."""
    banner("6️⃣  ÖRNEK GÖRÜNTÜ ÖNİZLEMESİ")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Her sınıftan 1 örnek al (ilk n_classes sınıf)
    sample_df = df.head(n_classes)
    n         = len(sample_df)
    cols      = 3
    rows      = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.patch.set_facecolor("#1a1a2e")
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        class_path = pathlib.Path(row["class_path"])
        imgs = list(class_path.glob("*.jpg"))[:1] + \
               list(class_path.glob("*.JPG"))[:1] + \
               list(class_path.glob("*.png"))[:1]

        if not imgs:
            axes[i].set_visible(False)
            continue

        img = Image.open(imgs[0]).convert("RGB")
        axes[i].imshow(img)
        color = "#00ff88" if row["is_healthy"] else "#ff4466"
        axes[i].set_title(
            f"{row['plant']}\n{row['disease']}",
            color=color, fontsize=10, fontweight="bold", pad=8
        )
        axes[i].axis("off")
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PlantVillage – Örnek Görüntüler", color="white",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = REPORTS_DIR / "01_sample_images.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Kaydedildi: {out}")
    return out


def plot_class_distribution(df: pd.DataFrame) -> pathlib.Path:
    """Sınıf bazında görüntü sayısı çubuk grafiği."""
    banner("7️⃣  SINIF DAĞILIMI GRAFİĞİ")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("image_count", ascending=True)
    colors = ["#00cc66" if h else "#ff4455"
              for h in df_sorted["is_healthy"].tolist()]

    fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.35)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(
        df_sorted["class_name"],
        df_sorted["image_count"],
        color=colors,
        edgecolor="none",
        height=0.7,
    )

    # Çubuk değerlerini yaz
    for bar, count in zip(bars, df_sorted["image_count"]):
        ax.text(
            bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", ha="left",
            color="#c9d1d9", fontsize=7.5
        )

    ax.set_xlabel("Görüntü Sayısı", color="#c9d1d9", fontsize=12)
    ax.set_title("PlantVillage – Sınıf Başına Görüntü Dağılımı",
                 color="white", fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#30363d")

    legend = [
        mpatches.Patch(facecolor="#00cc66", label="Sağlıklı"),
        mpatches.Patch(facecolor="#ff4455", label="Hastalıklı"),
    ]
    ax.legend(handles=legend, loc="lower right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=10)

    plt.tight_layout()
    out = REPORTS_DIR / "02_class_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Kaydedildi: {out}")
    return out


def plot_plant_distribution(df: pd.DataFrame) -> pathlib.Path:
    """Bitki türü bazında toplam görüntü pie grafiği."""
    banner("8️⃣  BİTKİ TÜRÜ DAĞILIMI")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plant_totals = df.groupby("plant")["image_count"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")

    # Pie chart
    wedge_colors = sns.color_palette("husl", len(plant_totals))
    wedges, texts, autotexts = axes[0].pie(
        plant_totals.values,
        labels=plant_totals.index,
        autopct="%1.1f%%",
        colors=wedge_colors,
        pctdistance=0.82,
        startangle=140,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
    )
    for text in texts + autotexts:
        text.set_color("white")
        text.set_fontsize(8)
    axes[0].set_title("Bitki Türü Dağılımı (Görüntü %)",
                      color="white", fontsize=13, fontweight="bold")
    axes[0].set_facecolor("#0d1117")

    # Yatay çubuk grafik
    axes[1].set_facecolor("#161b22")
    bar_colors = sns.color_palette("husl", len(plant_totals))
    bars = axes[1].barh(
        plant_totals.index[::-1],
        plant_totals.values[::-1],
        color=bar_colors[::-1],
        edgecolor="none",
    )
    for bar, val in zip(bars, plant_totals.values[::-1]):
        axes[1].text(
            bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", color="#c9d1d9", fontsize=9
        )
    axes[1].set_title("Bitki Türü Başına Toplam Görüntü",
                      color="white", fontsize=13, fontweight="bold")
    axes[1].tick_params(colors="#c9d1d9", labelsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        axes[1].spines[sp].set_color("#30363d")

    plt.tight_layout()
    out = REPORTS_DIR / "03_plant_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Kaydedildi: {out}")
    return out


def analyze_image_properties(df: pd.DataFrame, sample_per_class: int = 5) -> None:
    """
    Her sınıftan birkaç görüntü alarak boyut ve kanal dağılımlarını analiz eder.
    """
    banner("9️⃣  GÖRÜNTÜ ÖZELLİKLERİ ANALİZİ")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    widths, heights, modes = [], [], []
    print(f"  🔬  Her sınıftan {sample_per_class} görüntü örnekleniyor...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Analiz", unit="sınıf"):
        class_path = pathlib.Path(row["class_path"])
        imgs = list(class_path.glob("*.jpg")) + \
               list(class_path.glob("*.JPG")) + \
               list(class_path.glob("*.png"))
        imgs = imgs[:sample_per_class]
        for img_path in imgs:
            try:
                with Image.open(img_path) as im:
                    widths.append(im.width)
                    heights.append(im.height)
                    modes.append(im.mode)
            except Exception:
                pass

    if not widths:
        print("  ⚠️  Görüntü analizi için örnek bulunamadı.")
        return

    print(f"""
  📐  GÖRÜNTÜ BOYUTU İSTATİSTİKLERİ
  ─────────────────────────────────────────
  Genişlik  → Ort: {np.mean(widths):.0f}px | Min: {min(widths)}px | Max: {max(widths)}px
  Yükseklik → Ort: {np.mean(heights):.0f}px | Min: {min(heights)}px | Max: {max(heights)}px
  Renk Modu → {collections.Counter(modes).most_common(3)}
""")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax, data, label, color in zip(
        axes,
        [widths, heights],
        ["Genişlik (px)", "Yükseklik (px)"],
        ["#00aaff", "#ff6600"],
    ):
        ax.set_facecolor("#161b22")
        ax.hist(data, bins=30, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(data), color="white", linestyle="--",
                   linewidth=1.5, label=f"Ort: {np.mean(data):.0f}px")
        ax.set_title(f"Görüntü {label} Dağılımı",
                     color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel(label, color="#c9d1d9")
        ax.set_ylabel("Frekans", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    plt.tight_layout()
    out = REPORTS_DIR / "04_image_size_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅  Kaydedildi: {out}")


def save_class_mapping(df: pd.DataFrame) -> pathlib.Path:
    """Sınıf→indeks eşleşme tablosunu CSV olarak kaydeder."""
    banner("🔟  SINIF-İNDEKS TABLOSU KAYDETME")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mapping_df = df[["class_name", "plant", "disease", "is_healthy", "image_count"]].copy()
    mapping_df = mapping_df.sort_values("class_name").reset_index(drop=True)
    mapping_df.index.name = "class_index"

    out_csv = DATA_DIR / "class_mapping.csv"
    mapping_df.to_csv(out_csv)
    print(f"  ✅  Sınıf tablosu kaydedildi: {out_csv}")

    out_json = DATA_DIR / "class_mapping.json"
    class_dict = {i: row["class_name"] for i, row in mapping_df.iterrows()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(class_dict, f, ensure_ascii=False, indent=2)
    print(f"  ✅  JSON eşleme kaydedildi: {out_json}")

    return out_csv


# ─────────────────────────────────────────────────────────────────────────────
#  ANA AKIŞ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    print("\n" + "=" * 70)
    print("  🌱  TARIM YAPAY ZEKASI – VERİ HAZIRLAMA AŞAMASI")
    print(f"  ⏰  Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Kaggle kimlik bilgileri
    if not check_kaggle_credentials():
        print("\n  ⛔  Script, kaggle.json olmadan devam edemiyor.")
        print("       Yukarıdaki talimatları uygulayın ve tekrar çalıştırın.\n")
        sys.exit(1)

    # 2. İndir
    zip_path = download_dataset()

    # 3. Çıkar
    extract_dir = extract_dataset(zip_path)

    # 4. Görüntü kökünü bul (color klasörü)
    image_root = find_image_root(extract_dir)
    print(f"\n  📂  Görüntü kök dizini: {image_root}")

    # 5. Veri seti bilgilerini topla
    df = collect_dataset_info(image_root)

    # 6. Terminale özet yazdır
    print_summary(df)

    # 7. Grafikler oluştur
    sample_images_preview(df)
    plot_class_distribution(df)
    plot_plant_distribution(df)
    analyze_image_properties(df)

    # 8. Sınıf eşleme tablosunu kaydet
    save_class_mapping(df)

    # 9. Bitiş
    elapsed = (datetime.now() - start_time).seconds
    banner("✅  VERİ HAZIRLAMA TAMAMLANDI")
    print(f"""
  ⏱️   Toplam süre    : {elapsed // 60} dk {elapsed % 60} sn
  📁  Ham veri       : {RAW_DIR}
  📊  Raporlar       : {REPORTS_DIR}
  📋  Sınıf tablosu  : {DATA_DIR / 'class_mapping.csv'}

  Sonraki adım: model_training.py  →  Transfer learning ile model eğitimi
""")


if __name__ == "__main__":
    main()
