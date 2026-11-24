# Literatür Taraması Raporu: CNN algı modelimizi hızlandırmak ve doğruluğunu artırmak

**Tarih**: 2026-11-24 (aynı gün, kullanıcı geri bildirimiyle CNN odağına göre yeniden yazıldı)
**Taranan kaynaklar**: OpenAlex API, 61 sorgu, 2016+; başlık, yıl, venue, atıf ve özet. Akademik MCP sunucuları bu oturumda bağlı değildi; genel web araması kullanılmadı. Önemli makalelerin özetleri tam okundu, tam metinler okunmadı. Özette olmayan sayılar iddia edilmiyor.
**Anahtar kelimeler**: efficient CNN backbone latency (FasterNet, RepViT, MobileNetV4, LowFormer, "Beyond MACs", GPUNet); depthwise convolution GPU memory-bound; FastViT / RepMixer; train-test resolution discrepancy; low-resolution distillation; channel-wise / boundary / relational distillation for dense prediction; multi-task distillation; side-tuning / frozen backbone; structural pruning (DepGraph); 2:4 sparsity; quantization of reparameterized nets (QARepVGG); depth distillation (Depth Anything V2, Distill Any Depth); video sparse inference (DeltaCNN, Skip-Convolutions).
**İncelenen makale sayısı**: ~55 ilgili (taranan ~1.200 kayıt).

## Bizim model ve ölçülmüş durum

**Çekirdek:** FastViT-SA24'ün ilk üç aşaması (stem + `stages_0..2`), toplam 10,7 M parametre.
- Bloklar RepMixer: derinlemesine 3×3 konvolüsyon + 1×1 konvolüsyonlu MLP. Dışa aktarmada tek konvolüsyona birleştiriliyor.
- Üstünde basit özellik piramidi ve yoğun başlıklar: derinlik, 11 sınıflı segmentasyon, kapı.
- Eklenti başlıkları kopyasız paylaşılan ara özellikleri (taps) okuyor.

**Orin NX (MAXN, TensorRT FP16) ölçümleri:**

| Ölçüm | Değer |
|---|---|
| Sistem motoru | 22,7 ms |
| Çekirdek | 15,3 ms; aşama 2 tek başına 6,45 ms |
| Başlıklar | ~7 ms |
| Uçtan uca | 32,7 ms (30,6 FPS) |
| Ağır başlıklar seyrek çalışınca motor | 19,3 ms |
| 384 px | 15,3 ms |
| 320 px | 11,8 ms |
| INT8 | %3 kazanç, doğruluk bozuldu |
| DLA | derleyici kurulu değil |

**Doğruluk zayıf noktaları:**
- Gerçek gece segmentasyonu.
- Derinlik: KITTI AbsRel 0,079'a karşı Depth Anything V2-Small 0,051; DIODE iç mekan 0,132'ye karşı 0,088.
- Düşük çözünürlükte isabet kaybı (başlıklar yeniden eğitilmeden): 384'te ~3, 320'de ~8 puan.

---

## 1. Özet

Literatürün bize en net söylediği şey şu: **FLOPs düşük ama GPU'da yavaş bir yapıdayız.** Derinlemesine konvolüsyon GPU'da bellek erişimine takılıyor; hesap gücü boşta kalıyor. FasterNet (CVPR 2023), LowFormer (WACV 2025), "Beyond MACs" (IJCV 2026) ve derinlemesine + 1×1 konvolüsyon birleştirme çalışması (2024) bunu bağımsız olarak gösteriyor. Bu, iki ölçümümüzü açıklıyor: INT8'in neden sadece %3 kazandırdığını ve çekirdeğin neden 15 ms sürdüğünü.

Bu yüzden en büyük hız kazancı nicemlemeden değil, iki yoldan gelir:
- **(a) Çözünürlük:** FixRes, MSAD ve Knowledge Adaptation, düşük çözünürlüğün kaybını başlık ince ayarı ve damıtmayla geri almanın yolunu veriyor.
- **(b) GPU-dostu, aynı ara özellikleri üreten yeni bir çekirdek:** Mevcut çekirdeğin özellik haritalarını taklit edecek şekilde damıtılır. Böylece bütün eklenti başlıkları değişmeden çalışmaya devam eder.

Doğruluk tarafında:
- **Derinlik:** Damıtmada derinlik normalizasyonu belirleyici (Distill Any Depth); öğretmen farkımızın bir kısmı buradan geliyor olabilir.
- **Segmentasyon:** Kanal bazlı ve kenar-ayrık damıtma (CWD, BPKD) ve güçlü öğretmenden ilişki-koruyan damıtma (DIST), sadece logit damıtmasından güçlü.
- **Mimari seçimimiz:** Dondurulmuş çekirdek + eklenti, literatürde side-tuning / LST çizgisiyle ve "ince ayar, önceden öğrenilmiş özellikleri bozup dağılım dışında kötüleştirebilir" bulgusuyla destekleniyor (Kumar ve ark. 2022). Bu makale için önemli bir dayanak.

---

## 2. Trend Analizi

### Yükselen alt-temalar
- **Gecikmeyi ölçerek tasarım (FLOPs yerine):** FasterNet 2023, GPUNet 2022, LANA 2022, LowFormer 2025, "Beyond MACs" 2026. Ortak sonuç: MACs/FLOPs gerçek süreyi tahmin etmiyor. Bellek erişimi ve paralellik belirleyici; derinlemesine konvolüsyon ve çok sayıda eleman bazlı işlem GPU'da pahalı.
- **Temel modelden (DINOv2, Depth Anything) küçük öğrenciye görev-özel damıtma:** EdgeCrafter (TMLR 2026): küçük modellerdeki farkın asıl nedeni yetersiz görev-özel temsil öğrenimi, mimarinin kendisi değil. "Preserving Angles Improves Feature Distillation" (TMLR 2025), Distill Any Depth (2025), Depth Anything V2 (NeurIPS 2024). Bizim çekirdek de DINOv2 + Depth Anything V2'den damıtıldı; bu çizginin içindeyiz.
- **Yoğun tahmin için yapısal damıtma:** CWD (ICCV 2021, 680+ atıf), BPKD (WACV 2024), DIST (NeurIPS 2022), çok görevli damıtma (ICCV 2023), JointDistill (ICME 2025).
- **Dondurulmuş omurga + görev adaptörleri:** ViT-Adapter, Mixture of Task Adapters (TPAMI 2026), E3VA (ACM MM 2024), LST (NeurIPS 2022), Side-Tuning (ECCV 2020).

### Durağan/düşüşte
- **FLOPs odaklı hafif ağlar** (MobileNet tarzı derinlemesine ağırlıklı tasarımlar): GPU'da yerini daha "yoğun" bloklara bırakıyor. FasterNet'in kısmi konvolüsyonu, RepViT, MobileNetV4'ün evrensel blokları.
- **Optik akışla özellik taşıma:** DFF, Accel. Hareketli kamerada hata birikiyor. Seyrek fark tabanlı çıkarım (DeltaCNN, MotionDeltaCNN, Skip-Convolutions) yeni ama TensorRT'de pratik değil.

---

## 3. Gap (Boşluk) Analizi

### Gap 1: Eklenti uyumlu çekirdek değişimi
- **Kanıt:** Damıtma literatürü öğrenciyi ya görev çıktısına ya da genel öğretmen özelliğine göre eğitiyor. Yeni ve daha hızlı bir çekirdeğin eski çekirdeğin ara özelliklerini taklit ederek mevcut eklenti başlıklarını yeniden eğitmeden kullanması üzerine bir çalışma bulmadım. Side-tuning ve adaptör çalışmaları omurgayı sabit tutuyor, değiştirmiyor.
- **Neden önemli:** Dondurulmuş çekirdek + eklenti mimarisinin en büyük sorusu "çekirdek eskiyince ne olacak?". Bunu cevaplamak hem hız kazancı verir hem makale için gerçek bir katkı olur.
- **Bizimle ilişkisi:** Bütün eklentilerimiz (kapı, köşe, seg v2, iç mekan, koşul modülleri) beş ara özellikten okuyor; uyumluluk ölçülebilir ve test edilebilir.

### Gap 2: Dondurulmuş çekirdekte çözünürlük değişimi
- **Kanıt:** FixRes, MSAD ve Knowledge Adaptation tüm ağı uyarlıyor. Sadece başlıkları uyarlayan bir çalışma bulmadım.
- **Bizimle ilişkisi:** 384'e geçiş için doğrudan uygulanabilir ve çekirdeğe dokunmuyor.

### Gap 3: Yeniden parametrize hibrit omurgada (RepMixer) INT8
- **Kanıt:** QARepVGG (AAAI 2024) ve RepQ (2023) RepVGG tipi ağlara bakıyor; RepMixer için reçete bulamadım. "Is INT8 Portable?" (2026) Orin'de INT8 çıktılarının seçilen çekirdeğe göre değiştiğini gösteriyor.
- **Bizimle ilişkisi:** Bellek darboğazlı olduğumuz için INT8'in getirisi zaten düşük. Akademik olarak açık bir soru, pratik önceliği düşük.

---

## 4. Yöntem Karşılaştırması ve Bize Uygulanabilirlik

| Yöntem | Ne yapıyor | Kanıt (kaynak) | Bize beklenen etki | Risk / maliyet |
|---|---|---|---|---|
| **Profil: derinlemesine / 1×1 / eleman bazlı ayrımı** | Aşama 2'nin 6,45 ms'inin hangi işlemde gittiğini görmek | FasterNet, LowFormer, Beyond MACs, DW+PW fusion (2024) | Sonraki her kararın temeli | 30 dk, risksiz |
| **Çözünürlük-uyumlu başlık ince ayarı + öz-damıtma** | Başlıkları 384'te, 448 çıktılarımızı öğretmen alarak eğitmek | FixRes (NeurIPS 2019); MSAD (CVPR 2021): düşük çözünürlüklü öğrenci +2,1–3,6 mAP; Knowledge Adaptation (CVPR 2019) | Motor 19,3 → ~15 ms; isabet kaybını ~3 puandan ≤1 puana çekmek (ölçülecek) | Çekirdek donmuş kalır; birkaç saatlik eğitim |
| **GPU-dostu, eklenti uyumlu yeni çekirdek (v5)** | Daha az derinlemesine / daha yoğun bloklarla küçük çekirdek; kayıp = v4'ün 5 ara özelliğini taklit + görev çıktıları | FasterNet (PConv), RepViT, MobileNetV4, GPUNet; CWD ve açı-koruyan özellik damıtması; DepGraph ile kanal budama alternatifi | Hedef çekirdek ~8 ms (tahmin; mimari aramayla doğrulanmalı). Bütün eklentiler yeniden eğitilmeden çalışmalı | Büyük iş (günler). Uyumluluk testleri tanımlanmalı. Gap 1'e karşılık geliyor |
| **Yapısal budama (DepGraph / Torch-Pruning)** | v4'ün aşama 2 kanallarını budayıp v4'e damıtarak toparlamak | DepGraph (CVPR 2023); "Rethinking the Value of Pruning" (2018): budanmış mimariyi baştan eğitmek çoğu zaman aynı veya daha iyi; PruningBench (2024) | Mimari değişmediği için eklentiler daha kolay uyumlu kalır | Budanan kanal sayısı kadar kazanç; GPU'da gerçek hız ölçülmeli |
| **2:4 yapısal seyreklik** | Seyrek tensor çekirdekleri | NVIDIA 2021: yoğun matris işlemlerinde 2×, ince ayarla doğruluk korunuyor | Sadece 1×1 / yoğun konvolüsyonlarda. Derinlemesine ağırlıklı yapıda kazanç sınırlı olabilir | Önce `trtexec --sparsity=force` ile üst sınır (10 dk) |
| **INT8 (QAT / niceleme dostu yeniden parametrize)** | 8-bit | QARepVGG (>%20 → <%2 kayıp), White Paper (2021), QuartDepth (CVPR 2025) | Bellek darboğazında getiri düşük (bizde %3). v5 çekirdek yoğun bloklarla gelirse yeniden değerlendirilir | Yüksek emek; ertele |
| **Derinlik damıtmasında normalizasyon** | Pseudo-etiket damıtmasında küresel yerine yerel/hibrit normalizasyon | Distill Any Depth (2025); Boosting Lightweight Depth via KD (2023); Depth Anything V2 | Derinlik farkının (KITTI 0,079'a karşı 0,051) bir kısmını kapatmak | Derinlik eklenti modülüyle denenebilir; çekirdek donmuş kalır |
| **Segmentasyon damıtması: CWD + kenar + ilişki** | Kanal dağılımı eşleme, kenar/gövde ayrı damıtma, güçlü öğretmende ilişki koruma | CWD (ICCV 2021), BPKD (WACV 2024), DIST (NeurIPS 2022), MTL-KD (ICCV 2023) | Seg eklentilerinde (gece, dış mekan) daha keskin sınır ve daha az parçalı çıktı | Eklenti eğitim betiklerine kayıp eklemek |
| **Seyrek kare farkı çıkarımı** | Değişmeyen bölgeleri atlamak | DeltaCNN (CVPR 2022), MotionDeltaCNN (ICCV 2023, hareketli kamera), Skip-Conv (CVPR 2021) | Hareketli kamerada kazanç düşük; TensorRT'de özel çekirdek gerekir | Düşük öncelik |
| **Erken çıkışlı yoğun tahmin** | Kolay karelerde erken çıkış | Multi-Exit Semantic Segmentation (ECCV 2022), Anytime Dense Prediction (2021) | Süre değişken olur; gerçek zaman sisteminde istenmez | Düşük öncelik |

**Çelişkili bulgu:** INT8 genelde "varsayılan kenar reçetesi" olarak sunuluyor. Ama bellek darboğazlı ve yeniden parametrize ağlarda hem kazanç düşük hem doğruluk riski yüksek. Bizim ölçümümüz bu tarafta.

**Konumlandırmamız (makale için):**
- Dondurulmuş çekirdek + eklenti, side-tuning / adaptör literatürüyle ve "ince ayar özellikleri bozar" bulgusuyla destekleniyor.
- Literatürden ayrıldığımız yer: adaptörlerin çoğu büyük ViT'ler ve sunucu GPU'su için. Biz 10,7 M parametrelik CNN çekirdekle Orin'de ölçülmüş gecikme veriyoruz.
- Gap 1 (eklenti uyumlu çekirdek yükseltmesi) yapılırsa, bu mimarinin bilinen tek zayıf noktasını ("çekirdek eskiyince") kapatan bir katkı olur.

---

## 5. Öneri: sıra ve beklenen kazanç

| # | Adım | Beklenen | Doğruluk riski |
|---|---|---|---|
| 0 | Ağır başlıkları seyrek çalıştırma (Orin'de test ediliyor) | motor 19,3 ms | yok |
| 1 | Aşama 2 profili: derinlemesine / 1×1 / eleman bazlı süre dağılımı | karar verisi | yok |
| 2 | Refiner + son işlem ek yükü (10 ms → ~4 ms) | ~45–50 FPS | yok |
| 3 | 384 + başlıkların çözünürlük-uyumlu ince ayarı + öz-damıtma | ~55–60 FPS | hedef ≤1 puan |
| 4 | `trtexec --sparsity=force` üst sınır testi | karar verisi | yok |
| 5 | Eklenti uyumlu, GPU-dostu çekirdek v5 (yoğun bloklar ve/veya DepGraph budaması; v4 özelliklerine damıtma; 2:4 opsiyonel) | çekirdek ~2× (tahmin) | uyumluluk testleriyle kontrol |
| 6 | Derinlik eklentisinde normalizasyon-farkında damıtma; seg eklentilerinde CWD + BPKD | derinlik / seg doğruluğu | yok (eklenti) |
| 7 | INT8 QAT (v5 sonrası yeniden değerlendirme) | ? | yüksek |

---

## Ek not: drone yarışı literatürü (yan bilgi)

Kullanım senaryosu (20 m/s drone) için kısa not: A2RL 2025 şampiyonu MonoRace (Sci. Robotics 2026) ve diğer A2RL sistemleri (JFR 2026, arXiv 2026) kapı tespitini IMU / drone modeliyle çalışan filtreye ölçüm olarak veriyor; kontrol algıdan çok daha hızlı çalışıyor (MonoRace: 500 Hz). Yani araç tarafında gecikme telafisi ve IMU tahmini, algı FPS'i kadar önemli. Bu raporun konusu değil.

---

## 6. Kaynakça (DOI)

**GPU'da verimli CNN tasarımı**
- Chen ve ark., Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks (FasterNet), CVPR 2023 — 10.1109/cvpr52729.2023.01157
- LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones, WACV 2025 — 10.1109/wacv61041.2025.00681
- Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones, IJCV 2026 — 10.1007/s11263-026-02873-5
- Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs, 2024 — 10.1145/3677333.3678153
- Searching the Deployable Convolution Neural Networks for GPUs (GPUNet), CVPR 2022 — 10.1109/cvpr52688.2022.01191
- RepViT: Revisiting Mobile CNN From ViT Perspective, CVPR 2024 — 10.1109/cvpr52733.2024.01506
- MobileNetV4, ECCV 2024 — 10.1007/978-3-031-73661-2_5
- FastViT, ICCV 2023 — 10.1109/iccv51070.2023.00532
- LANA: Latency Aware Network Acceleration, ECCV 2022 — 10.1007/978-3-031-19775-8_9

**Çözünürlük**
- Touvron ve ark., Fixing the train-test resolution discrepancy, NeurIPS 2019 — 10.48550/arxiv.1906.06423
- Multi-Scale Aligned Distillation for Low-Resolution Detection, CVPR 2021 — 10.1109/cvpr46437.2021.01421
- Knowledge Adaptation for Efficient Semantic Segmentation, CVPR 2019 — 10.1109/cvpr.2019.00067

**Damıtma (yoğun tahmin, temel modeller)**
- Channel-wise Knowledge Distillation for Dense Prediction, ICCV 2021 — 10.1109/iccv48922.2021.00526
- BPKD: Boundary Privileged Knowledge Distillation, WACV 2024 — 10.1109/wacv57701.2024.00110
- Knowledge Distillation from a Stronger Teacher (DIST), NeurIPS 2022 — 10.52202/068431-2443
- Multi-Task Learning with Knowledge Distillation for Dense Prediction, ICCV 2023 — 10.1109/iccv51070.2023.01970
- Structured Knowledge Distillation for Dense Prediction, TPAMI 2020
- EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation, TMLR 2026 — 10.48550/arxiv.2603.18739
- Depth Anything V2, NeurIPS 2024
- Distill Any Depth, arXiv 2025 — 10.48550/arxiv.2502.19204
- DINOv2, TMLR 2024

**Dondurulmuş omurga / adaptörler**
- Side-Tuning, ECCV 2020 — 10.1007/978-3-030-58580-8_41
- LST: Ladder Side-Tuning, NeurIPS 2022 — 10.52202/068431-0944
- Kumar ve ark., Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution, 2022 — 10.48550/arxiv.2202.10054
- Parameter-efficient is not Sufficient (E3VA), ACM MM 2024 — 10.1145/3664647.3680940

**Budama, seyreklik, nicemleme**
- DepGraph: Towards Any Structural Pruning, CVPR 2023 — 10.1109/cvpr52729.2023.01544
- Rethinking the Value of Network Pruning, 2018 — 10.48550/arxiv.1810.05270
- A Comprehensive Study of Structural Pruning for Vision Models (PruningBench), 2024 — 10.48550/arxiv.2406.12315
- Accelerating Sparse Deep Neural Networks (2:4), 2021 — 10.48550/arxiv.2104.08378
- Make RepVGG Greater Again (QARepVGG), AAAI 2024 — 10.1609/aaai.v38i10.29045
- A White Paper on Neural Network Quantization, 2021 — 10.48550/arxiv.2106.08295
- QuartDepth, CVPR 2025 — 10.1109/cvpr52734.2025.01069
- Is INT8 Portable?, arXiv 2026 — 10.48550/arxiv.2609.16085

**Video / seyrek çıkarım**
- DeltaCNN, CVPR 2022 — 10.1109/cvpr52688.2022.01217
- MotionDeltaCNN, ICCV 2023 — 10.1109/iccv51070.2023.01586
- Skip-Convolutions, CVPR 2021 — 10.1109/cvpr46437.2021.00272
- Accel, CVPR 2019 — 10.1109/cvpr.2019.00907
- Multi-Exit Semantic Segmentation Networks, ECCV 2022 — 10.1007/978-3-031-19803-8_20

**Yan not (drone yarışı)**
- MonoRace, Science Robotics 2026 — 10.1126/scirobotics.aee9250
- Drift-Corrected Monocular VIO, J. Field Robotics 2026 — 10.1002/rob.70371
- Champion-level drone racing (Swift), Nature 2023 — 10.1038/s41586-023-06419-4
