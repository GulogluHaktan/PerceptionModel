# PerceptionModel — projenin tamamı

*Son güncelleme: 2026-11-24. Ayrıntılı deney günlüğü: `docs/handoff/perceptionmodel-sim-plan.md` (Update 1–164). Bu dosya o günlüğün özeti ve rehberidir; sayılar oradaki ölçümlerden alınmıştır.*

---

## 1. Proje ne?

Küçük bir araçta (drone, araba, robot) gerçek zamanlı çalışan **genel amaçlı bir görsel algı modeli**. Tek bir kameradan her karede şunları çıkarıyor:

- **Derinlik:** her piksel için göreli ters derinlik.
- **Segmentasyon:** 11 sınıf; iç mekanda 22 sınıf, dış mekanda 19 sınıf ek modüllerle.
- **Kapı tespiti:** drone yarış kapılarının 4 köşesi, kısmen görünen kapılar için tek köşeler. Projenin ilk somut görevi buydu.

Hedef donanım **NVIDIA Jetson Orin NX 8 GB**, kamera **IMX219**.

Projenin asıl fikri şu: **"Dondurulmuş genel çekirdek + takılıp çıkarılabilen modüller + kendi kendini geliştiren döngü."**
- Çekirdek bir kez eğitilip donduruluyor; böylece eski yetenekler hiç bozulmuyor.
- Yeni bir görev veya zor bir koşul (gece, sis, NIR kamera, iç mekan nesneleri) gerektiğinde çekirdeğe dokunulmuyor. Onun ara özelliklerini okuyan küçük bir modül eğitilip takılıyor.
- Uzun vadeli hedef: sistemin zor koşulu kendisi fark etmesi, simülasyon / 3DGS / LLM ile o koşul için veri üretmesi, modül eğitmesi, doğrulaması ve takması.

Kullanıcının önceliği (2026-11-21): **önce araçta çalışan sistem, makale sonra.**

---

## 2. Donanım ve ortam

| Ne | Ayrıntı |
|---|---|
| Eğitim makinesi | RTX 5090 (bu bilgisayar). Python ortamı `.venv`; testler `env -u PYTHONPATH .venv/bin/python -m pytest tests` |
| Araç bilgisayarı | Jetson Orin NX 8 GB geliştirme kiti (p3768 taşıyıcı), JetPack R39.2.1, TensorRT 10.16. Atölye ağında `10.9.25.123`, kullanıcı `lovelace`; çalışma zamanı `haktan` kullanıcısı altında `/home/haktan/...` |
| Kamera | IMX219 CSI (bugün geldi, henüz takılmadı). Hazır sürücü modları: 1640×1232 @30, 1280×720 @60 |
| Drone parçaları | 5" MANTA 5 DeadCat gövde, MicoAir743 V2 uçuş kontrolcüsü, TF02-Pro lidar, 6S |
| Geçici kamera | Telefon IP kamera; laptop hotspot + ters SSH tüneliyle Orin'e (`http://127.0.0.1:8080/video`) |

Orin'de DLA hızlandırıcı şu an kullanılamıyor: derleyici kütüphanesi `libnvdla_compiler` kurulu değil.

---

## 3. Mimari

```
Kamera karesi (1640×1232 veya 1280×720)
  │  GPU ön işlem: lens distorsiyonu düzeltme, kare kırpma, 448×448'e küçültme
  ▼
┌─────────────── ÇEKİRDEK — release_v4 "bigY_h", DONDURULMUŞ, 10,7 M parametre ───────────────┐
│ Omurga: FastViT-SA24'ün stem + stages_0..2 kısmı (RepMixer blokları = konvolüsyonel;         │
│         FastViT'nin dikkat içeren 4. aşaması YOK). Dışa aktarmada çok dallı bloklar           │
│         tek konvolüsyona birleştirilir (re-parametrizasyon).                                  │
│ Basit özellik piramidi (3 ölçek) → yoğun başlıklar:                                           │
│   • derinlik (ters derinlik)  • 11 sınıflı segmentasyon  • kapı tespiti (yoğun ızgara,       │
│     adım 4, 2 boyut seviyesi, kare başına en çok 8 kapı)                                      │
│ Eğitimde ayrıca öğretmen başlıkları (DINOv2 özelliği) — araçta kullanılmaz                    │
└────────────────────────────────────────────────────────────────────────────────────────────┘
  │ ara özellikler ("taps": s4, s8, feat, seg_core, s16) GPU belleğinde kopyasız paylaşılır
  ├──► Kapı eklentileri (her karede)
  │      • NearGateHead ng2: yakın / büyük kapılar
  │      • CornerPeakHead cpk3: tek köşe tepeleri; zayıf kapı tespitlerini "onaylar"
  │      • CornerRefiner v2: her köşe çevresinden 64×64 yama, küçük U-Net, soft-argmax ile alt-piksel köşe
  │      • eleme: eşik 0,05, tekilleştirme IoU 0,2, en küçük kapı 14 px ("min14" yapılandırması)
  │
  ├──► Sahne yönlendiricisi: çekirdeğin özet vektörü üzerinde lojistik regresyon (iç / dış, test %97,8)
  │      ├─ iç mekan → İç mekan modülü (22 sınıf, Swin-T Mask2Former, her 10 karede)
  │      └─ dış mekan → Dış mekan modülü (19 sınıf, Swin-T Mask2Former, her 10 karede)
  │                     + seg v1 + v2 (11 sınıfın iyileştirilmiş hali)
  │
  └──► Koşul modülleri (deneysel; araç paketinde değil): gece, NIR, sis, bulanıklık + koşul yönlendiricisi
  ▼
Çıktı: kapılar (4 köşe + skor), tek köşeler (EKF için), segmentasyon haritaları, derinlik
```

### 3.1 Çekirdek nasıl eğitildi (release_v4 = bigY_h)

Çok görevli öğretmen-öğrenci damıtması; tek ağ aynı anda şunları öğrendi:

| Kayıp | Kaynak |
|---|---|
| Özellik damıtma | DINOv2 ViT-S/14 (öğretmen özellikleri önbellekte) |
| Derinlik | Depth Anything V2 öğretmeni + gerçek derinlik etiketi olan yerlerde (DIODE vb.) ölçek-kayma değişmez L1, ters derinlik uzayı |
| Segmentasyon | 11 sınıf: background, gate, tree_pole, person, ground_road, building, opponent_drone, car, rock, lane_marking, other_obstacle. Sınıf ağırlıkları car 2, lane_marking 4, opponent_drone 3, person 4 |
| Kapı köşeleri | Yoğun kapı başlığı (köşe ağırlığı 5), yarım kapı artırımı (`half_gate_prob 0.4`), yakın geçiş kareleri ×6 |

Ayarlar: 448 px, batch 32, lr 2e-4, 16 epoch, seed 2. Artırım: ışık, sis, bulanıklık, yağmur, JPEG ve bunların birleşimi.

**Eğitim verisi (`data/mix/bigG/images`, ~80 bin görüntü):**

| Kaynak | Adet | Ne için |
|---|---|---|
| Kendi simülasyon setlerimiz (sim_race*, sim_outdoor, sim_indoor) | ~38.000 | kapılar, mükemmel etiket, zor bakış açıları |
| DroneRacing Isaac kareleri (dr1) | 12.000 | rakibin simülatöründen kapılar |
| BDD100K | 7.000 | yol / şehir segmentasyonu |
| SUN RGB-D | 3.964 | iç mekan |
| ADE20K | 3.000 | genel sahne |
| TII gerçek kapı uçuşları | 3.000 + 608 negatif | gerçek kapılar |
| DIODE | 2.700 | gerçek derinlik |
| Aeroscapes | 2.621 | havadan görüntü, drone sınıfı |
| TuSimple (+7 artırılmış kopya) | ~2.100 | şerit çizgisi |
| UAVid | 1.000 | havadan |
| RUGD | 634 | arazi |
| MonoRace "red" gerçek kapılar | 191 + 90 negatif | **hiç görülmemiş mekan testi için ayrıldı** (eğitimde sadece küçük bir kısmı) |

**Lisans notu:** BDD100K, RUGD ve IDD-AW ticari olmayan kullanım; UAVid ve KITTI CC BY-NC-SA.

**Sürüm geçmişi (kısaca):**
- v1 = bigL_h.
- v2 = bigN_h2: kapıyı merkezden tamamlama, K=8.
- v3 = bigR_h_s1: 2 boyut seviyesi, yakın geçiş ×6.
- v4 = bigY_h: v3 tarifi, 16 epoch. **O günden beri dondurulmuş**; sonraki her şey eklenti.

### 3.2 Eklentiler (çekirdeğe dokunmadan eklenenler)

| Eklenti | Mimari | Neden eklendi | Sonuç |
|---|---|---|---|
| CornerRefiner v2 | 64×64 yama U-Net, ~0,3–0,6 M parametre, soft-argmax | Köşe hassasiyeti (kapalı döngüde GateNet'in gerisindeydik) | Döngüde köşe medyanı 3,5 → 1,7 px; gerçek "red" setinde 2,57 → 0,68 px |
| CornerPeakHead cpk3 | Çekirdek özelliklerinden köşe tepe haritası | Zayıf kapıları onaylamak, yarım kapıda tek köşe vermek | Kapalı döngü başarısını belirgin artırdı |
| NearGateHead ng2 | Yakın / büyük kapı başlığı | Çok yakın kapıları kaçırıyorduk | Kayıp yakın kapılar azaldı |
| Seg v1 (SegRefine) + v2 (SegHead2) | Çekirdek özellikleri + görüntü üstünde hafif kod çözücüler; v2 sentetik gece denetimiyle | 11 sınıflı segmentasyonun kalitesi, gece | İkisinin ortalaması araç paketine alındı |
| Sahne yönlendiricisi | Özet vektörü üstünde lojistik regresyon | İç / dış modül seçimi | Test %97,8, 3 karelik histerezis |
| **İç mekan modülü** | Swin-T Mask2Former; herkese açık COCO ağırlığından başlatılıp bizim 22 sınıfa ince ayar + Swin-L'den damıtma | Mouse, klavye, monitör, laptop, masa... | Gerçek COCO iç mekan testi **0,695** (TRT FP16) |
| **Dış mekan modülü** | Swin-T Mask2Former; Cityscapes ağırlığından başlatılıp BDD'nin 19 sınıfına ince ayar | Yol, kaldırım, trafik ışığı, bisiklet, sürücü... | BDD **0,613**, gerçek gece **0,471** (TRT FP16) |
| Koşul modülleri (deneysel) | Gece, NIR, sis, bulanıklık için ayrı modüller + 6 sınıflı koşul yönlendiricisi | Kendi kendini geliştiren döngünün ilk adımı | Simülasyonda kabul edildi; **gerçek karanlık karelerde zarar verdi** (sentetik gece ≠ gerçek gece). Gerçekçi gece üreticisi yazıldı; araç paketinde değil |

İç ve dış mekan modüllerinin hikayesi:
- Önce donmuş çekirdek üstüne kendi küçük başlıklarımızı eğittik (ARCH1–7, damıtmalı). İç mekanda en iyi 0,64'e ulaştı ama görüntüde parçalı ve dağınıktı.
- Literatür taraması sonrası nesne-temelli (mask classification) mimariye, yani Mask2Former'a geçtik.
- Büyük modeller (Swin-L) öğretmen, Orin'de çalışan Swin-T öğrenciler bizim eğittiğimiz modeller.

---

## 4. Araç üstünde çalışma (deploy/)

**Paket:** `deploy/` klasörü. Orin'de `/home/haktan/pfm_m2f` (30 FPS paketi) ve `/home/haktan/pfm_drone448` (drone modu).

| Dosya | Görev |
|---|---|
| `export_system.py` | PyTorch → ONNX. Sistem grafiği (çekirdek + kapı eklentileri), refiner, ayrı başlık grafikleri. `PFM_SPLIT=1` ayrık paket, `PFM_DRONE=1` drone modu, `PFM_S` giriş boyutu, `PFM_INDOOR` iç mekan başlığı. ONNX ile PyTorch'u 24 test karesinde karşılaştırır |
| `export_m2f_indoor.py` | Mask2Former modüllerini ONNX'e çevirir. Deformable attention için TensorRT eklentisi (`MultiscaleDeformableAttnPlugin_TRT`); `CLASSSET=out` dış mekan |
| `build_engines.py` | Orin'de TensorRT FP16 motorlarını derler (motorlar cihaza özel). Var olan başlık motorlarını tekrar derlemez |
| `pfm_runtime.py` | Çalışma zamanı: iki tamponlu boru hattı, CUDA graph, başlıkların ara özellikleri kopyasız okuması, sahne yönlendiricisi, başlıkları her N karede çalıştırma, köşe refiner'ı, eleme |
| `gdet_post.py` | Kapı adaylarının seçimi ve tekilleştirmesi (GPU'da aday, CPU'da hızlı seçim) |
| `camera.py`, `run.py` | Kamera (CSI, URL, video dosyası), GPU distorsiyon düzeltme, kayıt (`--record`) |
| `check_device.py` | Orin'de TensorRT ile PyTorch referansının karşılaştırması (kapı seti, segmentasyon uyumu) ve gecikme dökümü |
| `rec_sheet.py` | Kayıtlardan sonuç sayfası (kamera, 11 sınıf, modül, derinlik) |

**Orin ölçümleri (MAXN, TensorRT FP16):**

| Yapılandırma | Süre / FPS | Not |
|---|---|---|
| Tüm başlıklar her karede (pfm_m2f) | 32,7 ms, **30,6 FPS** (iç ve dış) | Sistem motoru 22,7 ms (çekirdek 15,3 ms), refiner 4,8 ms |
| Drone modu (seg başlıkları her 3, Mask2Former her 10 karede) | 30,2 ms, **33,1 FPS** | Kapı seti 120/120 aynı. Beklenenden az kazanç: darboğaz refiner + son işlem |
| Sistem motoru, drone modu, 384 px | 15,3 ms (yalnız motor) | Başlıklar yeniden eğitilmeden kapı isabeti ~3 puan düşük |
| Sistem motoru, drone modu, 320 px | 11,8 ms (yalnız motor) | ~8 puan düşük |
| Mask2Former modülü (tek geçiş) | 47,4 ms | Her 10 karede çalıştığı için FPS'i az etkiliyor |
| INT8 | %3 kazanç, doğruluk bozuldu | Reddedildi |

**Hedef:** drone için 60 FPS (kullanıcı, 2026-11-24). Plan aşağıda (bölüm 9).

---

## 5. Değerlendirme: ne ölçüyoruz, nasıl ölçüyoruz

**İlkeler (`docs/PROTOCOL.md`):**
- Sabit test setleri ("golden").
- Eşit hesaplı karşılaştırma: aynı görüntü sayısı, batch, çözünürlük, epoch.
- Ayar yapılan tohumlar ile test tohumları ayrı.
- Gürültülü ölçümler tekrarlanır.
- Gecikme: medyan ve p95, sabit güç modu, ön ve son işlem ayrı.

| Test | Ne ölçer | release_v4 sonucu |
|---|---|---|
| **MonoRace "red"** (196 kare, 312 kapı, hiç görülmemiş gerçek mekan) | Köşeler 20 / 10 px içinde | **0,987 / 0,958**; iç içe kapılar 20/20 |
| TII uçuşları (görülmüş salon) | aynı | 0,960 |
| Yarım kapı (tek köşe görünür / sadece çubuk) | bulma oranı | 0,895 / 0,505 (zayıf) |
| TuSimple şerit | IoU | 0,803 |
| SUN iç mekan | mIoU (11 sınıf) | 0,761 |
| Aeroscapes rakip drone | IoU | 0,675 |
| IDD-AW (kötü hava) | mIoU | 0,602 |
| KITTI / DIODE derinlik | AbsRel | 0,079 / 0,132 iç, 0,260 dış |
| Kapalı döngü simülasyon (kendi) | geçilen kapı | 175/176 (test doymuş) |

---

## 6. Rakiplerle ve literatürle karşılaştırmalar

### 6.1 GateNet (DroneRacing projesinin dedektörü, en doğrudan rakip)

- **Gerçek kareler, çevrimdışı ("red", aynı eşleştirme ve aynı ölçek):** 20 px içinde **bizim 0,987, GateNet 0,776.** GateNet sadece simülasyonla eğitilmiş ve normalde kullandığı kırpma/EKF yardımı bu testte yok; bu bir dedektör karşılaştırması, uçuş başarısı değil.
- **Kapalı döngü simülasyon:** Onların simülatörü, politikası ve EKF'si; sadece dedektör değişti, onların ayarlarına dokunulmadı.
  - Temiz tohumlar 141–176, her model 128 uçuş: **bizim %87,5, GateNet %82,0.**
  - Tohum bazında 12 kazanç, 5 kayıp, 15 beraberlik; işaret testi p ≈ 0,14. Önde ama istatistiksel olarak kesin değil; döngü gürültüsü ±10–15 puan.
  - Uzak kapılarda (8 m+) GateNet daha hassas.
- **Yol:**
  - Başta çok gerideydik (bigY_h tek başına ~%47, GateNet %78).
  - Refiner, köşe onayı, yakın kapı başlığı, tekilleştirme ve 14 px eşiğiyle, çekirdeğe dokunmadan öne geçtik.
  - İki temiz testte de öndeyiz.
- **Dar pist (UZH Swift kopyası):** İki dedektör de ~%0–6. Sebep algı değil; politika bu pist için eğitilmemiş.

### 6.2 Hazır büyük modeller
- **İç mekan:** Bizim öğrenci COCO testinde 0,695, aynı modelin eğitimsiz hali 0,687. Ofis karelerinde Swin-L öğretmeniyle uyumda 0,448'e karşı 0,420. Fark küçük; asıl kayıp 448 kare sıkıştırmada. Oranı koruyunca 0,502.
- **Dış mekan:** Bizim öğrenci 448 px'te BDD 0,613; Swin-L öğretmen 1024 px'te 0,603. Gerçek gecede 0,471'e karşı 0,394.

### 6.3 Derinlik, Depth Anything V2-Small'a karşı

| Veri seti | Bizim (AbsRel) | DA-V2-Small | Daha iyi olan |
|---|---|---|---|
| KITTI | 0,079 | 0,051 | onlar |
| DIODE iç mekan | 0,132 | 0,088 | onlar |
| DIODE dış mekan | 0,260 | 0,452 | biz |

Derinlikte genel olarak gerideyiz.

### 6.4 Literatür taramaları (`docs/literatur/`)
- `dondurulmus-moduler-algi-literatur-taramasi.md` (2026-11-21): dondurulmuş çekirdek + modül yaklaşımının literatürdeki yeri, boşluklar, hedef dergiler (RA-L; açık erişim: Drones, IEEE Access).
- `kenar-segmentasyon-damitma-neyi-yanlis-yapiyoruz.md` (2026-11-23): parçalı segmentasyonun nedenleri. Sadece logit damıtması zayıf; kenar ve ilişki damıtması, nesne-temelli mimariler. Bu tarama Mask2Former'a geçişi getirdi.
- `hizlandirma-dogruluk-literatur-taramasi.md` (2026-11-24): CNN modelimizi hızlandırma.
  - Derinlemesine konvolüsyonlar GPU'da bellek darboğazı; INT8'in düşük kazancını açıklıyor.
  - Yol: çözünürlük-uyumlu ince ayar + damıtma, ve eklenti uyumlu GPU-dostu yeni çekirdek.

### 6.5 Makale durumu
PaperOrchestra aracıyla otomatik taslak yazıldı (`docs/paper_orchestra/`); otomatik hakem 3/10 verdi. Kullanıcı bunu sadece aracın denemesi olarak gördü, projeyi ona göre yönlendirmiyoruz. Gerçek eksikler:
- standart benchmark'larda yayınlanmış yöntemlerle karşılaştırma,
- modül başına Orin maliyeti (artık ölçülüyor),
- gerçek kamera ve gerçek uçuş testi.

---

## 7. Veri fabrikası ve kendi kendini geliştirme

**Simülasyon ve sentetik veri (`sim/`, `src/pfm/`):**
- **Fikir:** 3DGS gerçek mekanı (arka plan RGB + derinlik) verir. Isaac Sim eklenen nesneleri, ışığı ve mükemmel etiketleri verir. `effects.py` kamera ve hava efektlerini sona ekler (bulanıklık, sis, yağmur, gece, `night_real`).
- **LLM'in rolü:** Sadece katı bir senaryo JSON'u doldurur (`scenario.py`, `layout_llm.py`); asla piksel üretmez.
- **Kamera sözleşmesi doğrulandı:** Isaac ile analitik renderer arasında maske IoU 1,000.
- **Yapıldı:** iç mekan / dış mekan / yarış yerleşimleri (LLM ile id ve koordinat seçimi), Isaac yarış döngüsü, DroneRacing simülatörüyle kapalı döngü.
- **Eksik:** gerçek metrik ölçek kalibrasyonu, gölgeler, 3DGS + Isaac'in tam birleşik üretimi. İç mekan Isaac render'ı takıldı, tekrar denenmeli.

**Kendi kendini geliştiren döngü (`docs/SELF_IMPROVE_PLAN.md`):**
1. **Kayma tespiti** — yapıldı. Özellik uzaklığı AUROC 0,91–1,0; kapı güveni kare hatasını öngörüyor (Spearman −0,69).
2. **Koşula özel modül eğitimi + koşul yönlendiricisi** — simülasyonda kabul edildi, gerçek karanlıkta reddedildi. Ders: sentetik koşul gerçeğe benzemeli, ve gerçek veriyle ayrı kabul seti gerekiyor.
3. **Güvenli evrim kapısı + geri alma** (`gate.py`, `selfloop.py`) — kod var, sahte veriyle test edildi.
4. **Hedef:** araçta zor koşulu tespit et → veri üret → modül eğit → doğrula → tak. Henüz uçtan uca gerçek veriyle çalışmadı.

---

## 8. Kod haritası

| Yer | İçerik |
|---|---|
| `src/pfm/model.py` | `PerceptionModel`: omurga (ViT-B veya hibrit FastViT-SA24), öğretmen başlıkları, piramit, yoğun başlıklar, kapı kod çözücü |
| `src/pfm/train.py`, `distill.py`, `teachers.py`, `cache.py`, `data.py` | Eğitim, damıtma kayıpları, öğretmenler (DINOv2, Depth Anything V2), öğretmen özellik önbelleği, veri yükleyici |
| `src/pfm/cornerpeak.py`, `refine.py`, `gate_seg.py` | Köşe tepe, yakın kapı başlıkları, köşe refiner'ı |
| `src/pfm/indoor_classes.py`, `outdoor_classes.py`, `classes.py` | Sınıf tanımları ve eşlemeler (COCO → 22, Cityscapes 19) |
| `src/pfm/conditions.py`, `effects.py` | Koşul modülleri ve görüntü efektleri |
| `src/pfm/scenario.py`, `layout*.py`, `llm_scenario.py`, `compose.py`, `camera.py`, `trajectory.py`, `dataset_writer.py` | Veri fabrikası |
| `src/pfm/gate.py`, `selfloop.py`, `evaluate.py`, `bench.py`, `energy.py` | Kabul kapısı, kendi kendini geliştirme döngüsü, değerlendirme, gecikme ve enerji |
| `scripts/` | Eğitim zincirleri (`bige_chain.py`), modül eğitimleri (`train_*`), değerlendirmeler (`eval_*`), veri hazırlama (`prep_*`), teşhis (`diag_*`), görseller (`vis_*`) |
| `deploy/` | Araç paketi (bölüm 4) |
| `runs/release_v1..v4/` | Dondurulmuş sürümler (ağırlık, ONNX, README, SHA256) |
| `runs/modules/` | Eklenti modüllerin ağırlıkları ve günlükleri |
| `docs/handoff/perceptionmodel-sim-plan.md` | Tüm deney günlüğü |
| `docs/report/` | Türkçe raporlar v1–v5 (PDF) ve bugünkü durum özeti |
| `tests/` | Birim testleri |

---

## 9. Şu an neredeyiz ve sırada ne var

**Çalışan:**
- Orin'de tek kamera karesinden kapı + derinlik + 11 sınıf her karede; iç mekan 22 / dış mekan 19 sınıf modülü otomatik seçimle.
- 30,6 FPS (drone modunda 33,1 FPS).
- Gerçek ofis çekiminde ve gerçek dış mekan karelerinde doğrulandı.

**Öncelikli (kullanıcı, 2026-11-24): 60 FPS'e olabildiğince yaklaşmak.**
1. Refiner ve son işlemin ~10 ms'lik ek yükünü kısmak. Drone modu sonrası en büyük darboğaz bu.
2. 384 çözünürlük + kapı başlıklarının o çözünürlükte ince ayarı ve 448 çıktılarından öz-damıtma (FixRes / MSAD fikri). Çekirdek donmuş kalır; hedef ~55–60 FPS ve en fazla 1 puan kayıp.
3. Aşama 2 profili (derinlemesine / 1×1 / eleman bazlı) ve 2:4 seyreklik üst sınır testi.
4. Uzun vade: eklenti uyumlu, GPU-dostu çekirdek v5. Mevcut çekirdeğin ara özelliklerini taklit edecek şekilde damıtılır; böylece bütün eklentiler yeniden eğitilmeden çalışır.

**Kamera geldi, takılmayı bekliyor:**
- IMX219 sürücü overlay'i, 1280×720 @ 60 modu, kısa pozlama, kalibrasyon.
- Gerçek kamerayla ilk kayıt ve test.

**Açık sorunlar:**
- Gerçek gece segmentasyonu.
- Derinlikte öğretmen farkı.
- Sadece çubuğu görünen yarım kapılar (0,505).
- 8 px altı kapılar.
- Uzak kapı hassasiyeti.
- Drone modunda segmentasyon uyum kontrolünün düşük çıkması (0,22). Muhtemelen başlığın her 3 karede çalışmasından; doğrulanacak.

**Kurallar (kullanıcıyla anlaşılan):**
- Çekirdek release_v4 dondurulmuş kalır.
- Genel metrikler (seg / derinlik / şerit) bozulmadan iyileştirme yapılır.
- Zayıf ara modellerin demosu gösterilmez; önce ölçüt geçilir.
- Commit ve push sadece istenince.
- Yüksek kota harcayan `claude -p` zincirleri için önce izin alınır.
- Kullanıcının e-postası dış servislere gönderilmez.
