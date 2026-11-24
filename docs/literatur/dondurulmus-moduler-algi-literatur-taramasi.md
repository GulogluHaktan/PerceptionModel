# Literatür Taraması Raporu: Dondurulmuş genel çok görevli algı modeli + takılıp çıkarılabilir modüller + sentetik veriyle kendini geliştirme (uç cihaz, drone yarışı kapalı döngü)

**Tarih**: 2026-11-21
**Taranan kaynaklar**: arXiv MCP (`search_papers`, `get_abstract`), Semantic Scholar MCP (`get_paper_details`; `search_papers` oturum boyunca hız sınırına takıldı), OpenAlex REST API (curl ile, e-posta parametresi ve API anahtarı olmadan; `group_by=publication_year`, `group_by=primary_location.source.id`, `/sources`)
**Anahtar kelimeler**: frozen backbone multi-task perception; panoptic driving perception; vision foundation model + task heads on Jetson; adapter / side-tuning / ladder side network; self-expanding pre-trained model (SEMA); continual test-time adaptation; drone racing gate detection; Gaussian splatting simulator / synthetic data; diffusion synthetic data for adverse weather; VLM data engine; NIR in-cabin; ground-to-aerial cross-view adaptation; open-loop vs closed-loop evaluation
**İncelenen makale sayısı**: ~60 başlık/özet tarandı, 52'si kaynakçaya alındı

### Varsayımlar (kullanıcıya soru sorulmadı)
- Zaman aralığı: trend için 2019–2026; temel çalışmalarda tarih sınırı yok. **2026 sayıları kısmi** (yıl Kasım'da), yine de çoğu alt-temada 2025'i şimdiden geçmiş durumda.
- Dergi kriteri: Scopus indeksli + robotik/uç cihaz algı tematik uyumu, OA tercihli. Diamond OA zorunlu tutulmadı.
- "Bizim sistem" tanımı kullanıcının verdiği özetten alındı: dondurulmuş çok görevli öğrenci model + kapı ek modülleri (köşe düzeltici, yakın kapı başlığı); simülasyonda kapalı döngüde GateNet'e karşı 87.5'e 81.2 başarı; genel metrikler (seg/derinlik/şerit) değişmiyor. Bu sayılar raporda doğrulanmadı; olduğu gibi kullanıldı.
- Semantic Scholar arama ucu hız sınırına takıldığı için atıf sayıları ağırlıkla OpenAlex'ten alındı. Semantic Scholar yalnızca birkaç makalenin yayın durumunu (venue, DOI) doğrulamak için kullanıldı.

---

## 1. Özet

Alt-temaların hepsinde yayın hacmi 2023'ten sonra hızla artıyor. En dik artışlar vision PEFT/adapter (2023: 121 → 2026: 1303), test-time adaptation (135 → 1015) ve 3DGS + robotik/simülasyon (2024: 112 → 2026: 625) alanlarında. Drone yarışında ise gate algısı hâlâ **göreve özel dedektörlerle** yapılıyor (GateNet, PencilNet, MonoRace'in segmentasyon ağı). Genel/dondurulmuş bir modelin üstüne takılan modülün özel dedektörü **kapalı döngüde** geçtiğini gösteren bir çalışmaya rastlanmadı. Bu, bizim sistemin doldurduğu en net boşluk. Bizim sistemin de **henüz** kapatmadığı en büyük boşluk şu: **"ne zaman yeni modül eklenmeli" kararını otomatik veren** (SEMA tarzı kayma dedektörü) ve bu kararı **sentetik veri üretimi (3DGS/diffusion/VLM) + modül eğitimi + kapalı döngü doğrulama** ile birleştiren uçtan uca bir döngü. Literatürde bu parçalar ayrı ayrı var ama birleştirilmemiş. Dergi önerisi: tematik uyum için ilk sırada **IEEE Robotics and Automation Letters**, OA tercih edilirse **Drones (MDPI)** veya **IEEE Access**.

---

## 2. Trend Analizi

### Yıl bazlı yayın hacmi (OpenAlex, başlık+özet araması)

| Yıl | "drone racing" | çok görevli algı (driving/robot) | vision PEFT/adapter/side-net | test-time adaptation | diffusion + sentetik veri (algı) | CL + PTM + adapter/prompt | 3DGS + robot/sim/drone |
|---|---|---|---|---|---|---|---|
| 2019 | 25 | 16 | 9 | 2 | 2 | – | – |
| 2020 | 29 | 22 | 13 | 6 | 4 | – | – |
| 2021 | 29 | 17 | 19 | 22 | – | 1 | – |
| 2022 | 27 | 34 | 34 | 62 | 5 | 13 | – |
| 2023 | 23 | 60 | 121 | 135 | 38 | 24 | – |
| 2024 | 30 | 80 | 304 | 267 | 90 | 54 | 112 |
| 2025 | 40 | 131 | 611 | 488 | 267 | 65 | 416 |
| 2026* | 57 | 194 | 1303 | 1015 | 342 | 102 | 625 |

\*2026 kısmi (Kasım). Sorgu dizeleri: `"drone racing"`; `"multi-task" AND perception AND (driving OR robot)`; `("side-tuning" OR "side network" OR adapter OR "parameter-efficient") AND ("vision transformer" OR segmentation OR "dense prediction")`; `"test-time adaptation"`; `diffusion AND "synthetic data" AND (segmentation OR detection OR perception)`; `"continual learning" AND ("pre-trained" OR pretrained) AND (adapter OR prompt)`; `"gaussian splatting" AND (robot OR navigation OR simulator OR drone)`. Sayılar kaba anahtar kelime eşleşmesi olduğundan mutlak değerlere değil eğilime bakılmalı.

### Yükselen alt-temalar
- **Dondurulmuş VFM + hafif başlıklar / görev adapterleri.** DINOv2/v3 dondurulup üstüne görev başlıkları ya da LoRA/MoE adapterleri ekleniyor. Örnekler: VPEngine (Jetson AGX Orin'de ≥50 Hz, derinlik+tespit+seg başlıkları), DPNeXt (2026), ViT-Split (2025), "Orchestrate VFMs" (2026), D2MoE (Kasım 2026, dondurulmuş DINOv3 + görev MoE). Odak, tam ince ayardan **öncülleri koruyan** adaptasyona kayıyor.
- **3DGS hem simülatör hem veri üreteci olarak kullanılıyor.** SOUS VIDE/FiGS (3DGS içinde drone politikası, sıfır atışlı sim-to-real), ReaDy-Go (2026), EmbodiedSplat (2025). Veri üretimi tarafında Cut-and-Splat, "GS is an Effective Data Generator for 3D Detection" ve Deogan ve ark. (AMR'ler için 3DGS veri seti). 2024'ten 2026'ya yaklaşık 5,5 kat artış var.
- **Üretken veriyle alan adaptasyonu.** ControlUDA (diffusion ile hava koşulu üretimi), WeatherProof (dil rehberli). VLM tabanlı otomatik etiketleme/veri motoru: AIDE, Najibi ve ark. 2023.
- **Kendini genişleten / modüler sürekli öğrenme.** SEMA (CVPR 2025), RanPAC ve PILOT araç kutusu. "Evolving Without Ending" (2026) bunu çok görevli panoptik algıya taşımaya başlıyor.
- **Drone yarışında saha başarısı ve uçtan uca öğrenme.** Swift (Nature 2023, 1234 atıf), MonoRace (A2RL 2025 şampiyonu, monoküler gate segmentasyonu + 500 Hz NN kontrol), SkyDreamer (model tabanlı RL, segmentasyon maskesinden motor komutuna).
- **Açık döngü metriklerinin kapalı döngüyü öngörme gücünün sorgulanması.** NAVSIM/Bench2Drive korelasyon çalışması (2026), Bouzidi ve ark. 2025, EPSM güvenlik metriği (2025).

### Durağan/düşüşte olan yaklaşımlar
- **Sıfırdan eğitilen, tam ince ayarlı çok görevli sürüş ağları** (YOLOP → YOLOPv2 → TwinLiteNet+ → TriLiteNet → RMT-PPAD). Hâlâ yayın çıkıyor ama katkılar artımsal kalıyor: daha hafif bir omurga, birkaç puan mIoU. Bunun yerine sıkıştırma/KD (2511.05557) ve VFM tabanlı tasarımlar öne geçiyor.
- **Klasik domain randomization ile gate dedektörü eğitimi** (Deep Drone Racing 2019, AirSim Drone Racing Lab 2020). Çok atıflı ama yerini 3DGS/fotogerçekçi simülasyona ve uçtan uca RL'ye bırakıyor.
- **Ham "drone racing" yayın hacmi** 2019–2023 arasında yatay seyretti (23–29/yıl). Artış 2024'te başladı (A2RL ve Swift etkisi).

---

## 3. Gap (Boşluk) Analizi

### Gap 1: Genel/dondurulmuş model + takılan modülün, göreve özel dedektöre karşı kapalı döngüde karşılaştırılması
- **Kanıt**: Drone yarışı algısı neredeyse tamamen özel ağlarla yapılıyor: GateNet (IROS 2021), PencilNet (RA-L 2022), Qiao ve ark. 2024, MonoRace 2026, AlphaPilot 2020. Paylaşılan omurga + çok başlıklı uç cihaz çalışmaları (VPEngine, DPNeXt, Visual Perception Engine) yalnızca açık döngü metrik ve gecikme raporluyor, kapalı döngü görev başarısı ölçmüyor. Sürüş tarafında açık döngü metriklerin kapalı döngüyü zayıf öngördüğü gösterilmiş (2605.00066, 2505.05638), ama bu analizler **planlayıcı** için yapılmış, **algı modülü** için değil.
- **Neden önemli**: "Genel model mi, özel model mi" sorusu robotikte maliyet açısından belirleyici (tek omurga = tek bellek/gecikme bütçesi). Karşılaştırma görev başarısıyla yapılmadıkça yanıt eksik kalıyor.
- **Mevcut çalışmayla ilişkisi**: **Büyük ölçüde kapatıyor** (simülasyonda 87.5'e 81.2, genel metrikler sabit). Üstüne gidilecekler: (a) gerçek donanımda (Orin NX + IMX219) kapalı döngü tekrarı; (b) algı modülü için "açık döngü metrik ↔ kapalı döngü başarı" korelasyon analizi, yani hangi metrik (köşe hatası, yakın kapı recall'u, zamansal titreşim) başarıyı öngörüyor; (c) kapalı döngüdeki ±10–15 puanlık gürültüye karşı çoklu tohum ve güven aralığı raporu. Literatürde bu üçü de yok.

### Gap 2: Modül eklemenin otomatik tetiklenmesi (SEMA tarzı) yoğun tahmin / robotik algıda uygulanmamış
- **Kanıt**: SEMA, reuse/add kararını temsil tanımlayıcılarıyla (distribution-shift göstergesi) otomatik veriyor, ama yalnızca sınıflandırma CL kıyaslamalarında gösterilmiş. RanPAC ve PILOT da sınıflandırma odaklı. Yoğun/çok görevli tarafta CTTA yöntemleri (CoTTA 2022, CBNA 2022, DAT 2023, TransAdapt 2023) modülü **güncelliyor**, yeni modül **eklemiyor**; uzun vadede hata birikimini kendileri kısıt olarak kabul ediyor. "Evolving Without Ending" (2026) çok görevli CL'ye geçiyor ama kayma tetiklemeli genişleme yapmıyor.
- **Neden önemli**: Dondurulmuş omurga + modül kütüphanesi ancak **ne zaman** ve **hangi** modülün ekleneceğine kendi karar verirse "kendini geliştiren" sistem olur. Aksi halde insan eliyle modül ekleniyor demektir.
- **Mevcut çalışmayla ilişkisi**: **Kapatmıyor.** Kapı modülleri elle tasarlanıp eklendi. Doğal ilerleme şu olur: dondurulmuş özniteliklerde (mevcut başlıkların belirsizliği + öznitelik istatistiği) kayma dedektörü çalışsın, gece/sis/NIR/hava görüntüsü geldiğinde yeni adapter açılsın, eski koşullarda router onu devre dışı bıraksın. Genel metriklerin değişmemesi zaten bizim sistemin güçlü yanı; SEMA'nın stabilite iddiasını yoğun görevlerde test etmek için iyi bir zemin.

### Gap 3: "Kayma tespiti → sentetik veri üret → modül eğit → kapalı döngüde doğrula" döngüsü birleştirilmemiş
- **Kanıt**: Parçalar ayrı ayrı mevcut. 3DGS ile veri/sahne üretimi: Cut-and-Splat 2025, GS-3D-det 2025, Deogan 2025, Niżeniec 2025, SOUS VIDE 2024. Diffusion ile hava/koşul üretimi: ControlUDA 2024. VLM/oto-etiket veri motoru: AIDE 2024, Najibi 2023. Ancak taranan örneklemde bu üretim adımları bir **kayma dedektörüyle** tetiklenmiyor ve sonuç bir **kapalı döngü** görev metriğiyle kapatılmıyor. AIDE, açık küme nesne tespiti için en yakın örnek, ama açık döngü ve sürüş tespitiyle sınırlı.
- **Neden önemli**: Uydu, NIR kabin, drone gibi etiketli verinin az olduğu alanlarda insan etiketine bağımlılığı azaltan tek ölçeklenebilir yol bu.
- **Mevcut çalışmayla ilişkisi**: **Kısmen.** Kapalı döngü simülasyon ve kapı modülleri hazır. Eksik olan, döngünün otomatik tetiklenmesi ve üretken veri kaynağının (3DGS/diffusion/LLM sahne betiği) sisteme bağlanması. Raporun en güçlü **katkı fırsatı** bu boşluk.

### Gap 4: Genişleyen modüllerin uç cihazda (TensorRT/Jetson) maliyeti raporlanmıyor
- **Kanıt**: Side-network ve adapter çalışmaları (Side-Tuning 2020, LST 2022, HST 2023, LAST 2024, Fading Side Networks 2026, D2MoE 2026) verimliliği **eğitim belleği / parametre sayısı** üzerinden ölçüyor. Jetson gecikmesi raporlanmıyor ya da çok sınırlı. VPEngine Jetson AGX Orin'de çalışıyor ama başlık seti sabit; çalışma sırasında modül eklendiğinde motorun yeniden derlenme maliyeti ve router yükü incelenmemiş.
- **Neden önemli**: Kendini genişleten bir modelin sahada kullanılabilmesi, her yeni modülün gecikme bütçesine nasıl yansıdığına bağlı. Orin NX, AGX'ten belirgin biçimde daha kısıtlı bir cihaz.
- **Mevcut çalışmayla ilişkisi**: **Kısmen.** Orin NX ölçümleri var (docs/ORIN_RESULTS.md). Modül sayısına karşı gecikme/bellek eğrisi ve sıcak takma (engine yeniden derleme gerektirmeyen ayrı engine'ler) raporlanırsa literatürde boş bir hücre doldurulmuş olur.

### Gap 5 (olası, sınırlı kanıt): Tek dondurulmuş omurgada farklı bakış açısı ve modalitelere (araç → drone/uydu, RGB → NIR) modülle geçiş
- **Kanıt**: Örnekler dağınık ve her biri tek bir geçişe bakıyor: CROVIA 2023 (araç → drone segmentasyonu), NIR + LoRA ViT (uzaktan algılama, 2024), sürücü izleme için çapraz görüş/çapraz modalite UDA (2025), TICaM (kabin veri seti). AerialOcc 2026 hava algısında göreve özel başlığı kaldırmayı öneriyor. Bunların hiçbiri bir ana modeli koruyup her alan için modül eklemiyor.
- **Neden önemli**: "Bir omurga, çok alan" iddiası en çok bu tür sert alan kaymalarında sınanır.
- **Mevcut çalışmayla ilişkisi**: **Kapatmıyor** (yol haritasında var). Örneklem küçük olduğu için bu bir ön izlenim; daha hedefli bir taramayla doğrulanmalı.

### Gözlem (tek kaynağa dayalı, gap olarak sunulmuyor)
- Gate algısında **sürekli öğrenme** yalnızca Qiao ve ark. 2024'te ele alınmış (aydınlatma değişimi). Takip çalışması bulunamadı; bu bir "yetim" çalışma olabilir.
- MonoRace, kapı geometrisinden yararlanan çevrim dışı kalibrasyon optimizasyonunu kritik buluyor. Bizim köşe düzeltici de geometri öncülü kullanıyor; bu bağlantı bir sonraki çalışmada işlenebilir.

---

## 4. Yöntem / Materyal Karşılaştırması

| Yöntem/Materyal | Avantaj | Dezavantaj | Geçerli olduğu koşul | Kaynak makale(ler) |
|---|---|---|---|---|
| Tam ince ayarlı çok görevli CNN (tespit + sürülebilir alan + şerit) | Gerçek zamanlı, tek ağ, olgun kıyaslar (BDD100K) | Yeni görev/alan için tüm ağ yeniden eğitiliyor; görev çatışması; derinlik genelde yok | Sabit görev seti, tek alan (yol) | YOLOP, YOLOPv2, TwinLiteNet+, TriLiteNet, RMT-PPAD, AurigaNet |
| Sıkıştırma/KD ile çok görevli model | Uç cihaza sığar | Öğretmen ve görev seti sabit, genişleme yok | Dağıtım öncesi tek seferlik | 2511.05557 |
| Dondurulmuş VFM + paralel görev başlıkları | Paylaşılan öznitelik, görev ekleyince eski görevler bozulmuyor; Jetson'da ≥50 Hz | VFM'nin öncülleri yoğun geometrik görevlere tam uymuyor; başlık seti çoğunlukla sabit | Bellek kısıtlı robot, çok görev | VPEngine, DPNeXt, ViT-Split, Orchestrate-VFM |
| Adapter / LoRA / görev MoE (omurga içine) | Az parametreyle güçlü uyum, negatif transferi azaltıyor | Omurga içine girdiği için **modülü çıkarmak/sıcak takmak zor**; uç cihaz gecikmesi raporlanmıyor | Eğitim bütçesi sınırlı, görev seti biliniyor | AdaptFormer, ViT-Adapter, D2MoE |
| Side-network / ladder side-tuning | Omurgada geri yayılım yok, düşük eğitim belleği; modül fiziksel olarak ayrı (tak-çıkar) | Kapasite adapterden düşük olabilir; yoğun tahminde az test edilmiş | Dondurulmuş omurga zorunlu, cihaz üstü eğitim | Side-Tuning, LST, HST, LAST, CST, Fading Side Networks |
| Kendini genişleten adapter karışımı (SEMA) | Kaymaya göre ekle/yeniden kullan kararı otomatik, alt-doğrusal büyüme, tekrar belleği yok | Yalnızca sınıflandırma; router ve tanımlayıcı ek maliyet getiriyor | Görev kimliği bilinmeyen akış | SEMA, RanPAC, PILOT |
| Sürekli test zamanı adaptasyonu | Etiketsiz, çevrimiçi | Hata birikimi/unutma riski; modeli değiştirdiği için **genel metrikleri bozabilir** | Yavaş, sürekli kayma (hava, ışık) | CoTTA, CBNA, DAT, TransAdapt, Kerssies 2022 |
| Göreve özel gate dedektörü | Hafif ve hızlı; kapı geometrisine özel | Başka görevlere taşınmıyor; ışık/alan değişimine kırılgan | Tek görev, bilinen kapı tipi | GateNet, PencilNet, Qiao 2024, Morales 2020 |
| Gate segmentasyonu + geometri tabanlı durum kestirimi | Saha başarısı (A2RL), monoküler | Pist/kapı tipine özel kalibrasyon gerekiyor | Yarış ortamı | MonoRace, AlphaPilot |
| Uçtan uca RL (pikselden/maskeden komuta) | Şampiyon seviyesi hız, ara temsil gerekmiyor | Algı yeniden kullanılamıyor; ayrıcalıklı simülasyon bilgisi gerekiyor; genelleme sınırlı | Sabit pist, yoğun simülasyon | Swift, SkyDreamer, Dream to Fly, Song 2021 |
| Domain randomization ile sentetik eğitim | Ucuz, etiket bedava | Gerçekçilik açığı; tasarım elle yapılıyor | Basit geometri (kapı) | Deep Drone Racing, AirSim DRL, FlightGoggles |
| 3DGS simülatör / veri üreteci | Fotogerçekçi ve hızlı render (130 fps), gerçek sahneden üretim | Sahne başına yakalama gerekiyor; dinamik nesne ve hava koşulu zayıf; etiket aktarımı ayrı bir problem | Hedef sahne biliniyor | SOUS VIDE/FiGS, ReaDy-Go, EmbodiedSplat, Cut-and-Splat, GS-3D-det, Deogan 2025 |
| Diffusion ile koşul (gece/sis/yağmur) üretimi | Etiketli kaynaktan kontrollü hedef koşul | Geometri/etiket tutarlılığı garanti değil; yüksek hesap maliyeti | Etiketsiz hedef alan, 2D görev | ControlUDA, WeatherProof, D2SL |
| VLM/LLM ile veri motoru ve oto-etiket | Uzun kuyruk/yeni sınıfa ölçekleniyor | Halüsinasyon, doğrulama gerekiyor | Açık küme tespit | AIDE, Najibi 2023, VLM-AD anketi |

**Çelişkili bulgular**:
- *Uçtan uca mı, modüler mi?* SkyDreamer ve Swift uçtan uca/RL ile şampiyon seviyesine ulaşıyor. MonoRace ise açık bir algı (segmentasyon) + geometri ara katmanıyla yarışı kazanıyor ve uçtan uca sistemlerin "yarışa fazla özel" kaldığını vurguluyor. Yarışta açık ara temsilin değeri literatürde tartışmalı. Bizim sonucumuz modüler tarafı destekliyor.
- *Adapter mı, side-network mü?* D2MoE ve ViT-Adapter omurga içine yerleşen modüllerle daha yüksek doğruluk rapor ediyor. Side-network çalışmaları ise bellek/geri yayılım avantajını öne çıkarıyor. Tak-çıkar ve "genel metrikler değişmez" garantisi açısından side-network/başlık yaklaşımı daha güvenli. Doğruluk farkının yoğun görevlerde ne kadar olduğu açık bir soru.
- *CTTA vs dondurma*: CTTA çevrimiçi uyum sağlıyor ama modeli değiştiriyor (unutma riski). Dondurma + modül yaklaşımı stabiliteyi garanti ediyor ama etiketsiz çevrimiçi uyumdan vazgeçiyor.

**Mevcut çalışmanın konumlandırması**: Bizim sistem, tabloda "dondurulmuş VFM + paralel başlık" ile "side-network" satırlarının kesişiminde duruyor (modüller ayrı, genel metrikler sabit) ve göreve özel bir dedektörü (GateNet) **kapalı döngüde** geçiyor. Bu kombinasyon için taranan örneklemde doğrudan bir emsal bulunamadı. Zayıf kaldığı yerler: (i) modül ekleme kararı elle veriliyor (SEMA'nın otomatik tetikleyicisi yok); (ii) üretken veri kaynakları sisteme bağlanmamış; (iii) sonuç yalnızca simülasyonda gösterildi; (iv) modül sayısıyla uç cihaz maliyetinin nasıl ölçeklendiği raporlanmadı.

---

## 5. Dergi Önerileri

**Kullanılan kriterler**: Varsayılan: Scopus indeksli (dergi listesinden teyit edilmeli), robotik / uç cihaz algı / UAV tematik uyumu, OA tercih. Aday havuzu, OpenAlex'te konu sorgusunun (`multi-task|adapter|gate detection|drone racing|sim-to-real` × `perception|segmentation|depth` × `embedded|Jetson|edge|onboard|UAV|drone`, 2022–2026, makale) venue dağılımından ve drone yarışı makalelerinin en çok atıf alan venue'lerinden çıkarıldı. OA, DOAJ ve APC bilgisi OpenAlex `/sources` ucundan alındı.

| Dergi | Yayıncı | ISSN-L | Açık Erişim Tipi (OpenAlex) | Neden uygun |
|---|---|---|---|---|
| IEEE Robotics and Automation Letters (RA-L) | IEEE | 2377-3766 | Hibrit (is_oa=false; APC 2645 USD, isteğe bağlı OA) | Drone yarışı algısının ana yuvası (PencilNet, nano-drone racing, Learned Inertial Odometry). ICRA/IROS sunum seçeneğiyle kısa ve hızlı yayın. Kapalı döngü + uç cihaz vurgusuna en iyi tematik uyum. 2yr citedness 8.06 |
| Drones | MDPI | 2504-446X | Gold OA, DOAJ (APC 3137 USD) | UAV algısı ve drone yarışı yayınları burada (örn. VECTOR 2024). Konu sorgusunda 8 makale. OA tercihini karşılıyor ve hızlı. 2yr citedness 8.01 |
| IEEE Access | IEEE | 2169-3536 | Gold OA, DOAJ (APC 2075 USD) | Konu sorgusunda en yoğun ikinci venue (19). Geniş kapsamlı ve hızlı; sistem + deney ağırlıklı bir makale için güvenli OA seçenek. APC adaylar içinde en düşük |
| IEEE Open Journal of Intelligent Transportation Systems | IEEE | 2687-7813 | Gold OA, DOAJ (APC 2075 USD) | Çerçeve "araç önce" yönüne (seg/şerit/derinlik + NIR kabin) kayarsa uygun OA dergi. Daha küçük ve seçici (h=40) |
| Journal of Real-Time Image Processing | Springer | 1861-8200 | Hibrit (APC 3290 USD) | Orin NX gecikme/bellek ve modül sayısına göre ölçeklenme (Gap 4) ana katkıysa en uygun niş. 2yr citedness 7.45 |
| Frontiers in Robotics and AI | Frontiers | 2296-9144 | Gold OA, DOAJ (APC 2353 USD) | "Kendini geliştiren modüler robot algısı" gibi kavramsal ve sistem odaklı bir çerçeve için OA alternatif; daha düşük etki (2yr 5.52) |

Ek not: VPEngine'in yayınlandığı **IEEE Robotics and Automation Practice** yeni bir dergi; Scopus durumu belirsiz olduğu için listeye alınmadı. **Field Robotics** drone yarışı saha makalelerini yayımlıyor, ancak OpenAlex'te OA/DOAJ bilgisi boş göründüğü için doğrulanmadan önerilmedi.

**Öneri**: Kapalı döngü (tercihen gerçek donanım) + dondurulmuş genel model hikâyesiyle önce **RA-L**. OA şartsa **Drones**. Ağırlık uç cihaz ölçeklenmesindeyse **JRTIP**.

> Not: Scopus/WoS indeks durumu, çeyrek (Q) seviyesi ve güncel APC/etki puanı dergi web sitesinden veya Scopus Source List'ten teyit edilmelidir. OpenAlex Scopus indeksini doğrudan vermiyor; buradaki "2yr citedness" OpenAlex'in kendi metriği, JIF değil.

---

## 6. Kaynakça (taranan başlıca makaleler)

Kaynak kısaltmaları: **AX** = arXiv MCP, **S2** = Semantic Scholar MCP, **OA** = OpenAlex REST.

**Dondurulmuş VFM / çok görevli algı**
1. Wu ve ark., YOLOP: You Only Look Once for Panoptic Driving Perception, 2021/2022, Machine Intelligence Research. arXiv:2108.11250 (AX, OA: 580 atıf)
2. Han ve ark., YOLOPv2, 2022. arXiv:2208.11434 (AX)
3. Che ve ark., TwinLiteNet+, 2024. arXiv:2403.16958 (AX)
4. Che & Lam, TriLiteNet, 2025. arXiv:2509.04092 (AX)
5. Wang ve ark., RMT-PPAD, 2025. arXiv:2508.06529 (AX)
6. Ghasemzadeh & Dehghani, AurigaNet, 2026. arXiv:2602.10660 (AX)
7. Wang ve ark., Compressing Multi-Task Model for AD via Pruning and KD, 2025. arXiv:2511.05557 (AX)
8. Łucki ve ark., Visual Perception Engine (VPEngine), 2025, IEEE Robotics and Automation Practice, DOI 10.1109/RAP.2026.3733847. arXiv:2508.11584 (AX, S2)
9. Kang ve ark., DPNeXt, 2026. arXiv:2607.16012 (AX)
10. Li ve ark., ViT-Split, 2025. arXiv:2506.03433 (AX)
11. Han ve ark., Learning to Orchestrate VFMs for Multi-Task Dense Prediction, 2026. arXiv:2606.15765 (AX)
12. Zhuge ve ark., D2MoE: Mixture of Task Adapters, 2026. arXiv:2611.02769 (AX)
13. Liu ve ark., AerialOcc, 2026. arXiv:2611.18770 (AX)
14. Yuan ve ark., Evolving Without Ending: Continual Panoptic Perception, 2026. arXiv:2601.15643 (AX)

**Adapter / side-network / kendini genişletme**
15. Chen ve ark., AdaptFormer, NeurIPS 2022 (OA: 1432 atıf)
16. Chen ve ark., Vision Transformer Adapter for Dense Predictions, 2022 (OA: 936 atıf)
17. Zhang ve ark., Side-Tuning, ECCV 2020, LNCS (OA: 348 atıf)
18. Sung ve ark., LST: Ladder Side-Tuning, NeurIPS 2022. arXiv:2206.06522 (AX, OA: 389 atıf)
19. Lin ve ark., Hierarchical Side-Tuning for ViTs, 2023. arXiv:2310.05393 (AX)
20. Tang ve ark., Low-rank Attention Side-Tuning, 2024. arXiv:2402.04009 (AX)
21. Chen, CST: Calibration Side-Tuning (detection), 2024. arXiv:2402.12736 (AX)
22. Zhang ve ark., Fading Side Networks via Masked Dual Path Distillation, 2026. arXiv:2604.09088 (AX)
23. Wang ve ark., SEMA: Self-Expansion of Pre-trained Models with Mixture of Adapters, CVPR 2025, DOI 10.1109/CVPR52734.2025.00943. arXiv:2403.18886 (AX, S2: 83 atıf)
24. McDonnell ve ark., RanPAC, 2023. arXiv:2307.02251 (AX)
25. Sun ve ark., PILOT toolbox, 2023. arXiv:2309.07117 (AX)

**Test zamanı / sürekli adaptasyon, zorlu koşullar**
26. Wang ve ark., Continual Test-Time Domain Adaptation (CoTTA), CVPR 2022 (OA: 1100 atıf)
27. Klingner ve ark., Continual BatchNorm Adaptation, 2022. arXiv:2203.01074 (AX)
28. Ni ve ark., Distribution-Aware CTTA for Semantic Segmentation, 2023. arXiv:2309.13604 (AX)
29. Das ve ark., TransAdapt, 2023. arXiv:2302.14611 (AX)
30. Kerssies ve ark., Evaluating CTTA for Contextual and Semantic Shifts, 2022. arXiv:2208.08767 (AX)
31. Gan ve ark., Decorate the Newcomers: Visual Domain Prompt for CTTA, AAAI 2023 (OA: 166 atıf)
32. Shen ve ark., ControlUDA, 2024. arXiv:2402.06446 (AX)
33. Gella ve ark., WeatherProof (language guidance), 2024. arXiv:2403.14874 (AX)
34. Sun ve ark., D2SL foggy segmentation, 2024. arXiv:2404.04807 (AX)
35. Leta ve ark., LiDSeg (adverse weather efficient seg), 2026. arXiv:2611.20864 (AX)

**Drone yarışı**
36. Pham ve ark., GateNet (fish-eye gate perception), IROS 2021, DOI 10.1109/IROS51168.2021.9636207 (OA: 13 atıf)
37. Pham ve ark., PencilNet, RA-L 2022, DOI 10.1109/LRA.2022.3207545. arXiv:2207.14131 (AX, S2, OA)
38. Qiao ve ark., Continual Learning for Robust Gate Detection under Dynamic Lighting, 2024. arXiv:2405.01054 (AX)
39. Morales ve ark., Image Generation for Efficient NN Training in Drone Racing, 2020. arXiv:2008.02596 (AX)
40. Kaufmann ve ark., Champion-level drone racing using deep RL (Swift), Nature 2023 (OA: 1234 atıf)
41. Bahnam ve ark., MonoRace, 2026. arXiv:2601.15222 (AX)
42. Verraest ve ark., SkyDreamer, 2025. arXiv:2510.14783 (AX)
43. Foehn ve ark., AlphaPilot, Autonomous Robots 2020. arXiv:2005.12813 (AX, OA: 203 atıf)
44. Loquercio ve ark., Deep Drone Racing: Sim-to-Real with Domain Randomization, T-RO 2019. arXiv:1905.09727 (AX, OA: 339 atıf)
45. Madaan ve ark., AirSim Drone Racing Lab, 2020. arXiv:2003.05654 (AX)
46. Lamberti ve ark., Sim-to-Real Framework for Nano-drone Racing, RA-L 2024. arXiv:2312.08991 (AX, OA)
47. Hanover ve ark., Autonomous Drone Racing: A Survey, T-RO 2024 (OA: 235 atıf)

**3DGS / üretken veri / VLM veri motoru**
48. Low ve ark., SOUS VIDE (FiGS 3DGS simulator), 2024. arXiv:2412.16346 (AX)
49. Yoo ve ark., ReaDy-Go, 2026. arXiv:2602.11575 (AX); Chhablani ve ark., EmbodiedSplat, 2025. arXiv:2509.17430 (AX)
50. Vanherle ve ark., Cut-and-Splat, 2025. arXiv:2504.08473 (AX); Zanjani ve ark., GS is an Effective Data Generator for 3D Detection, 2025. arXiv:2504.16740 (AX); Deogan ve ark., Synthetic Dataset Generation for AMRs using 3DGS, 2025. arXiv:2506.05092 (AX); Niżeniec & Iwanowski, 2025. arXiv:2512.13411 (AX)
51. Liang ve ark., AIDE: Automatic Data Engine for Object Detection in AD, 2024. arXiv:2403.17373 (AX); Najibi ve ark., Unsupervised 3D Perception with 2D VL Distillation, 2023. arXiv:2309.14491 (AX); Zhou ve ark., VLMs in AD: Survey, 2023. arXiv:2310.14414 (AX)

**Alan/modalite kayması ve değerlendirme**
52. Truong ve ark., CROVIA (car → drone cross-view), 2023. arXiv:2304.07199 (AX); Ulku ve ark., NIR + LoRA for ViTs in Remote Sensing, 2024. arXiv:2405.17901 (AX); Bhalla ve ark., Cross-View Cross-Modal UDA for DMS, 2025. arXiv:2511.12196 (AX); Katrolia ve ark., TICaM, 2021. arXiv:2103.11719 (AX); Wang ve ark., Do Open-Loop Metrics Predict Closed-Loop Driving?, 2026. arXiv:2605.00066 (AX); Bouzidi ve ark., Closing the Loop: Motion Prediction beyond Open-Loop Benchmarks, 2025. arXiv:2505.05638 (AX); Gamerdinger ve ark., EPSM, 2025. arXiv:2512.15195 (AX)

### Yöntem notu / sınırlamalar
- Semantic Scholar `search_papers` oturum boyunca 429 döndürdüğü için çapraz doğrulama esas olarak arXiv + OpenAlex ile yapıldı. Semantic Scholar yalnızca SEMA, VPEngine ve PencilNet'in hakemli yayın bilgisini doğruladı.
- Gap 5 ve "yetim çalışma" gözlemi küçük örnekleme dayanıyor.
- Tam metin okunmadı. Kısıt/"future work" çıkarımları özetlerden yapıldı.
