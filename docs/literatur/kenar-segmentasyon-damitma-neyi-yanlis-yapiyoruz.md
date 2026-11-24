# Literatür Taraması: Kenar cihazda ince sınıflı segmentasyon — neyi yanlış yapıyoruz?

**Tarih**: 2026-11-23
**Taranan kaynaklar**: arXiv API, OpenAlex API, Semantic Scholar API (MCP sunucuları kayıtlı ama bu oturuma yüklenemedi; aynı veritabanları resmi API'leriyle doğrudan sorgulandı, genel web araması kullanılmadı). Semantic Scholar çoğu sorguda hız sınırına takıldı; atıf/venue bilgisi OpenAlex'ten.
**Anahtar kelimeler**: knowledge distillation semantic segmentation; structured / channel-wise / relational / boundary distillation; real-time semantic segmentation; efficient MaskFormer; frozen backbone + side network / adapter; class-incremental segmentation; Jetson TensorRT segmentation
**İncelenen makale**: ~250 kayıt tarandı, 20 çekirdek makale okundu (aşağıdaki tablo).

---

## 1. Özet — bizim sorunlarımız ve literatürün cevabı

| Bizim gözlem | Literatürdeki karşılığı | Ne yapmalıyız |
|---|---|---|
| Damıtılmış öğrenci 0.60–0.64'te tıkandı, öğretmen 0.76 | Piksel başına (logit) KD yoğun tahminde "sub-optimal"; yapısal/ilişkisel KD gerekli (SKD, CWD, CIRKD) | Sadece çıktı KL'i değil: kanal bazlı KD + piksel-ilişki KD + öğretmen ara-özellik damıtması |
| Maskeler parça parça, kenarlarda taşma | Öğrencinin kenar bölgeleri "bağlam sızıntısı" yüzünden en belirsiz yer; kenar ve gövde ayrı damıtılmalı (BPKD). Gerçek zamanlı ağlar ayrı bir **sınır dalı** taşıyor (PIDNet) | Sınır dalı + kenar/gövde ayrık damıtma; bizim "kenar ağırlıklı kayıp" bunun zayıf bir hali |
| Donmuş 11M çekirdek ince sınıfları ayıramıyor | Donmuş özellikler ancak **büyük temel modellerde** her işe yetiyor (DINOv2); küçük omurgalarda side-network / adapter ile ek kapasite şart (Side-Tuning, LST, ViT-Adapter) | Bizim ARCH 3/5 bir side-network'tü — doğru yön, ama tasarım ev yapımı |
| Kendi mask-head'imiz (ARCH 7) görüntüde Mask2Former'a yaklaşmadı | Kenar için kanıtlanmış verimli mask-classification tasarımları var: YOSO (CVPR'23, gerçek zamanlı panoptik), PEM (CVPR'24, kenar odaklı MaskFormer) | Öğrenci mimarisini sıfırdan icat etmek yerine yayımlanmış verimli bir tasarımı **bizim ağırlıklarımızla, bizim damıtma tarifimizle** eğitmek |
| 448 sıkıştırma (telefon 1080×1920) ince/küçük nesneleri bozuyor | Gerçek zamanlı segmentasyon ağları yüksek çözünürlükte ve oran korunarak eğitilip ölçülüyor (PP-LiteSeg, PIDNet, DDRNet Cityscapes 1024×2048) | Oranı koruyan giriş (ARCH 7 aspect ile +1,6 puan doğrulandı) |

**Makale açısından:** Literatürde kabul gören çerçeve "büyük öğretmen(ler)den kompakt öğrenciye damıtma" (TransKD, T-ITS 2024: büyük transformer'lardan sürüş için kompakt öğrenci). Bizim katkımız şu şekilde savunulabilir: **birden fazla temel ölçekli öğretmenden (iç mekan: COCO-panoptic Mask2Former, dış mekan: Cityscapes Mask2Former) damıtılmış, donmuş bir çekirdeğe takılan modüler kenar öğrencileri + sahne yönlendiricisi, Orin'de 30 FPS.** Hazır modeli doğrudan çalıştırmak katkı değil; öğretmen olarak kullanmak katkı.

---

## 2. Trend analizi (sınırlı örneklemle)

- **Logit KD → yapısal/ilişkisel KD:** 2019 SKD (TPAMI 2020, 247 atıf) piksel başına KD'nin yetersizliğini gösterdi; 2021 CWD (ICCV, 682 atıf), 2022 CIRKD (CVPR, 421 atıf), 2023 BPKD (WACV 2024) bu çizgiyi sürdürüyor: kanal dağılımı, görüntüler arası piksel ilişkileri, kenar/gövde ayrımı.
- **Transformer öğretmen → kompakt öğrenci:** TransKD (2022, T-ITS 2024) patch-embedding ve özellik haritası damıtması; FLOPs %85+ azalma.
- **Gerçek zamanlı segmentasyonda iki kol:** (a) piksel sınıflandırma + çift çözünürlük/sınır dalı (DDRNet, PIDNet CVPR 2023 1108 atıf, PP-LiteSeg); (b) verimli mask-classification (YOSO CVPR 2023, PEM CVPR 2024). Mobil tarafta TopFormer (CVPR 2022, 473), SeaFormer, AFFormer (AAAI 2023), EfficientViT.
- **Temel model + adapter/side-network:** ViT-Adapter (946 atıf), LST, Side-Tuning; DINOv2 (11.885 atıf) donmuş özelliklerin her göreve yettiğini **büyük ölçekte** gösteriyor.

## 3. Gap / bizim için fırsatlar

1. **Çok öğretmenli + modüler damıtma + yönlendirici kenarda:** Taranan damıtma makaleleri tek öğretmen–tek öğrenci; modüllerin sahneye göre seçildiği ve donmuş bir çok-görevli çekirdeği paylaştığı bir kenar sistemi taranan örneklemde görülmedi (sınırlı tarama — makaleden önce MCP ile genişletilmeli).
2. **Donmuş çekirdek korunarak sınıf zenginleştirme:** Class-incremental segmentasyon literatürü genelde aynı ağı güncelliyor (unutma problemi); bizim "çekirdek dokunulmaz, yan ağ ekle" yaklaşımı Side-Tuning/LST çizgisinde ama yoğun tahmin + kenar cihaz bağlamında az çalışılmış.
3. **Deformable attention'ın Orin'e taşınması:** Taranan kenar makalelerinde ölçüm pratiği az; bizim float64 ızgara bulgusu + TRT eklentisiyle 130→49 ms küçük ama raporlanabilir bir mühendislik katkısı.

## 4. Yöntem karşılaştırması (çekirdek makaleler)

| Makale | Yıl / yer | Ne yapıyor | Bize uygulanışı | Dezavantaj |
|---|---|---|---|---|
| SKD (Liu+) | 2019 / TPAMI | Pairwise + holistic (adversarial) yapısal KD | Piksel-ilişki KD'yi ekle | Adversarial kısım kararsız olabilir |
| CWD (Shu+) | 2021 / ICCV | Her kanalı softmax ile dağılıma çevirip KL | En ucuz eklenti; logit KL'imizin yerine/yanına | Kanal sayıları eşleşmeli (22 sınıf logitinde doğal) |
| CIRKD (Yang+) | 2022 / CVPR | Görüntüler arası piksel-piksel / piksel-bölge ilişki KD | Batch içi ilişki damıtması | Bellek (bellek bankası) |
| BPKD (Wang+) | 2023 / WACV'24 | Kenar ve gövde için ayrı KD kayıpları | Doğrudan kenar taşması sorunumuz | Kenar haritası gerek (etiketten türetilir) |
| MasKD (Huang+) | 2022 | Öğrenilen "receptive token" maskeleriyle özellik damıtma | Öğretmen piksel-çözücü özelliklerinden damıtma | Ek eğitim aşaması |
| TransKD (Liu+) | 2022 / T-ITS'24 | Transformer öğretmen → kompakt transformer öğrenci (özellik + patch) | Mask2Former → bizim öğrenci çerçevesi | Mimari eşleşmesi |
| PIDNet (Xu+) | 2023 / CVPR | 3 dallı (detay/bağlam/sınır) gerçek zamanlı ağ | Sınır dalı fikri | Piksel sınıflandırma |
| PP-LiteSeg | 2022 | Hafif çözücü + dikkat füzyonu + SPPM | Hızlı öğrenci adayı | Paddle ekosistemi |
| YOSO (Hu+) | 2023 / CVPR | Gerçek zamanlı panoptik, dinamik konvolüsyonla maske | Mask-classification öğrenci adayı (COCO 46.4 PQ, 45.6 FPS) | Kod hazır, TRT dönüşümü denenmeli |
| PEM (Cavagnero+) | 2024 / CVPR | Prototip tabanlı verimli MaskFormer, kenar cihaz odaklı | En doğrudan aday: "kenar için Mask2Former" | Deformable conv TRT'de eklenti isteyebilir |
| SegFormer | 2021 / NeurIPS | Hafif MLP çözücü + hiyerarşik transformer | Basit güçlü baseline (B0/B1) | Kenar taşması bilinen sorun |
| TopFormer / SeaFormer / AFFormer | 2022–23 | Mobil segmentasyon | Çok hızlı öğrenci | Doğruluk düşük |
| Side-Tuning / LST | 2020 / 2022 | Donmuş omurgaya yan ağ | Bizim ARCH 3/5 tasarımımızın literatür karşılığı | Kapasite yan ağda |
| ViT-Adapter | 2022 | Düz ViT'ye yoğun tahmin adaptörü | Çekirdek donukken adaptör | Büyük omurga varsayımı |
| DINOv2 | 2023 / TMLR | Büyük ölçekli öz-denetimli özellikler | Donmuş özellik ancak büyük modelde yeterli → bizim 11M çekirdek değil | Orin için büyük |
| Mask2Former | 2022 / CVPR | Maskeli dikkatli mask-classification | Öğretmen | Orin'de ağır (eklentiyle 49 ms) |

## 5. Neyi yanlış yaptık — somut liste

1. **Damıtma tarifi zayıftı:** Sadece öğretmen olasılıklarına piksel başına KL (stride-4). Literatür bunun yoğun tahminde yetersiz olduğunu 2019'dan beri gösteriyor. CWD + ilişkisel KD + ara-özellik damıtması eklenmeliydi.
2. **Sınır işini kayıp ağırlığıyla geçiştirdik:** Kenarlar için ayrı bir sınır dalı (PIDNet) ve kenar/gövde ayrı damıtma (BPKD) yok.
3. **Öğrenci mimarisini kendimiz uydurduk:** Kenar için yayımlanmış, test edilmiş verimli mask-classification tasarımları (YOSO, PEM) varken ev yapımı bir sorgu çözücü yazdık.
4. **Çözünürlük ve en-boy oranı:** 448 kare sıkıştırma; literatürdeki gerçek zamanlı ağlar oranı koruyarak ve daha yüksek çözünürlükte çalışıyor.
5. **Donmuş küçük çekirdekten fazla şey bekledik:** Donmuş özellikle ince sınıf ayırımı büyük temel modellerde işe yarıyor; bizim çekirdekte yan ağ kapasitesi gerekiyordu (ARCH 5 bunu kısmen yaptı).

## 6. Önerilen yol (makale ile uyumlu)

1. **Öğretmenler:** iç mekan COCO-panoptic Mask2Former (Swin-L/T), dış mekan Cityscapes Mask2Former + BDD gerçek etiketleri.
2. **Öğrenci:** PEM (veya YOSO) mimarisi, **bizim eğittiğimiz ağırlıklar**, çekirdeğin ara özelliklerini de girdi olarak alan side-network şeklinde ya da kendi küçük omurgasıyla; oranı koruyan giriş.
3. **Damıtma tarifi:** görev kaybı (Hungarian/maske) + CWD (logit) + ilişkisel KD + öğretmen piksel-çözücü özellik damıtması + BPKD kenar/gövde ayrımı.
4. **Sistem:** sahne yönlendiricisi modül seçer (mevcut), her modül Orin'de TRT ile; hedef ≥ 0.67 ve 30 FPS.
5. **Tarama genişletme:** MCP araçları yeni oturumda yüklendiğinde (claude mcp list: arxiv + semantic-scholar bağlı) bu rapor atıf grafiği ve yıl bazlı hacimle genişletilmeli; özellikle "multi-teacher distillation segmentation" ve "edge MaskFormer TensorRT".

## Kaynaklar (arXiv kimlikleri; venue/atıf OpenAlex, 2026-11-23)
- Liu et al., Structured Knowledge Distillation for Dense Prediction — arXiv:1903.04197 (TPAMI 2020, 247)
- Shu et al., Channel-wise Knowledge Distillation for Dense Prediction — arXiv:2011.13256 (ICCV 2021, 682)
- Yang et al., Cross-Image Relational Knowledge Distillation for Semantic Segmentation — arXiv:2204.06986 (CVPR 2022, 421)
- Wang et al., BPKD: Boundary Privileged Knowledge Distillation — arXiv:2306.08075 (WACV 2024, 76)
- Huang et al., Masked Distillation with Receptive Tokens — arXiv:2205.14589 (52)
- Liu et al., TransKD — arXiv:2202.13393 (T-ITS 2024, 104)
- Xu et al., PIDNet (CVPR 2023, 1108)
- Peng et al., PP-LiteSeg — arXiv:2204.02681 (349)
- Hu et al., YOSO — arXiv:2303.14651 (CVPR 2023, 109)
- Cavagnero et al., PEM — arXiv:2402.19422 (CVPR 2024)
- Xie et al., SegFormer (NeurIPS 2021, 11.899)
- Cheng et al., Mask2Former (CVPR 2022, 6.037)
- Chen et al., ViT-Adapter — arXiv:2205.08534 (946)
- Sung et al., LST — arXiv:2206.06522
- Cai et al., EfficientViT (multi-scale linear attention) — arXiv:2205.14756
- Oquab et al., DINOv2 — arXiv:2304.07193 (TMLR 2024, 11.885)
- Zhang et al., TopFormer (CVPR 2022, 473); Dong et al., AFFormer (AAAI 2023, 174)
