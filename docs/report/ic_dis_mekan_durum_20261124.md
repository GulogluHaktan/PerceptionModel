# İç ve dış mekan segmentasyon modülleri — durum (2026-11-24)

## Sistem
- Donmuş çekirdek (FastViT-SA24, release_v4): kapı, derinlik, 11 sınıflı segmentasyon — her karede.
- Sahne yönlendiricisi (çekirdeğin özet vektörü üzerinde lojistik regresyon, test %97,8): iç mekan → iç mekan modülü, dış mekan → dış mekan modülü.
- Modüller her 10 karede bir çalışır (sınıf haritası ~3 Hz), geri kalan her şey 30 FPS.
- Jetson Orin NX (MAXN, TensorRT 10.16 FP16): **iç mekan 30,6 FPS, dış mekan 30,6 FPS**; kapılar 5 başlıkta değişmedi (120/120).

## İç mekan modülü (22 sınıf)
Sınıflar: other, wall, floor, ceiling, door, window, person, chair, sofa, table, cabinet_shelf, bed, monitor_tv, laptop, keyboard, mouse, phone_remote, cup_bottle, book_paper, plant, bag, appliance.

Model: Swin-T Mask2Former — herkese açık COCO-panoptic ağırlıklarıyla başlatıldı, **bizim** 22 sınıflı taksonomimize ~56 bin gerçek etiketli COCO + öğretmen etiketli SUN/ADE ile ince ayar yapıldı, Swin-L öğretmenden damıtıldı.

| Gerçek COCO iç mekan testi (1000 görüntü, 448 px) | mIoU | monitör | laptop | klavye | mouse | sandalye | insan |
|---|---|---|---|---|---|---|---|
| Eğitimsiz başlangıç | 0.687 | 0.78 | 0.70 | 0.62 | 0.79 | 0.60 | 0.90 |
| **Bizim öğrenci (TensorRT FP16)** | **0.695** | 0.79 | 0.72 | 0.65 | 0.80 | 0.60 | 0.90 |
| Öğretmen Swin-L (çok ölçek + çevirme, 800 px) | 0.757 | 0.85 | 0.75 | 0.76 | 0.91 | 0.60 | 0.92 |

Ölçüt ≥ 0.67 → geçildi. Önceki kendi öğrencilerimiz (donmuş çekirdek + başlık): 0.27 → 0.64; görüntüde dağınıktı.
Orin çıktısı: `deploy/orin_results/20261123/orin_m2f_ours_cekim1.jpg` (atölye ofis çekimi).

## Dış mekan modülü (19 sınıf, Cityscapes)
Sınıflar: road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

Model: Swin-T Mask2Former — herkese açık Cityscapes ağırlıklarıyla başlatıldı, BDD100K gerçek 19 sınıflı etiketleriyle (6.7k) **bizim** ince ayarımız.

| Test (448 px) | BDD 19 sınıf | Gerçek gece |
|---|---|---|
| Donmuş çekirdek + başlık (ilk deneme) | 0.479 | 0.373 |
| Hazır Swin-T Cityscapes (1024 px) | 0.527 | 0.259 |
| Hazır Swin-L Cityscapes (öğretmen, 1024 px) | 0.603 | 0.394 |
| **Bizim öğrenci (TensorRT FP16)** | **0.613** | **0.471** |

Ölçütler ≥ 0.60 / ≥ 0.40 → geçildi. Sınıf örnekleri (BDD): yol 0.96, araba 0.89, otobüs 0.72, bisiklet 0.67, insan 0.68, sürücü 0.54, trafik ışığı 0.56, tabela 0.53, direk 0.47.
Orin çıktısı: `deploy/orin_results/20261123/orin_outdoor_student.jpg` (BDD + gerçek gece kareleri).

## Açık işler
- IMX219 kamera testi (kamera gelmedi). Paket hazır: `run.py --source csi --backend trt --bundle bundle`.
- Makale: büyük modeller öğretmen, öğrenciler bizim; COCO testinde ince ayar katkısı küçük (+0,8), alan uyumu ve sistem tasarımı asıl katkı. Literatür: `docs/literatur/kenar-segmentasyon-damitma-neyi-yanlis-yapiyoruz.md`.
- İç mekanda sandalye (0.60) ve kapı (0.41) en zayıf sınıflar; dış mekanda motosiklet (0.46), çit (0.43), direk (0.47).
