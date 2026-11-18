"""Turkish PDF report, third edition (2026-11-17): v2 edition plus Updates 70-82 (K=8 decoding, size levels, release v2 and v3, fair sheet, side-view test).
Plain language, one page summary first, every number with its test conditions. Numbers are copied by hand from docs/handoff/perceptionmodel-sim-plan.md (Updates 44-82).
Usage: python scripts/make_report_v3_tr.py -> docs/report/PerceptionModel_rapor_v3.pdf (+ .html, charts in docs/report/)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import weasyprint
D = Path("docs/report"); V = Path("runs/eval/vis").resolve(); INK, MUTED, GRID, BLUE, ORANGE, GREY = "#111111", "#52514e", "#e4e3df", "#2a6fdb", "#d9822b", "#8a8f98"


def bars(path, rows, xlabel, width=7.6):
    """One measure over ordered steps: horizontal bars, one hue, value at the bar end."""
    fig, ax = plt.subplots(figsize=(width, 0.5 * len(rows) + 0.9), dpi=200); y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] * 100 for r in rows], height=0.55, color=BLUE)
    for yy, r in zip(y, rows): ax.text(r[1] * 100 + 1.5, yy, f"%{round(r[1] * 100)}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK); ax.set_xlim(0, 110); ax.set_xticks([0, 25, 50, 75, 100]); ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); [s.set_visible(False) for s in ax.spines.values()]; ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def grouped(path, rows, names, cols, xlabel, width=7.6):
    """A few series per condition: grouped horizontal bars with a legend and the value at each bar end."""
    fig, ax = plt.subplots(figsize=(width, 0.62 * len(rows) + 1.2), dpi=200); n = len(names); h = 0.8 / n
    for i, (nm, c) in enumerate(zip(names, cols)):
        ys = [len(rows) - 1 - k + (n / 2 - i - 0.5) * h for k in range(len(rows))]; vals = [r[1 + i] for r in rows]
        ax.barh(ys, [0 if v is None else v * 100 for v in vals], height=h * 0.9, color=c, label=nm)
        for yy, v in zip(ys, vals):
            if v is not None: ax.text(v * 100 + 1.2, yy, f"%{round(v * 100)}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK); ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); [s.set_visible(False) for s in ax.spines.values()]; ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.9 / (0.62 * len(rows) + 1.2)), ncol=n, frameon=False, fontsize=8.5); fig.tight_layout(); fig.savefig(path); plt.close(fig)


bars(D / "v2_journey.png", [("Yalnızca simülasyon", 0.0), ("+ tek salondan gerçek kareler", 0.42), ("+ başka yerlerden gerçek kapılar", 0.51), ("+ genel gerçek veri, büyük tarif", 0.48),
                              ("+ yeni kapı başlığı, veri artırma, kapısız örnekler", 0.94), ("+ ince ızgara", 0.98), ("+ yol verisi, etiket ve derinlik düzeltmeleri", 0.99), ("Sürüm 1 (küçük model, iki tohum ortalaması)", 0.96)],
     "Bulunan kapı payı: hiç görülmemiş gerçek yer, köşe hatası 20 piksel içinde")
grouped(D / "v2_loop.png", [("Temiz görüntü, 8 m/s", 40 / 132, 87 / 88, 1.0), ("Temiz görüntü, 12 m/s", 42 / 132, 87 / 88, 1.0), ("Harita hatası 2 m, 12 m/s", 10 / 132, 87 / 88, 1.0), ("Karanlık + bulanık, 12 m/s", 42 / 132, 88 / 88, 1.0),
                            ("Temiz görüntü, 20 m/s", 39 / 132, 47 / 88, 66 / 132), ("Temiz görüntü, 30 m/s", 26 / 132, 23 / 88, 45 / 132)],
        ["Yalnızca kaba harita", "Kaba harita + model", "Gerçek kapı konumları (üst sınır)"], [ORANGE, BLUE, GREY], "Geçilen kapı payı (simülasyon, 6 parkur, 44 kapı)")
grouped(D / "v2_size.png", [("8 pikselden küçük (çok uzak)", 0.38, None), ("8–16 piksel", 0.58, None), ("16–40 piksel", 0.92, None), ("40–100 piksel", 0.96, None), ("100 pikselden büyük (çok yakın)", 0.40, 0.69)],
        ["20 piksel içinde", "Kapı boyutunun çeyreği içinde"], [BLUE, GREY], "Gerçek uçuş videosunda bulunan kapı payı, kapının görüntüdeki boyutuna göre")


def table(head, rows, cls=""):
    return f"<table class='{cls}'><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


html = f"""<html><head><meta charset='utf-8'><style>
@page {{ size: A4; margin: 17mm 16mm 18mm 16mm; @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #777; }} @bottom-left {{ content: "PerceptionModel · sürüm 3 raporu"; font-size: 8pt; color: #999; }} }}
body {{ font-family: 'DejaVu Sans', sans-serif; font-size: 9.7pt; line-height: 1.47; color: {INK}; }}
h1 {{ font-size: 21pt; margin: 0 0 1mm 0; }} h2 {{ font-size: 13.5pt; margin: 8mm 0 2.5mm 0; border-bottom: 1.2px solid #999; padding-bottom: 1mm; break-after: avoid; }}
h3 {{ font-size: 10.8pt; margin: 4.5mm 0 1.5mm 0; break-after: avoid; }} p {{ margin: 0 0 2.2mm 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 3mm 0; font-size: 8.9pt; break-inside: avoid; }} th, td {{ border: 1px solid #c8c8c8; padding: 1.5mm 2mm; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; }} td:not(:first-child), th:not(:first-child) {{ text-align: center; }} table.txt td:not(:first-child), table.txt th:not(:first-child) {{ text-align: left; }}
.box {{ background: #f3f6fa; border-left: 3px solid {BLUE}; padding: 3mm 4mm; margin: 3mm 0; break-inside: avoid; }} .warn {{ border-left-color: #c98500; background: #fbf6e9; }}
.small {{ font-size: 8.4pt; color: #444; }} img {{ max-width: 100%; }} li {{ margin-bottom: 1.1mm; }} ul {{ margin: 1mm 0 2.5mm 0; padding-left: 5mm; }}
.kpi {{ width: 100%; border: none; margin: 3mm 0; }} .kpi td {{ border: none; background: #f3f6fa; padding: 3mm 3mm; text-align: left !important; width: 25%; border-right: 2mm solid white; }}
.kpi b {{ font-size: 15pt; display: block; }} .kpi span {{ font-size: 8.2pt; color: #444; }} .pb {{ break-before: page; }} figure {{ margin: 2mm 0 3mm 0; break-inside: avoid; }} figcaption {{ font-size: 8.3pt; color: #444; margin-top: 1mm; }}
</style></head><body>
<h1>PerceptionModel: sürüm 3 raporu</h1>
<p class='small'>17 Kasım 2026 · üçüncü baskı · RTX 5090 üzerinde yapılan eğitim ve ölçümler · ayrıntılı deney günlüğü: docs/handoff/perceptionmodel-sim-plan.md (Update 24–82) · model dosyaları: runs/release_v3/ (v1 ve v2 de yerinde)</p>

<div class='box'><b>Bir cümleyle.</b> Tek kamera görüntüsünden yarış kapılarını, sahnedeki nesne sınıflarını ve derinliği çıkaran küçük model bu baskıda iki kez iyileşti ve sürüm 3 ("bigR_h_s1") donduruldu:
eğitimde hiç görmediği gerçek bir yerde tam görünen kapıların %99'unu 20 piksel içinde buluyor (sürüm 1: %95–97), üst üste duran iç içe kapıları da buluyor, kapısız görüntülerde yanlış kapı üretmiyor ve simülasyonda kapalı döngüde kapıların 175/176'sından geçiriyor.
Çok uzak ve çok küçük kapılar, yarım kapılar ve kendi kameramızla doğrulama hâlâ açık.</div>

<table class='kpi'><tr>
<td><b>%99</b><span>tam görünen kapı, hiç görülmemiş gerçek yer, 20 piksel içinde (iki tohum: 0.987 ve 0.990)</span></td>
<td><b>20 / 20</b><span>iç içe kapı bulundu, aynı yer (sürüm 2: 12 / 20)</span></td>
<td><b>175 / 176</b><span>kapı geçildi, kapalı döngü SİMÜLASYON (gerçek başarı oranı değildir)</span></td>
<td style='border-right:none'><b>0.9 ms</b><span>RTX 5090'da ağ süresi (TensorRT fp16); Orin'de bu ağırlıklar ölçülmedi</span></td></tr></table>

<h3>İyi olanlar</h3>
<ul><li>Tam görünen kapılar, görülmemiş gerçek bir yerde: sürüm 3'te 20 piksel içinde 0.99, 10 piksel içinde 0.94 (sürüm 1: 0.95–0.97 ve 0.88–0.91). Fark tek tohumda ±0.06 gürültü bandının içinde; iki tohumda aynı yönde ve nedeni biliniyor (iç içe kapılar).</li>
<li>Kareden kareye kararlılık: gerçek uçuş videosunda köşe titremesi ortanca 0.8 piksel, tek karelik kopma %1'in altında.</li>
<li>Simülasyonda döngüde: 1–2 m hatalı bir kapı haritası tek başına kapıların %8–32'sinden geçirirken, model eklenince %99–100.</li>
<li>Genel algı: hiç veri görülmeyen yollarda sınıf haritası kullanılabilir düzeyde (ortalama kesişim oranı 0.61), derinlik hiç görülmeyen araç kamerasında %8 hata.</li></ul>
<h3>Zayıf olanlar</h3>
<ul><li>Gerçek videoda çok yakın kapılar (100 pikselden büyük): sürüm 1'de 0.40, sürüm 3'te 0.79–0.84 (iyileşti ama hâlâ en zayıf aralıklardan). Çok uzak kapılar: 8–16 piksel 0.87, 8 pikselden küçük yalnızca ~0.2–0.4 (55 örnek).</li>
<li>Yalnızca bir köşesi ya da bir çerçeve parçası görünen "yarım" kapılar: taklit ölçümde bir köşe görünürken 0.64–0.73, yalnızca çerçeve parçası görünürken 0.15–0.27; gerçek yakın geçişte doğrulanmadı.</li>
<li>Kendi kameramızla (IMX219) tek bir kare bile görülmedi; simülasyon 130° kamerayla üretildi, lens farklıysa sonuçlar değişir.</li>
<li>20 m/s ve üstünde kapalı döngü testi algıyı ölçemiyor (basit kontrolcü sınırlıyor).</li></ul>
<div class='box warn'><b>Sayıları okurken.</b> Kapı sayıları tek bir görülmemiş yerde (196 görüntü, 312 kapı) ölçüldü ve aynı tarifle yeniden eğitimde ±0.06 oynuyor. Simülasyon sonuçları "başarı oranı" değildir.
TII uçuş videoları eğitimde görülen salonda çekildi; oradaki sayılar yeni bir yere genelleme göstermez.</div>

<h2 class='pb'>1. Model ne yapıyor?</h2>
<p>Tek bir kamera görüntüsünden üç çıktı üretir: <b>(1)</b> en çok 8 yarış kapısının (sürüm 1'de 4) dört köşesi (köşe görüntünün dışında kalsa bile tahmin edilir), <b>(2)</b> her pikselin sınıfı
(zemin/yol, bina, ağaç ve bitki örtüsü, insan, araç, kaya, şerit çizgisi, rakip drone, diğer engel, kapı; gökyüzü "arka plan"), <b>(3)</b> göreli derinlik. Büyük hazır modellerden
(DINOv2, Depth Anything v2) bilgi damıtılarak ve etiketli verilerle eğitilir.</p>
{table(["", "Sürüm 1: küçük model (bigL_h)", "Büyük model (bigL)"], [["Omurga", "FastViT-SA24, 11 milyon parametre", "ViT-B, 92 milyon parametre"], ["Amaç", "Drone üzerindeki Jetson Orin", "Karşılaştırma ve masaüstü"],
 ["RTX 5090, yalnızca ağ (TensorRT fp16)", "0.71 ms", "1.16 ms"], ["Jetson Orin NX 8 GB, 448 piksel, MAXN, yalnızca ağ *", "16.7 ms (60 kare/sn)", "25.2 ms (40 kare/sn)"], ["Dosyalar", "runs/release_v1/", "runs/bigL/vit/"]], "txt")}
<p class='small'>* Orin ölçümü 10 Kasım'da aynı mimariyle fakat rastgele ağırlıklarla yapıldı (docs/ORIN_RESULTS.md); kamera okuma ve ön/son işlem süresi ile eğitilmiş ağırlıkların Orin'deki doğruluğu ölçülmedi.
Önceki raporda "Orin'de hiç ölçüm yok" yazıyordu; bu yanlıştı, hız ölçümü vardı.</p>

<h2>2. Kapı bulma: hiç görülmemiş gerçek yer</h2>
<p>MonoRaceGate veri kümesinin "red" adlı yeri eğitime hiç konmadı. Aşağıdaki grafik, o yerde 20 piksel hata içinde bulunan kapı payının süreç boyunca nasıl değiştiğini gösterir.</p>
<figure><img src='{(D / "v2_journey.png").resolve()}' style='width:84%'></figure>
<p>Asıl sıçrama üç değişiklikle geldi: kapı başlığının yeniden tasarlanması (her kapı için merkez noktası + köşe uzaklıkları), veri artırma ve kapısız gerçek görüntülerin "burada kapı yok" örneği olarak kullanılması.
Sonraki adımlar kapı sayısını değil, modelin diğer yeteneklerini geliştirdi; kapı sayısı 0.95–0.99 aralığında tohumdan tohuma oynuyor.</p>
{table(["Ölçüt (sürüm 1, küçük model)", "Tohum 0", "Tohum 1"], [["Tam görünen kapı, 20 piksel içinde", "0.971", "0.952"], ["Tam görünen kapı, 10 piksel içinde", "0.913", "0.881"], ["Köşe hatası (ortanca, 256×192 ölçeğinde)", "2.5 piksel", "2.6 piksel"],
 ["Yanlış kapı: kapısız yol ve dış mekân görüntüleri (600 kare)", "0.00", "0.00"]])}
<figure class='pb'><img src='{V / "real_gates_compare.jpg"}' style='height:225mm; width:auto; display:block; margin:0 auto'><figcaption>Önce / sonra: aynı gerçek karelerde yalnızca simülasyonla eğitilmiş model ve gerçek veriyle eğitilmiş model.</figcaption></figure>

<h2>2b. Bu baskıda ne değişti: sürüm 2 ve sürüm 3</h2>
<p>Sürüm 1'den sonra, kayıp kapılara tek tek bakıp nedenlerini aradık. Üç bulgu çıktı:</p>
<ul><li><b>Kapı sınırı (sürüm 2).</b> Çözücü kare başına yalnızca 4 kapı bırakıyordu; 4'ten fazla kapılı karelerde en küçük (en uzak) kapılar kesiliyordu. Önceki raporlarda "uzak kapıyı bulamıyor" diye yorumladığımız kaybın büyük kısmı buydu. Sınırı 8'e çıkarınca (yeniden eğitim gerekmedi) gerçek videoda 8–16 piksellik kapılarda bulma 0.64'ten 0.84'e çıktı. Önceki yorumum (çözünürlük ya da eğitim yetersizliği) yanlıştı ve düzeltildi.</li>
<li><b>Kısmen görünen kapıda merkez tanımı (sürüm 2).</b> Kısmen görünen kapının merkezi, kapının tamamlanmış karenin merkezine konunca yakın kapılarda kazanç geldi (kısmen görünen kapılarda, gerçek video: 0.914 → 0.943).</li>
<li><b>İç içe kapılar (sürüm 3).</b> Bir kapının içinden başka bir kapı görününce ikisinin merkezi aynı hücreye düşüyor ve biri kayboluyordu; görülmemiş yerde kalan kaçırmaların 10'dan 8'i buydu (iç içe 20 kapıdan yalnızca 12'si bulunuyordu). Çözüm: iki boyut düzeyli kapı başlığı (küçük ve büyük kapılar ayrı haritalarda) ve yakın kapıların eğitimde 6 kat sık görülmesi.</li></ul>
{table(['Hiç görülmemiş yer (196 kare, 312 kapı)', 'Sürüm 1', 'Sürüm 2', 'Sürüm 3 (iki tohum)'], [['Tam görünen kapı, 20 piksel içinde', '0.95–0.97', '0.974', '0.987 / 0.990'], ['10 piksel içinde', '0.88–0.91', '0.913', '0.942 / 0.942'], ['İç içe kapı (20), 20 piksel içinde', '–', '0.60', '1.00 / 1.00'], ['Gerçek video (salon eğitimde görüldü): kapı-kare başına', '0.904', '0.946', '0.953 / 0.951'], ['Gerçek video: 100 pikselden büyük kapı', '0.40', '0.54', '0.84 / 0.79'], ['Kapalı döngü SİMÜLASYON: geçilen kapı', '174/176', '175/176', '175/176'], ['RTX 5090, ağ süresi (TensorRT fp16)', '0.71 ms', '0.81 ms', '0.9 ms']])}
<p class='small'>Tek tohum sonuçlarında ±0.06 gürültü var. Sürüm 3'ün kazançlarının her biri bu bandın içinde; ama iki tohum aynı yönde çıktı ve iç içe kapılardaki 0.60 → 1.00 değişimi bilinen bir nedene bağlı. "Başarı oranı" değil, tek bir görülmemiş yerdeki ölçümdür.</p>
<figure class='pb'><img src='{V / "fair_gates_red_v3.jpg"}' style='height:225mm; width:auto; display:block; margin:0 auto'><figcaption>Sürüm 3, hiç görülmemiş yerden eşit aralıkla seçilmiş (seçmeden) 12 tam kare. Beyaz noktalar etiket köşeleri, yeşil çerçeveler model çıktısı. Önceki "kırpma kolajı" özenle seçilmiş ve 5 kat büyütülmüştü; bu görsel öyle değil, kamera eğriliği ve eğik bakış olduğu gibi görünüyor.</figcaption></figure>
<h3>Yandan / eğik bakış</h3>
<p>Kapıya karşıdan değil açıyla bakınca (kapı görüntüde dar görünür) başarı düşüyor mu diye ölçtük. Görülmemiş yerde kapıların çoğu (%78) hafif ya da orta açıdan, yalnızca 10 kapı dik yandan görünüyor; bu yüzden sonuç zayıf istatistik.
Dik yan görünüm (en/boy oranı 0.5'ten küçük, 10 kapı): sürüm 3'te 10 piksel içinde 0.70, ortanca hata 5.3 piksel (sürüm 2: 0.60 ve 9.3 piksel). Kapılar hafif ya da orta açıdan baktığında 0.99. Eğitim verisinde dik yan görünüm az (gerçek veride %7, simülasyonda %0.1).
Eğitime yan görünüm üreten bir veri artırma (kareyi yana sıkıştırıp perspektif eğriltme) ekleyip denedik: görülmemiş yerde ölçülebilir kazanç yok (dik yanda 10 piksel içinde 0.70, değişmedi; iç içe kapılarda 1.00 → 0.95). Kullanılmıyor. Asıl çözüm veridir: simülasyonda dik açılı kareler ya da kendi kameramızdan görüntü.</p>
<h2 class='pb'>3. Gerçek uçuş videosunda davranış</h2>
<p>Tek tek karelere bakmak uçuşta önemli olanı gizler: kapı kareden kareye kayboluyor mu, köşeler titriyor mu? Bunu TII veri kümesinin eğitimde kullanılmayan üç uçuşunda ölçtük
(saniyede yaklaşık 120 kare, 16.132 kare, 34.592 kapı-kare). <b>Salon eğitimde görülen salondur;</b> bu bölüm yeni yere genellemeyi değil, zaman içindeki davranışı ölçer.</p>
{table(["Ölçüt", "Sürüm 1 (bigL_h)", "Büyük model (bigL)"], [["Kapı-kare başına bulma (20 piksel içinde)", "0.903", "0.900"], ["Tek karelik kopma (1000 kapı-karede)", "6.7", "9.5"],
 ["100 milisaniyeden uzun kayıplardaki kapı-kare payı", "%4.0", "%3.0"], ["Son 3 kareyi tutan basit izleyiciyle bulma", "0.937", "0.940"], ["Köşe titremesi, ortanca / %90'lık", "0.84 / 3.0 piksel", "0.94 / 3.4 piksel"],
 ["Kapı görüşten çıkmadan önceki son 100 milisaniyede bulma", "0.80", "0.77"]])}
<figure><img src='{(D / "v2_size.png").resolve()}' style='width:92%'><figcaption>Sürüm 1, kapının görüntüdeki boyutuna göre (256×192 ölçeği). Orta boy kapılar çok iyi; iki uç zayıf.</figcaption></figure>
<p><b>Okuma.</b> Model kareden kareye sakin ve küçük modelin titremesi büyük modelden az. Basit bir izleyici (son tespiti birkaç kare tutmak) bulma oranını 0.90'dan 0.94'e çıkarıyor; uçuş yazılımında böyle bir izleyici olmalı.
İki gerçek zayıflık var: çok uzak ve çok yakın kapılar. Yakın kapı için eğitimde yapılan "yarım kapı" çalışması bu videoda neredeyse fark yaratmadı (boyuta göreli ölçütle 0.66'dan 0.69'a).</p>

<h2>4. Yarım kapılar (taklit ölçüm)</h2>
<p>Yalnızca bir köşesi ya da bir çerçeve parçası görünen kapılar eğitim verisinde neredeyse hiç yoktu. Tam görünen kapılı kareler eğitim sırasında kırpılarak bu durum üretildi; ölçüm de görülmemiş yerdeki kapılar aynı yolla kırpılarak yapıldı.
"Doğru yere kondu": dört köşenin (görünmeyenler dahil) ortalama hatası kapı boyutunun çeyreğinden az.</p>
{table(["Ölçüt", "Önceki küçük model", "Sürüm 1 (iki tohum)"], [["Bir köşe görünüyor (380 örnek): doğru yere kondu", "0.32", "0.71 – 0.77"], ["Bir köşe görünüyor: hiç tahmin yok", "0.45", "0.07 – 0.08"],
 ["Hiç köşe yok, yalnızca çerçeve parçası (99 örnek): doğru yere kondu", "0.04", "0.27"], ["Hiç köşe yok: hiç tahmin yok", "0.71", "0.41 – 0.43"]])}
<div class='box warn'><b>Dikkat.</b> Bu, uzaktaki bir kapının kırpılmasıyla yapılan bir taklittir. Gerçek uçuş videosundaki yakın kapılarda (bölüm 3) aynı kazanç görülmedi; "yakın geçiş çözüldü" denemez.
Gerçek yakın geçiş görüntüsü (kendi kameramızdan) gelince yeniden ölçülmeli.</div>

<h2 class='pb'>5. Kapalı döngü simülasyon: algı, geçişi bozuyor mu?</h2>
<p>Model bir simülatörde basit bir drone'a bağlandı: her adımda kameradan görüntü çizilir, model kapı köşelerini bulur, köşelerden kapının metre cinsinden konumu hesaplanır ve drone o konuma yönelir.
Drone'un elinde kaba bir kapı haritası vardır (her kapı 1–2 m hatalı); model bu haritayı düzeltir. Üç durum karşılaştırıldı: yalnızca harita, harita + model, gerçek kapı konumları (kontrolcünün üst sınırı).</p>
<figure><img src='{(D / "v2_loop.png").resolve()}' style='width:92%'></figure>
<div class='box warn'><b>Bu bir simülasyon sonucudur, gerçek başarı oranı değildir.</b> Sahneler eğitim verisini üreten çiziciyle çizildi (parkur yerleşimleri eğitimde yok), drone bir nokta kütledir, kendi konumunu tam bilir,
rüzgâr ve satır satır okuma (rolling shutter) yoktur, kamera 130°'dir.</div>
<ul><li><b>12 m/s'ye kadar</b> model döngüdeyken kapıların %99–100'ü geçildi (48 uçuşta 352 kapının 349'u); 2 m harita hatası ve karanlık + bulanık görüntü dahil. Bir kare gecikme (yaklaşık 17 ms) fark yaratmadı.</li>
<li><b>20 ve 30 m/s'de</b> test algı hakkında bir şey söylemiyor: kapı konumları tam bilinse bile kapıların %50'si ve %34'ü geçiliyor, çünkü kontrolcü sabit hızla uçuyor ve dönüşten önce yavaşlamıyor.
30 m/s'de modelin kapı konum hatası da büyüyor (geçiş anında 0.6 m; 12 m/s'de 0.15 m).</li>
<li><b>Mesafeye göre konum hatası</b> (simülasyon): 3 m içinde 7 cm, 6–12 m'de 0.2 m, 12–25 m'de 0.8 m, 25 m'den uzakta kullanılamaz. Uzak kapılar yalnızca yön bilgisi olarak kullanılmalı.</li></ul>
<figure><img src='{V / "closed_loop_c_race_0.jpg"}' style='width:100%'><figcaption>Bir parkurda üç uçuşun kuşbakışı yolu, tüm koşulların özeti ve modelin uçuş sırasında gördüğü kareler (yeşil: bulunan kapılar).</figcaption></figure>

<h2 class='pb'>6. Genel algı: sınıf haritası ve derinlik</h2>
{table(["Test (kareler eğitimde yok)", "Ölçüt", "Sürüm 1 (bigL_h)", "Büyük model (bigL)"], [["Kötü havada Hindistan yolları (IDD-AW); bu kaynaktan hiç eğitim verisi yok", "Ortalama kesişim oranı", "0.61", "0.58"],
 ["Araç kamerası yol (BDD); aynı kaynaktan eğitim verisi var", "Ortalama kesişim oranı", "0.73", "0.74"], ["Arazi / orman yolu (RUGD), ayrılmış 4 dizi", "Ortalama kesişim oranı", "0.51", "0.51"],
 ["Yüksekten drone bakışı şehir (UAVid)", "Ortalama kesişim oranı", "0.59", "0.62"], ["Otoyol şerit çizgisi (TuSimple, gerçek etiket)", "Yakalama", "0.78 – 0.80", "0.83"],
 ["Araç kamerası derinlik (KITTI); hiç eğitim verisi yok", "Göreli hata (düşük iyi)", "0.08", "0.08"], ["Yaya pikselleri (IDD-AW): insan / araç olarak işaretlenen", "Pay", "0.69 / 0.19", "0.63 / 0.18"],
 ["Motosiklet pikselleri (IDD-AW): araç / insan olarak işaretlenen", "Pay", "0.63 / 0.23", "0.61 / 0.19"]], "txt")}
<p><b>Bu bölümde yapılan düzeltmeler.</b></p>
<ul><li><b>Gökyüzü ve bitki örtüsü:</b> eski modeller gökyüzünü rastgele bir sınıfa boyuyordu ve eski puanlar bu pikselleri saymadığı için durumu olduğundan iyi gösteriyordu. Gökyüzü "arka plan", bitki örtüsü "ağaç" olarak etiketlendi;
hiç veri görülmeyen yollarda piksel doğruluğu 0.43'ten 0.87'ye çıktı. Bu tablodaki sayılar gökyüzünü ve bitki örtüsünü sayar.</li>
<li><b>Şehir içi yol boyası:</b> etiketli gerçek veri bulunamadı; yol üstündeki parlak boya otomatik bir kuralla etiketlendi. Model artık şehir içi yolda boyayı kısmen işaretliyor, ama bunu ölçecek bağımsız etiketli test yok.
Bedeli: düz yolun yaklaşık %2'si yanlışlıkla şerit sayılıyor.</li>
<li><b>Motosiklet:</b> motosikletler insan olarak boyanıyordu. Motosiklet ve bisiklet "araç" sınıfına alındı. İnsan sınıfının ağırlığı bir ödünleşim: yayalar öncelikli tutuldu, bu yüzden motosikletin bir kısmı (%23) hâlâ insan olarak işaretleniyor.</li></ul>
<p><b>Zayıf sınıflar:</b> hiç görülmemiş yollarda "diğer engel" (0.05–0.10), yüksekten bakışta insan (0.05–0.11; kare parçalara bölünürse 0.28), arazide kaya (0.08–0.16).</p>
<figure class='pb'><img src='{V / "model_sheet_bigL_h_hybrid.jpg"}' style='height:228mm; width:auto; display:block; margin:0 auto'><figcaption>Sürüm 1'in eğitimde kullanılmayan gerçek görüntülerdeki çıktısı: solda kamera ve bulunan kapılar (yeşil), ortada sınıf haritası, sağda göreli derinlik (kırmızı = yakın).</figcaption></figure>

<h2 class='pb'>7. Bilinen sınırlar</h2>
<ul><li><b>Kendi kameramız görülmedi.</b> Listedeki kamera IMX219; lens açısı bilinmiyor (standart sürüm yaklaşık 62°, simülasyon 130°). Bu sensör kareyi satır satır okur; hızlı uçuşta kapı eğik görünür ve eğitimde bu yok.</li>
<li><b>Tek test yeri.</b> Kapı genellemesi tek bir görülmemiş yerde ölçüldü; sayılar tohumdan tohuma ±0.06 oynuyor.</li>
<li><b>Yakın ve uzak kapılar</b> gerçek videoda zayıf (bölüm 3). Gerçek eğitim verisinde 0–1 köşesi görünen kapılar etiketsiz ve "kapı yok" sayılıyor.</li>
<li><b>Orin'de</b> sürüm 2 ve 3 ağırlıklarının hızı ve doğruluğu, kamera ve ön/son işlem süresi ölçülmedi.</li>
<li><b>Bulanıklık, sis, karanlık</b> yalnızca simülasyon efektleriyle sınandı; en zor karışımda (30 m/s + karanlık + sis) simülasyonda bile kapı bulma yaklaşık %60.</li>
<li><b>Kapalı döngü testi</b> basit: nokta kütle, kusursuz konum bilgisi, basit kontrolcü. 20 m/s üstü için daha iyi bir kontrolcü gerekiyor.</li>
<li><b>Veri lisansları:</b> BDD100K, RUGD ve IDD-AW yalnızca araştırma / ticari olmayan kullanım; UAVid ve KITTI CC BY-NC-SA. Ticari kullanım düşünülürse bu kaynaklar çıkarılıp yeniden eğitilmeli.</li></ul>

<h2>8. Sıradaki adımlar</h2>
{table(["Adım", "Neye bağlı", "Durum"], [["Dik açılı / yandan görünüm için veri (simülasyonda dik açılı kareler)", "Simülasyon üretimi için GPU", "Sırada"],
 ["Çok küçük (8 pikselden az) ve yalnızca çerçeve parçası görünen kapılar", "Yeni veri ya da daha yüksek giriş çözünürlüğü", "Sırada"], ["Eğitimi kamera geometrisine uydurmak (görüş açısı, satır satır okuma)", "IMX219 lens açısı", "Bilgi bekleniyor"],
 ["Orin'de sürüm 2/3 ağırlıklarının hız ve doğruluk ölçümü", "Orin'e erişim (okulda)", "Bekliyor"], ["Kendi kameradan görüntüyle ölçüm, gerekirse kısa ince ayar; yakın geçişin gerçek görüntüyle doğrulanması", "Kamera", "Bekliyor"],
 ["Yüksek hızda kapalı döngü testi", "Takımın kontrolcüsü ya da daha iyi bir kontrolcü", "Bekliyor"]], "txt")}

<h2>Ek: eğitim verisi</h2>
{table(["Kaynak", "Tür", "Ne için"], [["Simülasyon (kendi üretimimiz)", "32.000 kare; ışık, sis, bulanıklık çeşitlemeleriyle", "Kapı, sınıf haritası, derinlik (kusursuz etiket)"], ["TII Racing", "Gerçek, tek salon, yaklaşık 6.600 kare", "Gerçek kapı köşeleri, kapısız kırpmalar"],
 ["MonoRaceGate (red hariç)", "Gerçek, 3 farklı yer", "Gerçek kapı köşeleri"], ["DIODE, SUN RGB-D", "Gerçek, iç ve dış mekân", "Derinlik, sınıf haritası"], ["ADE20K, BDD100K", "Gerçek, dış mekân ve yol; 3.000 + 7.000 kare", "Sınıf haritası"],
 ["Aeroscapes, UAVid", "Gerçek, drone bakışı", "Sınıf haritası"], ["RUGD", "Gerçek, arazi; 634 kare", "Sınıf haritası"], ["TuSimple", "Gerçek, otoyol; 267 kare", "Şerit çizgisi"]], "txt")}
<p class='small'>Sözlük. <b>Ortalama kesişim oranı (mIoU):</b> her sınıf için tahmin ile etiketin örtüşme oranının ortalaması; 1 kusursuz. <b>Tohum:</b> eğitimin rastgele başlangıcı; aynı tarif farklı tohumla biraz farklı sonuç verir.
<b>Kapı-kare:</b> bir karede görünen bir kapı. <b>Kapalı döngü:</b> modelin çıktısının drone'un hareketini, hareketin de bir sonraki görüntüyü belirlediği test.</p>
</body></html>"""
(D / "PerceptionModel_rapor_v3.html").write_text(html, encoding="utf-8"); weasyprint.HTML(string=html, base_url=str(Path.cwd())).write_pdf(D / "PerceptionModel_rapor_v3.pdf")
print("wrote", D / "PerceptionModel_rapor_v3.pdf", round((D / "PerceptionModel_rapor_v3.pdf").stat().st_size / 1e3), "kB")
