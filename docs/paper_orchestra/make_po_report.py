"""PaperOrchestra run summary (Turkish): what the tool uses, token use, optimisation options, reviewer verdict, problems met.
Usage: .venv/bin/python docs/paper_orchestra/make_po_report.py -> docs/paper_orchestra/PaperOrchestra_ozet.pdf"""
from pathlib import Path
import weasyprint

D = Path(__file__).resolve().parent
html = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>
@page { size: A4; margin: 16mm 15mm; @bottom-right { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1d1d1f; }
h1 { font-size: 17pt; margin: 0 0 2mm; } h2 { font-size: 12.5pt; margin: 6mm 0 2mm; border-bottom: 1.5px solid #2b6cb0; padding-bottom: 1mm; color: #1a365d; }
h3 { font-size: 10.5pt; margin: 4mm 0 1.5mm; } .sub { color: #555; margin-bottom: 4mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 8.8pt; } th, td { border: 1px solid #cbd5e0; padding: 1.4mm 2mm; vertical-align: top; text-align: left; }
th { background: #ebf4ff; } .box { background: #f7fafc; border-left: 3px solid #2b6cb0; padding: 2mm 3mm; margin: 2mm 0; }
.warn { background: #fffaf0; border-left: 3px solid #dd6b20; padding: 2mm 3mm; margin: 2mm 0; } .ok { color: #276749; font-weight: bold; } .bad { color: #c53030; font-weight: bold; }
ul, ol { margin: 1mm 0 2mm; padding-left: 5mm; } li { margin: 0.6mm 0; } code { font-size: 8.6pt; background: #edf2f7; padding: 0 1mm; }
</style></head><body>
<h1>PaperOrchestra koşusu — özet raporu</h1>
<div class="sub">PerceptionModel makalesi • 21 Kasım 2026 • Araç: google-research/paper-orchestra, Claude ile birebir akış</div>

<h2>1. Araç ne kullanıyor? (MCP değil)</h2>
<p>PaperOrchestra bir Claude skill'i ya da MCP sunucusu değil. Google Cloud AI Research'ün yazdığı <b>Python tabanlı çok-ajanlı bir boru hattı</b>.
Ajanlar sırayla çalışır, her biri büyük bir prompt ile LLM API'sini doğrudan çağırır:</p>
<table><tr><th>Aşama</th><th>Ajan / ne yapıyor</th><th>Normalde kullandığı</th><th>Bizim koşuda</th></tr>
<tr><td>1</td><td>Outline: fikir + deney günlüğü + şablon → makale planı, grafik planı, literatür arama planı</td><td>gemini-3.1-pro</td><td>Claude Opus 5.5</td></tr>
<tr><td>2a</td><td>Literatür: her bölüm için arama görevi, 10-15 aday makale bulma (Google Search aracı), sonra Semantic Scholar REST API ile doğrulama, giriş + ilgili çalışmalar yazımı</td><td>gemini-3-flash + Google Search, gemini-pro</td><td>Claude Sonnet 5.5 + WebSearch, Opus</td></tr>
<tr><td>2b</td><td>Plotting: her grafik için planla → stil → kod/şema üret → <b>3 tur eleştirmen</b> (görüntüyü görerek) → başlık yaz</td><td>gemini-pro + gemini-3-pro-image</td><td>Opus (+Read ile görüntü), şemalar matplotlib ile</td></tr>
<tr><td>3</td><td>Section writing: tüm bölümleri tek seferde, grafikleri görerek yaz</td><td>gemini-pro</td><td>Opus</td></tr>
<tr><td>4</td><td>Content refinement: PDF derle → sayfa görüntüleri → <b>3'lü hakem ansamblı</b> puanlar → düzelt → tekrar (en fazla 3 tur)</td><td>gemini-pro</td><td>Opus (+Read)</td></tr></table>
<p>Kendi kodu içinde MCP yok; dış servis olarak yalnızca LLM API'si, Google Search ve Semantic Scholar API'si var. Bizim koşuda Gemini istemcisini
<code>claude -p</code> (Claude Code'un başsız modu) çağıran bir ara katmanla değiştirdik; aracın kodu, promptları, sırası ve tur sayıları aynı kaldı.
<b>Her <code>claude -p</code> çağrısı senin Claude aboneliğinin kotasından düşüyor</b> — bu yüzden 5 saatlik limit iki kez, sonunda haftalık limit doldu.</p>

<h2>2. Bu koşuda ne kadar harcandı?</h2>
<table><tr><th>Çağrı tipi</th><th>Adet</th><th>Gönderilen metin</th><th>Üretilen metin</th><th>Not</th></tr>
<tr><td>Opus, yalnız metin (outline, kod, bölüm yazımı, düzeltme)</td><td>103</td><td>1.61 M karakter</td><td>1.03 M karakter</td><td>en pahalı kalem: uzun Opus çıktıları</td></tr>
<tr><td>Opus + görüntü/PDF okuma (grafik eleştirmeni, hakem)</td><td>113</td><td>3.37 M karakter + görüntüler</td><td>0.53 M karakter</td><td>her çağrı en az 2 model turu (oku + cevapla)</td></tr>
<tr><td>Sonnet + web araması (literatür adayları)</td><td>66</td><td>0.15 M karakter</td><td>0.12 M karakter</td><td>görece ucuz</td></tr>
<tr><td><b>Toplam</b></td><td><b>282</b></td><td><b>5.13 M karakter</b></td><td><b>1.68 M karakter</b></td><td>85 çağrı limit nedeniyle başarısız (hızlı, az maliyet)</td></tr></table>
<p>Kaba tahmin: <b>~2-3 milyon girdi tokenı</b> (görüntüler ve araç turları dahil) ve <b>~0.4 milyon çıktı tokenı</b>, çoğu Opus. Toplam model süresi ≈ 3.5 saat.
Kesin token sayısı ölçülmedi (CLI metin modunda çalıştırıldı); bu karakter sayımından yapılmış bir tahmindir.</p>

<h2>3. Neden bu kadar çok token yiyor?</h2>
<ol>
<li><b>Aynı büyük bağlam her çağrıda baştan gönderiliyor.</b> Fikir + deney günlüğü + LaTeX şablonu + outline + atıf listesi neredeyse her ajana tam metin gidiyor (tek çağrıda 15-30 bin karakter).</li>
<li><b>Eleştirmen döngüleri çarpan etkisi yapıyor.</b> 8 grafik × (plan + stil + üretim + 3 eleştiri turu + yeniden üretim + başlık) ≈ 60-80 çağrı; her eleştiri görüntü okuyor.</li>
<li><b>Hakem ansamblı.</b> Her iyileştirme turunda 3 ayrı hakem, her biri derlenmiş makalenin tüm sayfa görüntülerini okuyor.</li>
<li><b>Tam metin yeniden yazımı.</b> Bölüm yazımı ve her düzeltme turu makalenin <i>tamamını</i> yeniden üretiyor (tek çağrıda ~20 bin karakter çıktı); sadece değişen kısım değil.</li>
<li><b>Hep en büyük model.</b> Varsayılan ayar yazım, eleştiri, başlık, kod — hepsi için "pro" model; bizde Opus. Basit işler (başlık, JSON düzeltme) için de.</li>
<li><b>Başarısızlıkta baştan başlama.</b> Hata olunca araç 3 kez <i>en baştan</i> (outline'dan) deniyor; ara sonuçları kullanmıyor. Biz bunu kendi "kaldığı yerden devam" betiğimizle aştık.</li>
<li><b><code>claude -p</code> katmanının ek yükü.</b> Her çağrı yeni bir oturum: araç tanımları, araç kullanım turları (oku → cevapla) ve önbellek ısınması her seferinde baştan; abonelik kotası API'deki toplu/önbellek indirimlerinden yararlanmıyor.</li>
<li><b>Paralel çalışma.</b> Literatür ve grafik ajanları aynı anda koşuyor; 5 saatlik pencere çok hızlı doluyor.</li></ol>

<h2>4. Optimizasyon yöntemleri</h2>
<table><tr><th>#</th><th>Yöntem</th><th>Nasıl</th><th>Beklenen etki</th></tr>
<tr><td>1</td><td><b>Model kademelendirme</b></td><td>Yazım ve son düzeltme Opus; outline, eleştirmen, başlık, JSON, grafik kodu Sonnet; basit format kontrolleri Haiku</td><td>en büyük kazanç: kota/maliyet yarıdan fazla düşer</td></tr>
<tr><td>2</td><td><b>Tur sayılarını düşürme</b></td><td><code>--plotting_max_critic_rounds 1</code>, hakem ansamblı 3→1, iyileştirme 3→1-2 tur</td><td>grafik + hakem çağrıları ~%60 azalır</td></tr>
<tr><td>3</td><td><b>Grafikleri kendimiz vermek</b></td><td><code>--use_plotting false</code> + <code>raw_materials/figures/</code> içine hazır grafiklerimiz (zaten çoğu bizde var)</td><td>~70 çağrı tamamen gider</td></tr>
<tr><td>4</td><td><b>Bağlamı kısaltma</b></td><td>Deney günlüğünü öz tablolara indirgemek; şablonu ve outline'ı her ajana değil, gerekene göndermek</td><td>girdi tokenı %30-50 azalır</td></tr>
<tr><td>5</td><td><b>Prompt önbelleği (API ile)</b></td><td>Anthropic API'de sabit ön ek (fikir, günlük, şablon) cache'lenir; önbellekten okuma temel girdi fiyatının ~0.1 katı</td><td>tekrarlayan girdide ~%90 indirim</td></tr>
<tr><td>6</td><td><b>Batch API</b></td><td>Acele gerektirmeyen çağrılar (eleştiri, hakem) Message Batches ile: girdi ve çıktıda %50 indirim; önbellek indirimiyle birleşebilir</td><td>%50 ek indirim (24 saate kadar gecikme)</td></tr>
<tr><td>7</td><td><b>Fark tabanlı düzeltme</b></td><td>Düzeltme ajanından tüm makale yerine yalnızca değişen bölümleri / yamaları istemek</td><td>çıktı tokenı büyük ölçüde azalır</td></tr>
<tr><td>8</td><td><b>Kontrol noktası + devam</b></td><td>Her aşama çıktısını kaydedip hata olunca kaldığı yerden sürdürmek (bizim <code>resume_with_claude.py</code>)</td><td>başarısızlıkta israfı önler</td></tr>
<tr><td>9</td><td><b>Görüntüleri küçültme</b></td><td>Eleştirmen ve hakeme düşük çözünürlüklü sayfa/grafik; ya da yalnız değişen sayfa</td><td>görüntü tokenı azalır</td></tr>
<tr><td>10</td><td><b>Kota dışı backend</b></td><td>Aracın kendi tasarımına uygun bir Gemini API anahtarı veya ayrı bir Anthropic API anahtarı; abonelik kotası günlük işe kalır (veri gizliliği için onay gerekir)</td><td>abonelik kotası korunur</td></tr>
<tr><td>11</td><td><b>Seri çalışma + bütçe sınırı</b></td><td>Paralelliği kapatmak, çağrı başına çıktı sınırı, toplam çağrı/token bütçesi aşılınca durma</td><td>limit sürprizlerini önler</td></tr></table>
<div class="box"><b>Öneri:</b> Bir sonraki koşuda 1 + 2 + 3 + 4 + 8 birlikte uygulanırsa aynı makale kabaca <b>%70-80 daha az kota</b> ile çıkar. Kalıcı kullanım için ayrı API anahtarı + önbellek + batch (5, 6, 10) en ucuz yoldur.</div>

<h2>5. Otomatik hakemin kararı: 3/10, Reject</h2>
<p>Alt puanlar: Özgünlük 2, Kalite 2, Netlik 3, Önem 2, Sağlamlık 2, Sunum 3, Katkı 2 (hakem güveni 4/5). İlk taslak ve v1 aynı puanı aldı.</p>
<h3>Güçlü bulduğu yanlar</h3><ul>
<li>Dürüst çerçeve: gerilememenin basit bir özdeşlik olduğunu kabul etmesi, üstünlük değil eşitlik iddia etmesi, sınırlamaları açıkça yazması.</li>
<li>Dikkatli kapalı döngü protokolü: doğrulama / geliştirme / final test seed'leri ayrı; final seed'lere ayar dondurulana kadar dokunulmamış; seed bazında işaret testi.</li>
<li>Anlamlı değerlendirme: üçüncü taraf yığında yalnızca dedektörün değiştirilmesi; çevrimdışı metriklerin kapalı döngü başarısını öngörmediğini göstermesi.</li>
<li>Bilgilendirici döngü içi tanılar: kapı boyutuna göre köşe hatası, çift tespitin EKF çöküşünü öngörmesi, mesafeye göre hata (bizim kötü olduğumuz uzak mesafe dahil).</li>
<li>Pratik Orin NX ölçümleri ve fp16'da sayısal özdeşliği korumak için ayrı TensorRT motoru notu.</li>
<li>Olumsuz sonuçların şeffaf raporlanması (yeniden eğitim, kayıp ağırlıkları, stride-2 başlık, model çorbası).</li></ul>
<h3>Ret sebepleri</h3>
<table><tr><th>Eleştiri</th><th>Yapılacak</th><th>Durum</th></tr>
<tr><td>Özgünlük sınırlı: dondurulmuş gövde + yeni başlık bilinen bir yöntem; tek simülatör, tek yığın, tek rakip</td><td>Kendini geliştirme döngüsü, birden çok koşul modülü, yeni görev</td><td>başladı (koşul modülleri + yönlendirici)</td></tr>
<tr><td>Verimlilik iddiası kanıtsız: modül gecikmesi, GateNet'in parametre/gecikme/enerjisi ölçülmemiş</td><td>Orin'de tüm modüller + GateNet</td><td>Orin erişimi gerekli</td></tr>
<tr><td>Modüller tarif edilmemiş (mimari, parametre, veri, kayıp, eğitim)</td><td>Makaleye tablo</td><td>kolay</td></tr>
<tr><td>Asimetrik karşılaştırma: tekrar silme / min-size yalnız bize ayarlandı</td><td>Aynı son işlemle GateNet</td><td class="ok">kod hazır, 128 uçuş kuyrukta</td></tr>
<tr><td>Son ayarın ablasyonu yok; CornerRefiner döngüde ölçülebilir kazanç göstermeden tutulmuş</td><td>Her modül/filtre çıkarılarak doğrulama seed'lerinde</td><td>sırada</td></tr>
<tr><td>Eşitlik, sıfır hipotezini reddedememekten çıkarılmış; eşdeğerlik testi / güven aralığı yok</td><td>Önceden belirlenmiş marjla eşdeğerlik testi + bootstrap GA</td><td>kolay</td></tr>
<tr><td>Çekirdeği değiştirmenin işe yaramadığı yalnız niteliksel; adapter / LoRA / ince ayarlı kopya karşılaştırması yok</td><td>Aynı veriyle bu temellerin hem görev kazancı hem genel metrik kaybı</td><td>sırada</td></tr>
<tr><td>Yalnız simülasyon; tam modül hattı gerçek kapılarda denenmemiş; çekirdek zaten kapı görmüş; genel metrikler 448 px, kullanım 384 px</td><td>Gerçek veri, gerçekten yeni görev, 384 px metrikleri</td><td>planlandı</td></tr>
<tr><td>İlgili çalışmalar zayıf (yalnız 10 kaynak); side-tuning ve dondurulmuş gövde literatürü eksik</td><td>Literatür raporumuzdaki 52 kaynaktan genişletme</td><td>kolay</td></tr></table>

<h2>6. Koşarken karşılaşılan sorunlar</h2>
<table><tr><th>Sorun</th><th>Ne yaptık</th></tr>
<tr><td>Araç bir Claude skill'i değil; yalnızca Gemini/OpenAI anahtarıyla çalışıyor, makinede anahtar yoktu</td><td>Kodunu değiştirmeden Gemini istemcisini <code>claude -p</code>'ye yönlendiren ara katman (<code>run_with_claude.py</code>); veri Google/OpenAI'ye gitmedi</td></tr>
<tr><td>Kavramsal şemalar Gemini'nin resim modeliyle çiziliyor; Claude resim üretemiyor</td><td>Şemalar Claude'un yazdığı matplotlib koduyla çizildi (tek sapma)</td></tr>
<tr><td>LaTeX (pdflatex) yoktu, sudo yok</td><td>Tek dosyalık tectonic kuruldu, pdflatex/bibtex sarmalayıcısı</td></tr>
<tr><td>PaperBanana örnek veri seti yayınlanmamış</td><td>Araç kendi uyarısıyla örneksiz devam etti (orijinal sürümle aynı)</td></tr>
<tr><td>5 saatlik kullanım limiti iki kez doldu; araç hata olunca en baştan başlıyor</td><td>Kaydedilmiş outline / literatür / grafiklerden kaldığı yerden devam betiği (<code>resume_with_claude.py</code>)</td></tr>
<tr><td><b>Haftalık limit doldu</b> (iyileştirme turu 2'de)</td><td>Araç v1 ile bitirdi; 3 tur yerine 1 iyileştirme turu yapılabildi</td></tr>
<tr><td>Ayrı Claude oturumu ilk denemede kredi isteyen bir modeli seçti</td><td>Model açıkça Opus 5.5 olarak verildi</td></tr>
<tr><td>Semantic Scholar hız sınırı</td><td>Literatür doğrulaması yavaş/sınırlı; kaynakça 10 makalede kaldı</td></tr>
<tr><td>Bir yeniden denemede şablon dosyası boş şablonla ezildi</td><td>Devam betiği literatür çıktısındaki güncel şablonu geri kopyalıyor</td></tr></table>

<h2>7. Plana etkisi</h2>
<div class="box">Hakemin istedikleri literatür taramasındaki boşluklarla aynı yeri gösteriyor. Plan (<code>docs/SELF_IMPROVE_PLAN.md</code>) buna göre güncellendi:
(1) koşula özel modüller + koşul farkındalıklı yönlendirici — <b>şu an eğitiliyor</b>; (2) aynı son işlemle GateNet — <b>kuyrukta</b>; (3) ablasyon + eşdeğerlik testi;
(4) Orin'de modül gecikmesi; (5) LoRA / adapter / ince ayarlı kopya temelleri; (6) gerçekten yeni görev + gerçek gece/NIR verisi; (7) makalede modül tarifleri ve geniş literatür.</div>
<div class="warn">Kota uyarısı: Bundan sonra <code>claude -p</code> zinciri gibi abonelik kotasını yiyen uzun işler kullanıcı onayı olmadan başlatılmayacak; yerel GPU işleri kota harcamaz.</div>
<p style="font-size:8pt;color:#666">Kaynaklar: github.com/google-research/paper-orchestra (README, kaynak kod); Anthropic fiyatlandırma ve prompt caching belgeleri (platform.claude.com/docs/en/about-claude/pricing, .../build-with-claude/prompt-caching). Sayılar bu koşunun çağrı günlüğünden (po_calls.log) hesaplandı.</p>
</body></html>"""
(D / "PaperOrchestra_ozet.html").write_text(html)
weasyprint.HTML(string=html).write_pdf(D / "PaperOrchestra_ozet.pdf"); print("wrote", D / "PaperOrchestra_ozet.pdf")
