"""Turkish PDF report of the whole PerceptionModel effort (plain language, tables, before/after picture). Numbers come from docs/report/results.json (kept by hand from the experiment log,
docs/handoff/perceptionmodel-sim-plan.md). Usage: python scripts/make_report_tr.py -> docs/report/PerceptionModel_rapor.pdf (+ .html, chart png)."""
import json, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import weasyprint
D = Path("docs/report"); R = json.load(open(D / "results.json")); VIS = Path(R.get("compare_image", "runs/eval/vis/real_gates_compare.jpg")).resolve()
# chart: one measure, ordered steps -> horizontal bars, one hue, value at the bar end
J = R["journey"]; fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(J) + 0.9), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb"); y = range(len(J))[::-1]
ax.barh(list(y), [j[1] * 100 for j in J], height=0.55, color="#2a78d6")
for yy, j in zip(y, J): ax.text(j[1] * 100 + 1.5, yy, f"%{round(j[1] * 100)}", va="center", fontsize=9, color="#0b0b0b")
ax.set_yticks(list(y)); ax.set_yticklabels([j[0] for j in J], fontsize=8.5, color="#0b0b0b"); ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100]); ax.tick_params(axis="x", labelsize=8, colors="#52514e", length=0); ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xlabel("Bulunan kapı payı (%): hiç görülmemiş gerçek yer, köşe hatası 20 px içinde", fontsize=8.5, color="#52514e"); fig.tight_layout(); fig.savefig(D / "journey.png"); plt.close(fig)
def table(head, rows, cls=""):
    return f"<table class='{cls}'><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
def sec(key, title, head, intro="", after=""):
    return f"<h2>{title}</h2>{intro}{table(head, R[key])}{after}" if R.get(key) else ""
M = R["models"]
html = f"""<html><head><meta charset='utf-8'><style>
@page {{ size: A4; margin: 18mm 16mm; @bottom-center {{ content: counter(page); font-size: 8pt; color: #777; }} }}
body {{ font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; line-height: 1.45; color: #111; }}
h1 {{ font-size: 19pt; margin: 0 0 2mm 0; }} h2 {{ font-size: 13pt; margin: 7mm 0 2mm 0; border-bottom: 1px solid #bbb; padding-bottom: 1mm; break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 3mm 0; font-size: 8.8pt; break-inside: avoid; }} th, td {{ border: 1px solid #c8c8c8; padding: 1.4mm 2mm; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; }} td:not(:first-child), th:not(:first-child) {{ text-align: center; }} table.txt td:not(:first-child), table.txt th:not(:first-child) {{ text-align: left; }}
.box {{ background: #f4f6f8; border-left: 3px solid #2a78d6; padding: 2.5mm 3.5mm; margin: 3mm 0; }} .warn {{ border-left-color: #c98500; background: #fbf6e9; }}
.small {{ font-size: 8.3pt; color: #444; }} img {{ max-width: 100%; }} li {{ margin-bottom: 1mm; }}
</style></head><body>
<h1>PerceptionModel: süreç ve son durum raporu</h1>
<p class='small'>{R['date']} · RTX 5090 üzerinde yapılan eğitim ve ölçümler · ayrıntılı deney günlüğü: docs/handoff/perceptionmodel-sim-plan.md (Update 24–{R.get('last_update', 45)})</p>
<div class='box'><b>Kısa özet.</b> {R['summary']}</div>
<h2>1. Model ne yapıyor?</h2>
<p>Tek bir kamera görüntüsünden üç şeyi aynı anda çıkaran küçük bir "öğrenci" ağ eğitiyoruz: <b>(1)</b> yarış kapılarının dört köşesi (en çok 4 kapı, kapı kısmen görüntü dışında olsa bile),
<b>(2)</b> her pikselin sınıfı (zemin/yol, bina, ağaç/direk, insan, araba, kaya, şerit çizgisi, rakip drone, diğer engel, kapı), <b>(3)</b> göreli derinlik (yakın/uzak haritası).
Ağ, büyük hazır modellerden (DINOv2 ve Depth Anything v2) bilgi damıtılarak ve etiketli verilerle birlikte eğitiliyor. İki öğrenci türü var: <b>vit</b> (daha büyük, şu an ana model) ve
<b>hybrid</b> (FastViT; Jetson Orin'de hızlı çalışması hedeflenen küçük model).</p>
<h2>2. Hangi verilerle eğitildi?</h2>
{table(["Kaynak", "Tür", "Ne için"], R["data_sources"], "txt")}
<p class='small'>Kapı genellemesini ölçmek için MonoRaceGate'in "red" adlı yeri eğitime hiç konmadı; aşağıdaki tüm "hiç görülmemiş yer" sayıları oradan gelir (196 görüntü, 312 kapı).
TII salonundaki altın set eğitimle aynı salondan olduğu için genelleme sayısı olarak kullanılmadı.</p>
<h2>3. Yolculuk: sıfırdan bugüne</h2>
<p>Başlangıçta yalnızca simülasyonla eğitilen model simülasyonda çok iyi görünüyordu (kapı IoU 0.98), ama gerçek görüntülerde <b>hiçbir kapıyı bulamadı</b>. Aşağıdaki grafik, hiç görülmemiş gerçek bir yerde
bulunan kapı payının adım adım nasıl değiştiğini gösteriyor.</p>
<img src='{(D / 'journey.png').resolve()}'>
<ul>{''.join(f'<li>{x}</li>' for x in R['journey_notes'])}</ul>
<h2>4. Önce / sonra (hiç görülmemiş gerçek yer)</h2>
<p>Beyaz çizgi insan etiketidir. Solda eski başlıklı model (bigB, kırmızı tahmin), sağda yoğun başlıklı model (bigC, yeşil tahmin). Görüntüler eğitimde hiç kullanılmayan "red" yerinden.</p>
<img src='{VIS}' style="width: 62%; display: block; margin: 0 auto;">
<h2>5. Son durum: sayılar</h2>
{table(["Ölçüt"] + M, R["rows"])}
<p class='small'>{R['rows_note']}</p>
<div class='box warn'><b>Sayıları okurken.</b> Tek tohumla (tek eğitim denemesiyle) alınan sonuçlarda yaklaşık ±0.06 oynama vardır; bu büyüklükten küçük farklar anlamlı sayılmamalı.
Simülasyon içi sonuçlar (simülasyonda eğit, simülasyonda ölç) gerçek dünyadaki başarı oranı değildir; yalnızca "model bozuldu mu" kontrolü olarak verilir.</div>
{("<h2>Modelin kendi etiketlediği örnekler</h2><p>" + R.get("sheet_text", "") + "</p><img src='" + str(Path(R["sheet_image"]).resolve()) + "' style='width: 66%; display: block; margin: 0 auto;'>") if R.get("sheet_image") else ""}
<h2>6. Simülasyondaki "uzak kapı kaybı" neydi?</h2>
<p>{R['sim_text']}</p>
{table(["Kapı boyutu (256×192 görüntüde)"] + R.get("sim_models", M), R["sim_size"])}
<p class='small'>Ölçüt: köşe hatası, 3 pikselden ve kapı boyutunun dörtte birinden küçük olmalı (simülasyon içi ölçüm, 1.110 kare). Eski "20 px içinde" ölçütü 3 piksellik bir kapı için çok gevşekti ve kaba tahmini ödüllendiriyordu.</p>
{sec("ablation", "7. Hangi değişiklik ne kadar katkı yaptı? (ablation ve ikinci tohum)", R.get("ablation_head", []), R.get("ablation_intro", ""), R.get("ablation_after", ""))}
{sec("hybrid", "8. Küçük model (hybrid, FastViT)", R.get("hybrid_head", []), R.get("hybrid_intro", ""), R.get("hybrid_after", ""))}
{sec("export", "9. Dışa aktarım (ONNX / TensorRT)", R.get("export_head", []), R.get("export_intro", ""), R.get("export_after", ""))}
{sec("envs", "Farklı ortamlarda sınama", R.get("envs_head", []), R.get("envs_intro", ""), R.get("envs_after", ""))}
{("<img src='" + str(Path(R["envs_image"]).resolve()) + "' style='width: 66%; display: block; margin: 0 auto;'>") if R.get("envs_image") else ""}
<h2>10. Dürüst sınırlar: neyi bilmiyoruz?</h2>
<ul>{''.join(f'<li>{x}</li>' for x in R['limits'])}</ul>
<h2>11. Sıradaki adımlar</h2>
<ul>{''.join(f'<li>{x}</li>' for x in R['next'])}</ul>
</body></html>"""
(D / "PerceptionModel_rapor.html").write_text(html); weasyprint.HTML(string=html, base_url=str(Path.cwd())).write_pdf(D / "PerceptionModel_rapor.pdf"); print("wrote", D / "PerceptionModel_rapor.pdf", os.path.getsize(D / "PerceptionModel_rapor.pdf") // 1024, "kB")
