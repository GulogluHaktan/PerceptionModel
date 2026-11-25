"""Live preview window for rec_only.py (run on the Orin desktop / RDP session):  python3 ~/rec_view.py
Shows /tmp/pfm_preview.jpg (~3 fps) and the last status line of ~/rec_last.log; a red banner when no fresh frame arrives (recording stopped)."""
import os, time, tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
P = Path("/tmp/pfm_preview.jpg"); LOG = Path.home() / "rec_last.log"
root = tk.Tk(); root.title("PFM kayit onizleme"); root.geometry("1240x560"); root.configure(bg="black")
status = tk.Label(root, font=("DejaVu Sans Mono", 13), fg="white", bg="#202020", anchor="w"); status.pack(fill="x")
pic = tk.Label(root, bg="black"); pic.pack(fill="both", expand=True); last = [0.0]


def tick():
    try:
        line = [l for l in LOG.read_text(errors="ignore").splitlines() if l.strip()][-1] if LOG.exists() else "log yok"
    except Exception: line = "log okunamadi"
    age = time.time() - P.stat().st_mtime if P.exists() else 1e9
    if age < 3:
        status.config(text=f" KAYIT  | {line}", bg="#145a32")
        mt = P.stat().st_mtime
        if mt != last[0]:
            try:
                im = Image.open(P); w = max(200, pic.winfo_width()); h = max(100, pic.winfo_height()); im.thumbnail((w, h))
                tkimg = ImageTk.PhotoImage(im); pic.config(image=tkimg); pic.image = tkimg; last[0] = mt
            except Exception: pass
    else:
        status.config(text=f" GORUNTU YOK ({'hic' if age > 1e8 else f'{age:.0f} s'})  | {line}", bg="#922b21")
    root.after(300, tick)


tick(); root.mainloop()
