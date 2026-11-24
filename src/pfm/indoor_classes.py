"""Fine indoor class set for the plug-in indoor segmentation module (the frozen core's 11 classes put every desk object into other_obstacle).
Index 0 = other (anything not listed: outdoor stuff, food, animals, ...). 255 = ignore (unlabelled).
COCO panoptic (133 categories) -> indoor id via COCO2IN (used for the real COCO labels and for the COCO-panoptic Mask2Former teacher)."""
IN_CLASSES = ["other", "wall", "floor", "ceiling", "door", "window", "person", "chair", "sofa", "table", "cabinet_shelf", "bed",
              "monitor_tv", "laptop", "keyboard", "mouse", "phone_remote", "cup_bottle", "book_paper", "plant", "bag", "appliance"]
K_IN = len(IN_CLASSES)
_ix = {c: i for i, c in enumerate(IN_CLASSES)}
_GROUPS = {
    "wall": ["wall-brick", "wall-stone", "wall-tile", "wall-wood", "wall-other-merged"],
    "floor": ["floor-wood", "floor-other-merged", "rug-merged"],
    "ceiling": ["ceiling-merged"],
    "door": ["door-stuff"],
    "window": ["window-blind", "window-other", "curtain"],
    "person": ["person"],
    "chair": ["chair", "bench"],
    "sofa": ["couch"],
    "table": ["dining table", "table-merged", "counter"],
    "cabinet_shelf": ["cabinet-merged", "shelf"],
    "bed": ["bed", "pillow", "blanket"],
    "monitor_tv": ["tv"],
    "laptop": ["laptop"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "phone_remote": ["cell phone", "remote"],
    "cup_bottle": ["cup", "bottle", "wine glass", "bowl", "vase"],
    "book_paper": ["book", "paper-merged", "cardboard"],
    "plant": ["potted plant", "flower"],
    "bag": ["backpack", "handbag", "suitcase"],
    "appliance": ["refrigerator", "oven", "microwave", "toaster", "sink", "toilet"],
}
COCO2IN = {name: _ix[c] for c, names in _GROUPS.items() for name in names}       # other COCO names -> 0
PALETTE = [(60, 60, 60), (170, 130, 90), (128, 64, 128), (200, 200, 230), (140, 80, 20), (90, 190, 230), (220, 20, 60), (255, 160, 0),
           (190, 90, 200), (120, 70, 30), (100, 140, 60), (240, 120, 160), (0, 80, 230), (0, 200, 200), (255, 255, 0), (255, 0, 255),
           (0, 255, 120), (0, 120, 255), (250, 250, 250), (0, 160, 0), (150, 40, 40), (110, 110, 160)]


def coco_lut(id2label: dict) -> list:
    """COCO category / teacher label ids -> indoor ids (list indexed by the source id)."""
    n = max(int(k) for k in id2label) + 1; lut = [0] * n
    for k, name in id2label.items(): lut[int(k)] = COCO2IN.get(name, 0)
    return lut
