"""Fine outdoor class set for the plug-in outdoor segmentation module = the 19 Cityscapes train ids (BDD100K raw labels use them directly).
The frozen core's 11 classes put fence / pole / signs / lights / trucks / buses / two-wheelers into a few coarse classes (other_obstacle, car, tree_pole)."""
OUT_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
               "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
K_OUT = len(OUT_CLASSES)
PALETTE_OUT = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
               (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
               (0, 0, 230), (119, 11, 32)]
