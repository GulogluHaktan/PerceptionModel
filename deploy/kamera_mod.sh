#!/bin/bash
# Switch the IMX219 between our wide raw modes and the stock driver (Update 167). Reboots the Orin (~1.5 min).
#   bash kamera_mod.sh ham     -> custom driver: raw V4L2 1640x616 @ 84 fps / 1640x864 @ 60 fps, full width (~122 deg); our GPU ISP. Argus does NOT work.
#   bash kamera_mod.sh argus   -> stock NVIDIA driver + Argus ISP: 1640x1232 @ 30 (~122 deg), 1280x720 @ 60 (~94 deg); clean colours.
#   bash kamera_mod.sh durum   -> which one is installed
set -e
B=/home/lovelace/imx219_backup; K=/lib/modules/$(uname -r)/updates/drivers/media/i2c/nv_imx219.ko; O=/boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo
case "$1" in
  ham)   sudo cp $B/nv_imx219.ko.wide_final $K; sudo cp $B/ab.dtbo $O ;;
  argus) sudo cp $B/nv_imx219.ko.orig $K;    sudo cp $B/imx219-dual.dtbo.orig $O ;;
  durum) if cmp -s $K $B/nv_imx219.ko.orig; then echo "argus (stock driver)"; else echo "ham (custom wide modes)"; fi; exit 0 ;;
  *) echo "usage: bash kamera_mod.sh ham|argus|durum"; exit 1 ;;
esac
sudo depmod -a; sync; echo "installed: $1 -> rebooting"; sudo reboot
