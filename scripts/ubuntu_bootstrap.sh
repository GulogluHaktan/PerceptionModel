#!/usr/bin/env bash
# Sifirdan Ubuntu (22.04 / 24.04) kurulum scripti. En az insan mudahalesi: bir kere sudo sifresi, gerisi otomatik. Tekrar calistirilabilir (kurulu olani atlar).
#   bash ubuntu_bootstrap.sh                       hepsini kurar
#   SKIP="isaac ros" bash ubuntu_bootstrap.sh      adimlari atla      (adimlar: vault hostname base nvidia python docker ros mujoco isaac claude ollama vscode micro calc obsidian tailscale anydesk)
#   ONLY="tailscale vault" bash ubuntu_bootstrap.sh sadece bunlar
#   DRY_RUN=1 bash ubuntu_bootstrap.sh             hicbir sey kurmadan akisi dener
# Ayarlar (ortam degiskeni): ISAACSIM_VERSION=6.0.1.0  ISAACLAB_REF=v3.0.0-beta2  (Sim 6.2 icin: 6.2.0.0 + v3.0.0)  OLLAMA_MODELS="qwen3-vl:8b-instruct qwen3:8b"  NVIDIA_DRIVER=nvidia-driver-580-open  TS_AUTHKEY=tskey-...  INSTALL_ANYDESK=1  VAULT=~/Documents/ObsidianVault
# ONERILEN (ikinci makine): NVIDIA_DRIVER=nvidia-driver-580-open HOSTNAME_NEW=<ad> TS_AUTHKEY=tskey-... bash ubuntu_bootstrap.sh      (surucu kurulursa: yeniden baslat, ayni komutu tekrar calistir)
# Hata olursa script durmaz, sonda OZET verir; duzeltip ayni komutu tekrar calistir (kurulu olani atlar, indirilenler onbellekten gelir). ISAAC adimi ~30 GB indirir: ag kesilirse tekrar dener.
# Her adimin sonucu Obsidian vault'taki log.md'ye yazilir. Surucu kurulursa script "yeniden baslat, sonra tekrar calistir" der (Secure Boot aciksa MOK sifresi sorar: tek elle adim).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}   # buyuk Isaac/PyTorch paketlerinde uv varsayilani 30 sn zaman asimina dusuyordu
ISAACSIM_VERSION=${ISAACSIM_VERSION:-6.0.1.0}; ISAACLAB_REF=${ISAACLAB_REF:-v3.0.0-beta2}; OLLAMA_MODELS=${OLLAMA_MODELS:-}; TS_AUTHKEY=${TS_AUTHKEY:-}
INSTALL_ANYDESK=${INSTALL_ANYDESK:-0}; VAULT=${VAULT:-$HOME/Documents/ObsidianVault}; SKIP=${SKIP:-}; ONLY=${ONLY:-}; DRY_RUN=${DRY_RUN:-0}
VENVS=${VENVS:-$HOME/venvs}; ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}; HOSTNAME_NEW=${HOSTNAME_NEW:-}; BIN=$HOME/.local/bin; mkdir -p "$VENVS" "$BIN" "$VAULT"; export PATH="$BIN:$HOME/.local/share/uv/bin:$PATH"
LOG=$HOME/bootstrap_$(date +%Y%m%d_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
[ "$(id -u)" -ne 0 ] || { echo "root olarak degil, normal kullanici olarak calistir (sudo'yu script kendisi kullanir)"; exit 1; }
. /etc/os-release; CODENAME=${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}; ID=${ID:-}; VERSION_ID=${VERSION_ID:-}
if [ "$DRY_RUN" != 1 ]; then
  [ "$ID" = ubuntu ] || { echo "Bu script Ubuntu icin (bulundu: $ID)"; exit 1; }
  case "$VERSION_ID" in 22.04|24.04) ;; *) echo "UYARI: sadece 22.04/24.04 denendi (bulundu $VERSION_ID)";; esac
  sudo -v; ( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &    # sudo sifresi bir kere
fi
APT="sudo apt-get -y -o DPkg::Lock::Timeout=600"; rm -f /tmp/.pfm_need_reboot; declare -A RESULT; ORDER=()
have() { command -v "$1" >/dev/null 2>&1; }
vlog() { printf -- "- %s Ubuntu bootstrap: %s\n" "$(date +%F)" "$*" >> "$VAULT/log.md"; }
dl() { curl -fsSL --retry 3 "$@"; }
retry() { local n=0; until "$@"; do n=$((n+1)); [ $n -ge 3 ] && return 1; echo "basarisiz, tekrar deneniyor ($n/3): $1 $2"; sleep 10; done; }   # ag kesintisi (uv zaman asimi) icin

step() {                                   # step <ad> <fonksiyon>: hata olursa digerlerine devam eder, sonunda ozet
  local n=$1 f=$2; ORDER+=("$n")
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $n "* ]]; then RESULT[$n]="atlandi (ONLY)"; return; fi
  if [[ " $SKIP " == *" $n "* ]]; then RESULT[$n]="atlandi (SKIP)"; return; fi
  echo; echo "=== $n ($(date +%H:%M:%S))"
  if [ "$DRY_RUN" = 1 ]; then RESULT[$n]="dry-run"; return; fi
  ( set -e; $f ); local rc=$?                                                   # NOT: "if ( set -e; ... )" kullanma, bash if icinde set -e'yi yok sayar ve hatali adim OK gorunur
  if [ $rc -eq 0 ]; then RESULT[$n]="OK"; vlog "$n kuruldu/dogrulandi"; else RESULT[$n]="HATA (kod $rc, bkz. $LOG)"; vlog "$n HATA, log: $LOG"; fi
}

s_hostname() {                                                                  # HOSTNAME_NEW verilmisse bilgisayar adini degistirir (kullanici adina dokunmaz)
  [ -n "$HOSTNAME_NEW" ] || { echo "HOSTNAME_NEW verilmedi: atlandi"; return 0; }
  sudo hostnamectl set-hostname "$HOSTNAME_NEW"
  if grep -q '^127.0.1.1' /etc/hosts; then sudo sed -i "s/^127.0.1.1.*/127.0.1.1 $HOSTNAME_NEW/" /etc/hosts; else echo "127.0.1.1 $HOSTNAME_NEW" | sudo tee -a /etc/hosts >/dev/null; fi
  [ "$(hostnamectl --static)" = "$HOSTNAME_NEW" ]
}
s_base() {
  $APT update; $APT install ca-certificates curl wget gnupg lsb-release software-properties-common git build-essential unzip jq pciutils cmake pkg-config htop tmux
}
s_nvidia() {                                                                    # RTX 5090 (Blackwell) icin 570+ ACIK cekirdek modulu sart (Isaac Sim icin 580+ onerilir)
  lspci | grep -qi 'nvidia' || { echo "NVIDIA kart yok: atlandi"; return 0; }
  if nvidia-smi >/dev/null 2>&1; then
    local v; v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    [ "${v%%.*}" -ge 570 ] || echo "UYARI: surucu $v, 5090 icin 570+ gerekir: NVIDIA_DRIVER=nvidia-driver-580-open ile tekrar calistir"; return 0
  fi
  $APT install ubuntu-drivers-common
  if [ -n "${NVIDIA_DRIVER:-}" ]; then $APT install "$NVIDIA_DRIVER"                  # elle sec: NVIDIA_DRIVER=nvidia-driver-580-open
  else echo "onerilen surucu:"; ubuntu-drivers devices | grep -E 'recommended|driver' | head -5; sudo ubuntu-drivers install; fi
  touch /tmp/.pfm_need_reboot; echo "NVIDIA surucusu kuruldu: YENIDEN BASLAT, sonra scripti tekrar calistir (surum kontrol edilir, GPU'ya bagli adimlar dogrulanir)"
}
s_python() {
  have uv || { dl https://astral.sh/uv/install.sh | sh; }
  uv python install 3.11 3.12; uv python find 3.11 >/dev/null && uv python find 3.12 >/dev/null && echo "python 3.11 + 3.12 hazir"       # sistem python'una dokunmaz (ROS sistem python'unu kullanir, Isaac kendi venv'inde)
}
s_docker() {
  if ! have docker; then
    sudo install -m 0755 -d /etc/apt/keyrings; dl https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null; sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    $APT update; $APT install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
  sudo systemctl enable --now docker; sudo usermod -aG docker "$USER"; echo "docker grubu icin bir kere cikis-giris gerekir"
  if lspci | grep -qi nvidia && ! have nvidia-ctk; then
    dl https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    dl https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $APT update; $APT install nvidia-container-toolkit; sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker
  fi
}
s_ros() {                                                                       # 24.04 -> ROS 2 Jazzy + Gazebo Harmonic, 22.04 -> Humble + Gazebo
  local D; case "$VERSION_ID" in 24.04) D=jazzy;; 22.04) D=humble;; *) echo "bu surum icin ROS secilmedi"; return 1;; esac
  sudo add-apt-repository -y universe
  local V; V=$(dl https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | jq -r .tag_name)
  dl -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$V/ros2-apt-source_$V.${CODENAME}_all.deb"; sudo dpkg -i /tmp/ros2-apt-source.deb
  $APT update; $APT install "ros-$D-desktop" ros-dev-tools
  $APT install "ros-$D-ros-gz" || $APT install "ros-$D-ros-gz-sim" "ros-$D-ros-gz-bridge" || echo "UYARI: ros-gz paketi bu dagitimda bulunamadi, Gazebo'yu elle kur"
  test -f "/opt/ros/$D/setup.bash"
  grep -q "/opt/ros/$D/setup.bash" "$HOME/.bashrc" || echo "source /opt/ros/$D/setup.bash" >> "$HOME/.bashrc"          # ROS icin bash kullan (fish'te setup.bash calismaz)
}
s_mujoco() {
  $APT install libglfw3 libgl1 libosmesa6 libegl1; [ -d "$VENVS/mujoco" ] || uv venv --python 3.12 "$VENVS/mujoco"
  uv pip install -p "$VENVS/mujoco/bin/python" mujoco numpy; "$VENVS/mujoco/bin/python" -c "import mujoco; print('mujoco', mujoco.__version__)"
}
s_isaac() {                                                                     # Isaac Sim (pip, Python 3.12) + Isaac Lab, ayni venv
  have nvidia-smi || { echo "NVIDIA surucusu yok: once 'nvidia' adimi + yeniden baslatma"; return 1; }
  $APT install libglu1-mesa libxt6 libxrandr2 libxi6 libxkbcommon-x11-0 libgl1 libvulkan1
  [ -d "$VENVS/isaac" ] || uv venv --python 3.12 "$VENVS/isaac"
  retry uv pip install -p "$VENVS/isaac/bin/python" "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match    # uv varsayilani paketi ilk indekste bulunca durur (mujoco-usd-converter hatasi)
  [ -d "$ISAACLAB_DIR" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  ( cd "$ISAACLAB_DIR" && git fetch --tags -q && git checkout -q "$ISAACLAB_REF" && export VIRTUAL_ENV=$VENVS/isaac PATH="$VENVS/isaac/bin:$PATH" OMNI_KIT_ACCEPT_EULA=YES && retry ./isaaclab.sh --install )
  cat > "$BIN/isaac-py" <<EOF
#!/usr/bin/env bash
# Isaac Sim'in python'u (EULA kabulu dahil): isaac-py script.py
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y; exec "$VENVS/isaac/bin/python" "\$@"
EOF
  chmod +x "$BIN/isaac-py"; "$BIN/isaac-py" -c "import isaacsim, isaaclab, torch; print('isaacsim ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"   # gercek dogrulama: import edilemezse adim HATA olur
  cat > "$BIN/isaac-sim-gui" <<EOF
#!/usr/bin/env bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y; exec "$VENVS/isaac/bin/isaacsim" isaacsim.exp.full "\$@"
EOF
  chmod +x "$BIN/isaac-sim-gui"; mkdir -p "$HOME/.local/share/applications"
  printf '[Desktop Entry]\nType=Application\nName=Isaac Sim\nComment=NVIDIA Isaac Sim (pip kurulumu): tam arayuz\nExec=%s/isaac-sim-gui\nTerminal=false\nCategories=Development;Science;\nIcon=applications-science\nStartupNotify=true\n' "$BIN" > "$HOME/.local/share/applications/isaac-sim.desktop"
  command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true      # pip paketi menuye kendiliginden kisayol eklemez: Super tusuyla "Isaac Sim" aranabilir
  echo "Isaac: ilk calistirma shader derler (5-15 dk). Dogrulama: isaac-py -c 'import isaacsim; print(isaacsim.__file__)'"
}
s_claude() { have claude || { dl https://claude.ai/install.sh | bash; }; "$BIN/claude" --version 2>/dev/null || claude --version; }
s_ollama() {
  have ollama || { dl https://ollama.com/install.sh | sh; }; sudo systemctl enable --now ollama 2>/dev/null || true
  [ -z "$OLLAMA_MODELS" ] || { until curl -s localhost:11434 >/dev/null; do sleep 2; done; for m in $OLLAMA_MODELS; do ollama pull "$m"; done; }
}
s_vscode() {
  have code && return 0
  dl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/packages.microsoft.gpg >/dev/null
  echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null
  $APT update; $APT install code
}
s_micro() { have micro || $APT install micro; }
s_calc() { have localc || $APT install libreoffice-calc; }
s_obsidian() {
  have obsidian && return 0
  local U; U=$(dl https://api.github.com/repos/obsidianmd/obsidian-releases/releases/latest | jq -r '.assets[].browser_download_url' | grep -E 'amd64\.deb$' | head -1)
  [ -n "$U" ] || { echo "Obsidian .deb adresi bulunamadi"; return 1; }; dl -o /tmp/obsidian.deb "$U"; $APT install /tmp/obsidian.deb
}
s_tailscale() {                                                                 # uzaktan baglanti: AnyDesk yerine
  have tailscale || { dl https://tailscale.com/install.sh | sh; }; sudo systemctl enable --now tailscaled
  if [ -n "$TS_AUTHKEY" ]; then sudo tailscale up --authkey="$TS_AUTHKEY" --ssh; tailscale ip -4
  else echo "Tailscale girisi icin bir kere: sudo tailscale up --ssh   (verilen linki tarayicida ac). Tamamen otomatik icin TS_AUTHKEY=tskey-... ver."; fi
}
s_anydesk() {
  [ "$INSTALL_ANYDESK" = 1 ] || { echo "AnyDesk kurulmadi (uzaktan baglanti icin Tailscale var). Istersen INSTALL_ANYDESK=1"; return 0; }
  dl https://keys.anydesk.com/repos/DEB-GPG-KEY | sudo gpg --dearmor --yes -o /etc/apt/keyrings/keys.anydesk.com.gpg
  echo "deb [signed-by=/etc/apt/keyrings/keys.anydesk.com.gpg] http://deb.anydesk.com/ all main" | sudo tee /etc/apt/sources.list.d/anydesk.list >/dev/null; $APT update; $APT install anydesk
}
s_vault() {                                                                     # yapilan tum islemler Obsidian'da log.md'de
  mkdir -p "$VAULT"; [ -f "$VAULT/log.md" ] || printf "# Log\n\n" > "$VAULT/log.md"
  [ -f "$VAULT/index.md" ] || printf -- "---\ntype: index\ntitle: \"Vault Index\"\n---\n\n# Vault Index\n\nLog: [[log]]\n" > "$VAULT/index.md"
  echo "vault: $VAULT (log.md, index.md). Obsidian'da bu klasoru 'Open folder as vault' ile bir kere ac."
}

step vault s_vault          # once: sonraki adimlar buraya log yazar
step hostname s_hostname; step base s_base; step nvidia s_nvidia; step python s_python; step docker s_docker; step ros s_ros; step mujoco s_mujoco; step isaac s_isaac
step claude s_claude; step ollama s_ollama; step vscode s_vscode; step micro s_micro; step calc s_calc; step obsidian s_obsidian; step tailscale s_tailscale; step anydesk s_anydesk
RESULT[agy-terminal]="ATLANDI: 'Agy terminal' hangi uygulama? (ghostty/alacritty/kitty?) -> adini soyle, eklenir"

echo; echo "================ OZET ($LOG)"
for n in "${ORDER[@]}" agy-terminal; do printf "  %-12s %s\n" "$n" "${RESULT[$n]:-?}"; done
echo "Gercek kontrol:"
chk() { local n=$1; shift; if "$@" >/dev/null 2>&1; then printf "  %-12s ok\n" "$n"; else printf "  %-12s YOK/HATA\n" "$n"; fi; }
chk python3.11 uv python find 3.11; chk python3.12 uv python find 3.12; chk docker docker --version; chk docker-gpu bash -c "docker info 2>/dev/null | grep -q nvidia"
chk ros2 bash -c 'for d in /opt/ros/*; do [ -f $d/setup.bash ] && . $d/setup.bash && ros2 --version && exit 0; done; exit 1'; chk gazebo bash -c 'for d in /opt/ros/*; do [ -f $d/setup.bash ] && . $d/setup.bash; done; command -v gz'
chk mujoco "$VENVS/mujoco/bin/python" -c "import mujoco"; chk isaacsim "$BIN/isaac-py" -c "import isaacsim"; chk isaac-cuda "$BIN/isaac-py" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
for c in claude ollama code micro localc obsidian tailscale; do chk $c command -v $c; done
tailscale status >/dev/null 2>&1 && echo "  tailscale     baglandi" || echo "  tailscale     GIRIS YOK: sudo tailscale up --ssh"
[ -f /tmp/.pfm_need_reboot ] && echo ">>> YENIDEN BASLAT ve scripti tekrar calistir (surucu kuruldu)." || true
echo "Not: docker grubu ve PATH icin bir kere cikis-giris yap."
