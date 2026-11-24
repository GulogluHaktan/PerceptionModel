"""One-time: log in to the Orin with its password and append the dev PC's deploy public key to ~/.ssh/authorized_keys (afterwards key login only).
The password is read from the environment variable ORIN_PW (never written to disk). Usage: ORIN_PW=... /usr/bin/python3 deploy/ssh_key_install.py user@ip"""
import os, sys
from pathlib import Path
import paramiko
user, host = sys.argv[1].split("@"); pub = (Path.home() / ".ssh/id_ed25519_orin.pub").read_text().strip()
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); c.connect(host, username=user, password=os.environ["ORIN_PW"], timeout=15, look_for_keys=False, allow_agent=False)
cmd = f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && (grep -qF '{pub}' ~/.ssh/authorized_keys || echo '{pub}' >> ~/.ssh/authorized_keys) && chmod 600 ~/.ssh/authorized_keys && hostname && uname -m"
_, o, e = c.exec_command(cmd); print(o.read().decode(), e.read().decode()); c.close()
