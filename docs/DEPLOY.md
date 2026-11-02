# Deploying on a VPS (Ubuntu 24.04 LTS, behind nginx)

Serves the Streamlit app at `https://<name>.maulikvect.dev`. The FastAPI server is optional: the pages read the data and artefact files directly, so only Streamlit is run here. Replace `<name>` and `user@vps` below.

The app needs only `data/synth/` (about 27 MB) and `artefacts/` (about 184 KB). Both are gitignored, so they are built once on your own machine and copied up (or built on the server, see step 4).

## 1. DNS

Add an A record (and AAAA if the VPS has IPv6) for `<name>.maulikvect.dev` pointing at the VPS. Check it with `dig +short <name>.maulikvect.dev`.

## 2. Packages, user and firewall

```
sudo apt update && sudo apt install -y nginx git curl rsync certbot python3-certbot-nginx
sudo adduser --system --group --home /srv/secondlook secondlook
sudo ufw allow OpenSSH && sudo ufw allow "Nginx Full" && sudo ufw enable     # 8501 and 8000 stay closed
```

## 3. Code and Python 3.14

Ubuntu 24.04 ships Python 3.12, so use `uv` to get the 3.14 the project targets (`.python-version`). It installs into the service user's home and leaves the system Python alone.

```
sudo -u secondlook -H bash -c '
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  git clone https://github.com/Maulik-17/SecondLook.git /srv/secondlook/app-src
  cd /srv/secondlook/app-src
  uv venv --python 3.14 && uv pip install -r requirements.txt'
```

The code is now in `/srv/secondlook/app-src`. To update later: `sudo -u secondlook git -C /srv/secondlook/app-src pull`, then `sudo systemctl restart secondlook`.

## 4. Data and artefacts

Preferred: build on your machine (`bash scripts/run_all.sh`, about 7 minutes) and copy only what the app reads, keeping the folder layout:

```
cd ~/Documents/Hackathon/SecondLook
rsync -av --relative data/synth artefacts user@vps:/tmp/secondlook-upload/
# on the VPS:
sudo rsync -a /tmp/secondlook-upload/ /srv/secondlook/app-src/ && sudo chown -R secondlook:secondlook /srv/secondlook/app-src && rm -rf /tmp/secondlook-upload
```

Alternative: build on the server with `sudo -u secondlook -H bash -c 'cd /srv/secondlook/app-src && PYTHON=.venv/bin/python bash scripts/run_all.sh'`. The result is the same, it only takes longer.

## 5. systemd service

`/etc/systemd/system/secondlook.service`:

```
[Unit]
Description=SecondLook Streamlit app
After=network.target

[Service]
User=secondlook
WorkingDirectory=/srv/secondlook/app-src
Environment=PYTHONPATH=src
ExecStart=/srv/secondlook/app-src/.venv/bin/streamlit run app/Home.py --server.headless true --server.address 127.0.0.1 --server.port 8501
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload && sudo systemctl enable --now secondlook
systemctl status secondlook --no-pager          # should say active (running)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501/_stcore/health     # 200
```

## 6. nginx

`/etc/nginx/sites-available/secondlook`:

```
server {
    listen 80;
    listen [::]:80;
    server_name <name>.maulikvect.dev;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;       # Streamlit needs WebSockets
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

```
sudo ln -s /etc/nginx/sites-available/secondlook /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <name>.maulikvect.dev        # adds HTTPS and the redirect; renewal is automatic
```

## 7. Embedding in the portfolio

To show the app inside a page of the main site, use an iframe pointing at `https://<name>.maulikvect.dev/?embed=true`. Do not add an `X-Frame-Options` header in nginx.

## Checks and logs

- App logs: `journalctl -u secondlook -f`. nginx logs: `/var/log/nginx/error.log`.
- A blank page with a "connecting" spinner usually means the WebSocket headers in step 6 are missing.
- "Data or model not found" on a page means step 4 was skipped or the folder layout is wrong: `/srv/secondlook/app-src/data/synth/people.parquet` and `/srv/secondlook/app-src/artefacts/model.joblib` must exist.

## Licence note

The project is AGPL-3.0. A public deployment should let visitors reach the source, for example with a link to https://github.com/Maulik-17/SecondLook on the portfolio page or in the app.
