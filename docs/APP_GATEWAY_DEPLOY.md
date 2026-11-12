# Deploying the C2C App Gateway

The gateway lets other people run your workflows as apps while only you can open them. This guide sets it up on a
server you control. Read "What this protects" first.

## What this protects (and what it does not)

- **Protected:** runners (people you share an app with) never receive the workflow, its node graph, subgraphs, model
  names, node ids or the Vault payload. They reach only the gateway; the gateway reaches ComfyUI.
- **Not protected:** anyone with administrator or root access to the server, or to its backups. The workflow has to
  be readable by the ComfyUI process to run. Keep the server and the gateway's data folder as private as the workflow.
- ComfyUI's App Mode or hidden panels are **not** used as protection. The protection is the network: ComfyUI must not
  be reachable from outside.

## Layout

```
internet ──HTTPS──► reverse proxy (Caddy / nginx, TLS) ──► gateway 127.0.0.1:8390 ──► ComfyUI 127.0.0.1:8188
```

Everything except the reverse proxy listens on 127.0.0.1. Only ports 80/443 of the proxy are open in the firewall.

## 1. ComfyUI (the worker)

Start ComfyUI on loopback only, with its normal custom nodes and models:

```
python main.py --listen 127.0.0.1 --port 8188
```

- Do not pass `--listen 0.0.0.0`, and do not add `--enable-cors-header`.
- If ComfyUI runs on another machine, put both machines on a private network (VPN, VPC) and allow the ComfyUI port
  only from the gateway's address. Never publish it.

## 2. Gateway configuration

Create `gateway.json` outside any web-served folder:

```json
{
  "data_dir": "/srv/c2c-gateway/data",
  "bind": "127.0.0.1",
  "port": 8390,
  "public_origin": "https://apps.example.com",
  "worker_url": "http://127.0.0.1:8188",
  "secure_cookies": true,
  "trust_forwarded_for": true,
  "max_upload_mb": 50,
  "retention_h": 72,
  "worker_input_dir": "/srv/comfyui/input"
}
```

| Setting | Meaning |
|---|---|
| `public_origin` | The exact https address people type. Sign-in and every change are refused from any other origin. |
| `secure_cookies` | Keep `true` (cookies only over HTTPS). `false` is for local tests only. |
| `trust_forwarded_for` | `true` only behind your proxy, so limits and the audit log see the real client address. |
| `max_upload_mb` | Largest upload the gateway accepts (each app can set a lower limit). |
| `retention_h` | Finished runs, their outputs and their download links are deleted after this many hours. |
| `worker_input_dir` | ComfyUI's `input` folder when it is on the same disk: runner uploads are deleted when a run ends. Leave it out for a remote worker and clear `input/c2c_gateway/` there on a schedule. |
| `worker_headers` | Extra headers for the worker (for example an auth header if ComfyUI sits behind its own proxy). |
| `max_active_total`, `max_active_per_user`, `max_output_mb` | Server-wide limits on top of each app's own limits. |
| `studio_proxy` | `true` (default): the owner opens ComfyUI at `https://…/studio/`. `false`: the owner uses an SSH tunnel instead. |

Check it, add the owner, start it:

```
python -m c2c_app_gateway check --config gateway.json
python -m c2c_app_gateway add-user --config gateway.json --name you --role owner
python -m c2c_app_gateway serve --config gateway.json
```

`add-user` asks for the password (it is never taken from the command line). Add runners and publishers from the
owner account later (owner API) or with `add-user`.

Permissions: the data folder holds `secret.key`, the user list, the encrypted releases, outputs and the audit log.
Make it readable only by the account that runs the gateway (`chmod 700` on Linux; on Windows remove inherited
access for other users).

## 3. Reverse proxy with TLS

The proxy must (a) terminate TLS, (b) pass WebSockets for `/studio/ws`, (c) not buffer the progress stream
(`/api/v1/jobs/*/events`, Server-Sent Events), and (d) allow uploads up to `max_upload_mb`.

**Caddy** (automatic certificates, WebSockets pass through by default):

```
apps.example.com {
    request_body {
        max_size 60MB
    }
    reverse_proxy 127.0.0.1:8390 {
        flush_interval -1
    }
}
```

**nginx:**

```
server {
    listen 443 ssl http2;
    server_name apps.example.com;
    ssl_certificate     /etc/letsencrypt/live/apps.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/apps.example.com/privkey.pem;
    client_max_body_size 60m;

    location / {
        proxy_pass http://127.0.0.1:8390;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
map $http_upgrade $connection_upgrade { default upgrade; '' close; }
```

Set `X-Forwarded-For` to `$remote_addr` (not appended), so a client cannot forge the address the gateway records.

## 4. Firewall

Allow inbound 80 and 443 (the proxy) and your SSH port. Block everything else, in particular 8188 (ComfyUI) and
8390 (gateway). Check from another machine that `http://<server>:8188/` and `:8390/` do not answer.

## 5. Running it as a service

**Linux (systemd)**, `/etc/systemd/system/c2c-gateway.service`:

```
[Unit]
Description=C2C App Gateway
After=network.target

[Service]
User=c2cgw
WorkingDirectory=/srv/ComfyUI/custom_nodes/ComfyUI-CustomNodePacks
ExecStart=/srv/ComfyUI/venv/bin/python -m c2c_app_gateway serve --config /srv/c2c-gateway/gateway.json
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Windows:** run the same command from a service wrapper (for example NSSM) or a Task Scheduler task that starts at
boot under a dedicated account.

The gateway needs `aiohttp` and `cryptography` (both already present in a ComfyUI environment).

## 6. Publishing and sharing

1. Sign in at `https://apps.example.com/` as the owner, then open `https://apps.example.com/studio/` - that is your
   ComfyUI. Only owner accounts can open it; everyone else gets "not found".
2. Build the workflow, right-click the canvas, **📤 Publish as C2C App…**. Tick the inputs runners may change (with
   limits) and the outputs they receive. The ids and labels you type are public.
3. Share with runners in the dialog (or later). They open `https://apps.example.com/`, sign in, and see their apps.
4. Publishing again creates a new release; earlier releases stay unchanged, and you can roll back.

## 7. Operations

- **Revoking:** remove a runner from an app, or disable the account. It takes effect on their next request; open
  progress streams close and download links stop working.
- **Audit log:** `data/audit.jsonl` records sign-ins, publishing, sharing, runs, cancels, downloads and revocations
  (never input values or the workflow). It rotates at 20 MB, keeping 10 old files.
- **Restarts:** everyone signs in again (sessions live in memory). Finished runs stay available until the retention
  ends; runs still in ComfyUI's queue are picked up again.
- **Backups:** back up the data folder. `secret.key` decrypts the stored releases: keep backups as private as the
  server. Without `secret.key` the stored releases cannot be read (publish again from the Studio).
- **Updating ComfyUI or nodes:** a release stores the API prompt. If a node changes its inputs, publish the app again.

## Limits of this version

- One worker per gateway; no GPU scheduling across several ComfyUI servers yet.
- Uploads accept PNG, JPEG and WebP images.
- Changing `secret.key` (key rotation) is not supported yet.
