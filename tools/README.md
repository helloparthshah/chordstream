# Tools

- `lan_relay.mjs`: HTTPS + WebSocket relay from `0.0.0.0:<port>` to the chordstream server on
  `https://127.0.0.1:8000`. Use it when Windows Firewall blocks Python's inbound connections but
  another port or program is already allowed:
  `node tools/lan_relay.mjs data/certs/cert.pem data/certs/key.pem 8080`
- `allow-firewall.ps1`: run as Administrator to let the server accept connections on private
  networks (removes the Python block rules Windows created, opens TCP 8000 on Private only).
  It shows what it changes and asks first.
