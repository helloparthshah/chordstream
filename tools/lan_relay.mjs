// LAN relay: HTTPS on 0.0.0.0:<port> -> chordstream on https://127.0.0.1:8000 (HTTP + WebSocket).
// Exists because the firewall blocks python.exe inbound; node.exe on an already-allowed port is not.
import fs from 'node:fs';
import https from 'node:https';
import tls from 'node:tls';

const [certFile, keyFile, port = '8080'] = process.argv.slice(2);
const UP = { host: '127.0.0.1', port: 8000 };
const tlsOpts = { cert: fs.readFileSync(certFile), key: fs.readFileSync(keyFile) };

const server = https.createServer(tlsOpts, (req, res) => {
  const up = https.request(
    { ...UP, path: req.url, method: req.method, headers: req.headers, rejectUnauthorized: false },
    (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); },
  );
  up.on('error', (e) => { res.writeHead(502); res.end(`upstream error: ${e.message}`); });
  req.pipe(up);
});

// WebSocket: replay the upgrade request upstream, then splice the two sockets together.
server.on('upgrade', (req, sock, head) => {
  const up = tls.connect({ ...UP, rejectUnauthorized: false }, () => {
    const lines = [`${req.method} ${req.url} HTTP/1.1`];
    for (let i = 0; i < req.rawHeaders.length; i += 2) lines.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
    up.write(lines.join('\r\n') + '\r\n\r\n');
    if (head && head.length) up.write(head);
    up.pipe(sock); sock.pipe(up);
  });
  const close = () => { sock.destroy(); up.destroy(); };
  up.on('error', close); sock.on('error', close);
});

server.listen(Number(port), '0.0.0.0', () => console.log(`relay listening on https://0.0.0.0:${port} -> 127.0.0.1:8000`));
