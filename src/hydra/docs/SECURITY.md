# Security notes

- Zero network calls and no telemetry; a test blocks sockets. Laptop mode binds to loopback only.
- Server mode: scrypt password hashes, roles (viewer / researcher / admin), HMAC-signed expiring tokens, login lockout, audit log. The signing key `secret.key` is chmod 600 and excluded from backups.
- 50 MB request cap, security headers, sanitised attachment filenames, content-addressed attachments with integrity checks, path-traversal guard on restore.
- Jobs in server mode run in forked children with RAM (RLIMIT_AS) and CPU-affinity limits.
- Put a TLS-terminating reverse proxy in front of server mode; HYDRA does not terminate TLS.
