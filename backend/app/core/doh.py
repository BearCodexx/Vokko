import socket
import urllib.request
import json
import ssl

_orig_getaddrinfo = socket.getaddrinfo

# динамический кэш разрешенных сетевых адресов
_dns_cache = {}

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# проверенные быстрые провайдеры DNS over HTTPS
_doh_providers = [
    "https://dns.adguard-dns.com/resolve?name={host}&type=A",
    "https://dns.alidns.com/resolve?name={host}&type=A",
    "https://dns.google/resolve?name={host}&type=A",
    "https://cloudflare-dns.com/dns-query?name={host}&type=A"
]

# резервное разрешение адресов серверов через защищенный протокол DoH
def _resolve_doh(host: str):
    if host in _dns_cache:
        return _dns_cache[host]

    for prov_url in _doh_providers:
        try:
            req = urllib.request.Request(
                prov_url.format(host=host),
                headers={"Accept": "application/dns-json", "User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=3, context=_ssl_ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for answer in data.get("Answer", []):
                    if answer.get("type") == 1 and answer.get("data"):
                        ip = answer["data"]
                        _dns_cache[host] = ip
                        return ip
        except Exception:
            continue
    return None

# перехват системного преобразования доменных имен для обхода блокировок
def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and not host.replace(".", "").isdigit():
        if host in _dns_cache:
            try:
                return _orig_getaddrinfo(_dns_cache[host], port, family, type, proto, flags)
            except Exception:
                pass
        
        # пробуем сначала системный DNS
        try:
            return _orig_getaddrinfo(host, port, family, type, proto, flags)
        except Exception:
            resolved_ip = _resolve_doh(host)
            if resolved_ip:
                return _orig_getaddrinfo(resolved_ip, port, family, type, proto, flags)
            raise

    return _orig_getaddrinfo(host, port, family, type, proto, flags)

# активация безопасного разрешения адресов
def setup_doh():
    socket.getaddrinfo = custom_getaddrinfo

setup_doh()
