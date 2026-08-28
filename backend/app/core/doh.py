import socket
import urllib.request
import json
import ssl

_orig_getaddrinfo = socket.getaddrinfo

# предварительно заполненная таблица адресов для гарантированного доступа
_dns_cache = {
    "www.youtube.com": "142.251.152.4",
    "youtube.com": "142.251.152.4",
    "youtu.be": "142.251.152.4",
    "m.youtube.com": "142.251.152.4",
    "i.ytimg.com": "142.251.152.4",
    "yt3.ggpht.com": "142.251.152.4",
}

# резервное разрешение адресов серверов через защищенный протокол
def _resolve_doh(host: str):
    if host in _dns_cache:
        return _dns_cache[host]

    providers = [
        f"https://1.1.1.1/dns-query?name={host}&type=A",
        f"https://8.8.8.8/resolve?name={host}&type=A",
        f"https://9.9.9.9/dns-query?name={host}&type=A"
    ]

    ctx = ssl._create_unverified_context()
    for prov_url in providers:
        try:
            req = urllib.request.Request(
                prov_url,
                headers={"Accept": "application/dns-json", "User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=3, context=ctx) as resp:
                data = json.loads(resp.read().decode())
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
    if isinstance(host, str) and host in _dns_cache:
        try:
            return _orig_getaddrinfo(_dns_cache[host], port, family, type, proto, flags)
        except Exception:
            pass

    try:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)
    except Exception:
        if isinstance(host, str) and not host.replace(".", "").isdigit():
            resolved_ip = _resolve_doh(host)
            if resolved_ip:
                return _orig_getaddrinfo(resolved_ip, port, family, type, proto, flags)
        raise

# активация безопасного разрешения адресов
def setup_doh():
    socket.getaddrinfo = custom_getaddrinfo

setup_doh()
