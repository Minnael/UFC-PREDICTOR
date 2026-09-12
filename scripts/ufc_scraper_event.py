import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# Reutiliza o Playwright e as funções já prontas do scraper global
from ufc_scraper_global import get, get_fighter_profile, calc_stats

# ============================================================
# EVENTO A SER COLETADO
# Basta trocar a URL abaixo pelo evento desejado.
# Exemplo:
# http://ufcstats.com/event-details/f354c50b8d63d9b3
# ============================================================
EVENT_URL = "http://ufcstats.com/event-details/638cfec7ec559d6e"

print(f"Coletando evento: {EVENT_URL}")

# ── 1. Abre a página do evento ───────────────────────────────
soup = get(EVENT_URL)

# ── 2. Nomes + URLs dos lutadores ────────────────────────────
seen = set()
fighters = []

for a in soup.find_all("a", href=lambda h: h and "fighter-details" in h):
    href = a["href"].strip()
    name = " ".join(a.get_text().split())

    if href not in seen and name:
        seen.add(href)
        fighters.append({
            "name": name,
            "url": href,
        })

fights_list = [
    {
        "nome_a": fighters[i]["name"],
        "url_a": fighters[i]["url"],
        "nome_b": fighters[i + 1]["name"],
        "url_b": fighters[i + 1]["url"],
    }
    for i in range(0, len(fighters) - 1, 2)
]

print(f"{len(fights_list)} lutas encontradas\n")

# ── 3. Stats de cada lutador ─────────────────────────────────
fighter_cache = {}


def get_stats(url):
    if url not in fighter_cache:
        print(f"  Coletando: {url}")
        profile = get_fighter_profile(url)
        fighter_cache[url] = (
            calc_stats(profile, before=datetime.now()) if profile else {}
        )
    return fighter_cache[url]


# ── 4. Monta JSON ────────────────────────────────────────────
fights_json = []

for i, fight in enumerate(fights_list, 1):
    print(f"[{i}/{len(fights_list)}] {fight['nome_a']} vs {fight['nome_b']}")

    stats_a = get_stats(fight["url_a"])
    stats_b = get_stats(fight["url_b"])

    fights_json.append({
        fight["nome_a"]: stats_a,
        fight["nome_b"]: stats_b,
    })

# ── 5. Salva JSON ────────────────────────────────────────────
OUTPUT_FILE = "event_stats.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(fights_json, out, ensure_ascii=False, indent=2, default=str)

print(f"\nSalvo em {OUTPUT_FILE}")