import sys, os, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# Reutiliza o Playwright e as funções já prontas do scraper global
from ufc_scraper_global import get, get_fighter_profile, calc_stats, parse_date

# ── 1. Próximo evento ─────────────────────────────────────────
soup = get("http://ufcstats.com/statistics/events/upcoming?page=all")
event_url = None
for row in soup.find_all("tr", class_="b-statistics__table-row"):
    link = row.find("a", class_="b-link")
    span = row.find("span", class_="b-statistics__date")
    if link and span and parse_date(span.text):
        event_url = link["href"]
        print(f"Evento: {link.text.strip()} — {span.text.strip()}")
        break

soup = get(event_url)

# ── 2. Nomes + URLs dos lutadores ─────────────────────────────
seen = set()
fighters = []
for a in soup.find_all("a", href=lambda h: h and "fighter-details" in h):
    href = a["href"].strip()
    name = " ".join(a.get_text().split())
    if href not in seen and name:
        seen.add(href)
        fighters.append({"name": name, "url": href})

fights_list = [
    {"nome_a": fighters[i]["name"], "url_a": fighters[i]["url"],
     "nome_b": fighters[i+1]["name"], "url_b": fighters[i+1]["url"]}
    for i in range(0, len(fighters)-1, 2)
]
print(f"{len(fights_list)} lutas encontradas\n")

# ── 3. Stats de cada lutador ──────────────────────────────────
fighter_cache = {}

def get_stats(url):
    if url not in fighter_cache:
        print(f"  Coletando: {url}")
        profile = get_fighter_profile(url)
        # before=now: pega todo o histórico disponível (luta futura)
        fighter_cache[url] = calc_stats(profile, before=datetime.now()) if profile else {}
    return fighter_cache[url]

# ── 4. Monta JSON com nome como chave ─────────────────────────
fights_json = []
for i, f in enumerate(fights_list, 1):
    print(f"[{i}/{len(fights_list)}] {f['nome_a']} vs {f['nome_b']}")
    stats_a = get_stats(f["url_a"])
    stats_b = get_stats(f["url_b"])
    fights_json.append({
        f["nome_a"]: stats_a,
        f["nome_b"]: stats_b,
    })

# ── 5. Salva JSON ─────────────────────────────────────────────
with open("../next_event_stats.json", "w", encoding="utf-8") as out:
    json.dump(fights_json, out, ensure_ascii=False, indent=2, default=str)

print("\nSalvo em next_event_stats.json")