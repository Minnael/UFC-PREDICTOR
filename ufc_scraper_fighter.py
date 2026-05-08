"""
UFC Fighter Stats Scraper — Obtém os dados atualizados de um lutador específico.

Uso:
    python ufc_fighter_scraper.py --url "http://ufcstats.com/fighter-details/..."
"""

import requests
from bs4 import BeautifulSoup
import time, re, logging
from datetime import datetime
from typing import Optional
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ─────────────────────────────────────────────────────────────
# Helpers de Parsing
# ─────────────────────────────────────────────────────────────

def get(url: str) -> Optional[BeautifulSoup]:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log.warning(f"[{attempt+1}/3] {url} — {e}")
            time.sleep(3)
    return None

def clean(t: str) -> str:
    return " ".join(t.split())

def pf(v: str) -> Optional[float]:
    v = clean(str(v)).replace("%","").replace("--","").replace("---","")
    try: return float(v) if v else None
    except: return None

def parse_height(v: str) -> Optional[float]:
    m = re.match(r"(\d+)'\s*(\d+)\"", v.strip())
    return round(int(m.group(1))*30.48 + int(m.group(2))*2.54, 1) if m else None

def parse_reach(v: str) -> Optional[float]:
    m = re.search(r"([\d.]+)\"", v.strip())
    return round(float(m.group(1))*2.54, 1) if m else None

def parse_weight(v: str) -> Optional[float]:
    m = re.search(r"([\d.]+)\s*lbs", v.strip())
    return round(float(m.group(1))*0.453592, 1) if m else None

def parse_date(v: str) -> Optional[datetime]:
    for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y", "%b. %d,%Y"):
        try: return datetime.strptime(clean(v), fmt)
        except: pass
    return None

def p_texts(td) -> list[str]:
    return [clean(p.get_text()) for p in td.find_all("p", class_="b-fight-details__table-text")]

def fight_seconds(time_str: str, rnd) -> float:
    try:
        parts = str(time_str).split(":")
        mins, secs = int(parts[0]), int(parts[1])
        prev = max(0, int(rnd)-1) if rnd else 0
        return prev*300 + mins*60 + secs
    except: return 0


# ─────────────────────────────────────────────────────────────
# Perfil e Histórico do Lutador
# ─────────────────────────────────────────────────────────────

def get_fighter_profile(url: str) -> Optional[dict]:
    soup = get(url)
    if not soup: return None

    p = {"url": url}

    name_tag = soup.find("span", class_="b-content__title-highlight")
    p["name"] = clean(name_tag.text) if name_tag else "Unknown Fighter"

    for li in soup.find_all("li", class_="b-list__box-list-item_type_block"):
        txt = clean(li.get_text(separator=" "))

        if txt.startswith("Height:"):
            p["height_cm"] = parse_height(txt.replace("Height:","").strip())
        elif txt.startswith("Weight:"):
            p["weight_kg"] = parse_weight(txt.replace("Weight:","").strip())
        elif txt.startswith("Reach:"):
            p["reach_arm_cm"] = parse_reach(txt.replace("Reach:","").strip())
        elif txt.upper().startswith("STANCE:"):
            p["stance"] = txt.split(":",1)[-1].strip()
        elif txt.startswith("DOB:"):
            p["dob"] = parse_date(txt.replace("DOB:","").strip())

    p["fight_history"] = _parse_fight_history(soup)
    return p


def _parse_fight_history(soup: BeautifulSoup) -> list[dict]:
    history = []
    for row in soup.find_all("tr", attrs={"data-link": True}):
        tds = row.find_all("td")
        if len(tds) < 10:
            continue

        flag = tds[0].find("i", class_="b-flag__text")
        result_txt = clean(flag.get_text()) if flag else ""
        result_map = {"win": "W", "loss": "L", "draw": "D", "nc": "NC"}
        result = result_map.get(result_txt.lower(), result_txt.upper())

        kd_ps = p_texts(tds[2])
        kd = pf(kd_ps[0]) if kd_ps else None

        sig_ps = p_texts(tds[3])
        sig_hit = pf(sig_ps[0]) if sig_ps else None

        td_ps = p_texts(tds[4])
        td_hit = pf(td_ps[0]) if td_ps else None

        sub_ps = p_texts(tds[5])
        sub_att_n = pf(sub_ps[0]) if sub_ps else None

        date = None
        date_ps = p_texts(tds[6])
        if len(date_ps) >= 2:
            date = parse_date(date_ps[1])

        if not date:
            continue

        method_ps = p_texts(tds[7])
        method = method_ps[0] if method_ps else ""

        round_ps = p_texts(tds[8])
        rnd = None
        if round_ps:
            try: rnd = int(round_ps[0])
            except: pass

        time_ps = p_texts(tds[9])
        fight_time = time_ps[0] if time_ps else ""

        history.append({
            "date":            date,
            "result":          result,
            "method":          method,
            "round":           rnd,
            "time":            fight_time,
            "kd":              kd,
            "sig_strikes_hit": sig_hit,
            "td_hit":          td_hit,
            "sub_att":         sub_att_n,
        })

    return sorted(history, key=lambda x: x["date"])


# ─────────────────────────────────────────────────────────────
# Cálculo de Stats
# ─────────────────────────────────────────────────────────────

def calc_stats(profile: dict) -> dict:
    # Como queremos até hoje, usamos o momento atual
    before = datetime.now()
    hist = [f for f in profile.get("fight_history", []) if f["date"] < before]

    s = {
        "height_cm":    profile.get("height_cm"),
        "weight_kg":    profile.get("weight_kg"),
        "reach_arm_cm": profile.get("reach_arm_cm"),
        "stance":       profile.get("stance"),
    }

    dob = profile.get("dob")
    s["age"] = round((before - dob).days / 365.25, 1) if isinstance(dob, datetime) else None

    wins   = [f for f in hist if f["result"] == "W"]
    losses = [f for f in hist if f["result"] == "L"]

    s["wins"]        = len(wins)
    s["losses"]      = len(losses)
    s["draws"]       = len([f for f in hist if f["result"] == "D"])
    s["wins_ko"]     = sum(1 for f in wins if re.search(r"KO|TKO", f.get("method",""), re.I))
    s["wins_sub"]    = sum(1 for f in wins if re.search(r"Sub",    f.get("method",""), re.I))
    s["wins_dec"]    = sum(1 for f in wins if re.search(r"Dec",    f.get("method",""), re.I))
    s["total_fights"]= len(hist)

    NULLS = [
        "sig_strikes_hit_total", "td_hit_total", "sub_att_total", "kd_total",
        "sig_strikes_per_min", "td_per_15min", "sub_per_15min", "avg_fight_time_sec"
    ]

    if not hist:
        for k in NULLS: s[k] = None
        return s

    total_sec = sum(fight_seconds(f["time"], f["round"]) for f in hist)
    total_min = max(total_sec / 60, 0.01)

    sig_h = sum(f["sig_strikes_hit"] or 0 for f in hist)
    td_h  = sum(f["td_hit"]          or 0 for f in hist)
    sub   = sum(f["sub_att"]         or 0 for f in hist)
    kd    = sum(f["kd"]              or 0 for f in hist)

    s["sig_strikes_hit_total"] = sig_h
    s["td_hit_total"]          = td_h
    s["sub_att_total"]         = sub
    s["kd_total"]              = kd
    s["sig_strikes_per_min"]   = round(sig_h/total_min, 3)
    s["td_per_15min"]          = round(td_h/total_min*15, 3)
    s["sub_per_15min"]         = round(sub/total_min*15, 3)
    s["avg_fight_time_sec"]    = round(total_sec/len(hist), 1)

    return s


# ─────────────────────────────────────────────────────────────
# Gerador de Dicionário em Python
# ─────────────────────────────────────────────────────────────

def print_python_dict(name: str, stats: dict):
    print(f"\nFIGHTER_NOME = \"{name}\"\n")
    print("FIGHTER = {")
    for key, val in stats.items():
        if val is None:
            formatted_val = "None"
        elif isinstance(val, str):
            formatted_val = f'"{val}"'
        else:
            formatted_val = str(val)
        
        print(f'    "{key}": {formatted_val:<10},')
    print("}")


# ─────────────────────────────────────────────────────────────
# Execução Principal
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pega stats mais atualizados de um lutador específico")
    parser.add_argument("--url", type=str, required=True, help="URL do lutador no ufcstats.com")
    args = parser.parse_args()

    log.info(f"Buscando perfil: {args.url}")
    
    profile = get_fighter_profile(args.url)
    if not profile:
        log.error("Não foi possível acessar ou processar o perfil do lutador.")
        exit(1)
        
    stats = calc_stats(profile)
    log.info(f"Dados calculados com sucesso para {profile['name']}!")
    
    print_python_dict(profile['name'], stats)