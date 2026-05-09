"""
UFC Stats Scraper v3 — seletores baseados no HTML real inspecionado.

Uso:
    python ufc_scraper_v3.py --test            # 3 eventos
    python ufc_scraper_v3.py --max-events 50   # N eventos
    python ufc_scraper_v3.py                   # todos (horas)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re, logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE    = "http://ufcstats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
DELAY = 1.5


# ─────────────────────────────────────────────────────────────
# Helpers
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
    """Retorna textos de todos os <p> dentro de um <td>."""
    return [clean(p.get_text()) for p in td.find_all("p", class_="b-fight-details__table-text")]

def fight_seconds(time_str: str, rnd) -> float:
    try:
        parts = str(time_str).split(":")
        mins, secs = int(parts[0]), int(parts[1])
        prev = max(0, int(rnd)-1) if rnd else 0
        return prev*300 + mins*60 + secs
    except: return 0


# ─────────────────────────────────────────────────────────────
# 1. Eventos
# ─────────────────────────────────────────────────────────────

def get_all_events() -> list[dict]:
    soup = get(f"{BASE}/statistics/events/completed?page=all")
    if not soup: return []

    events = []
    for row in soup.find_all("tr", class_="b-statistics__table-row"):
        link = row.find("a", class_="b-link")
        span = row.find("span", class_="b-statistics__date")
        if not link or not span: continue
        date = parse_date(span.text)
        if date:
            events.append({"name": clean(link.text), "url": link["href"], "date": date})

    log.info(f"Eventos encontrados: {len(events)}")
    return events


# ─────────────────────────────────────────────────────────────
# 2. Lutas de um evento
# ─────────────────────────────────────────────────────────────

def get_fights_from_event(event: dict) -> list[dict]:
    soup = get(event["url"])
    if not soup: return []

    fights = []
    # Cada luta é uma <tr> com data-link
    for row in soup.find_all("tr", attrs={"data-link": True}):
        fight_url = row["data-link"].strip()
        if "fight-details" not in fight_url: continue
        fights.append({
            "event_name": event["name"],
            "event_date": event["date"],
            "fight_url":  fight_url,
        })

    log.info(f"  Lutas: {len(fights)}")
    return fights


# ─────────────────────────────────────────────────────────────
# 3. Detalhes de uma luta
# ─────────────────────────────────────────────────────────────

def get_fight_details(fight_url: str) -> Optional[dict]:
    soup = get(fight_url)
    if not soup: return None

    d = {}

    # Nomes e links dos lutadores
    person_divs = soup.find_all("div", class_="b-fight-details__person")
    if len(person_divs) < 2: return None

    names, urls_f, results_raw = [], [], []
    for div in person_divs[:2]:
        a = div.find("a")
        names.append(clean(a.text) if a else "")
        urls_f.append(a["href"] if a else "")
        status = div.find("i", class_=lambda c: c and "b-fight-details__person-status" in c)
        results_raw.append(clean(status.text) if status else "")

    if len(names) < 2: return None

    d["fighter_a_name"] = names[0]
    d["fighter_b_name"] = names[1]
    d["fighter_a_url"]  = urls_f[0]
    d["fighter_b_url"]  = urls_f[1]

    if results_raw[0] == "W":       d["winner"] = "A"
    elif results_raw[1] == "W":     d["winner"] = "B"
    else:                           d["winner"] = "NC"

    # Método, round, tempo
    for p in soup.find_all("p", class_="b-fight-details__text"):
        txt = clean(p.get_text(separator=" "))
        if "Method:" in txt:
            raw = txt.split("Method:")[-1].strip()
            d["method"] = re.split(r"\s{2,}|Round:|Time:|Referee:", raw)[0].strip()
        if "Round:" in txt:
            m = re.search(r"Round:\s*(\d+)", txt)
            d["round"] = int(m.group(1)) if m else None
        if "Time:" in txt:
            m = re.search(r"Time:\s*([\d:]+)", txt)
            d["time"] = m.group(1) if m else None

    return d


# ─────────────────────────────────────────────────────────────
# 4. Perfil do lutador
# ─────────────────────────────────────────────────────────────

def get_fighter_profile(url: str) -> Optional[dict]:
    soup = get(url)
    if not soup: return None

    p = {"url": url}

    # Nome
    name_tag = soup.find("span", class_="b-content__title-highlight")
    p["name"] = clean(name_tag.text) if name_tag else ""

    # Todos os <li> com classe b-list__box-list-item_type_block
    # contêm TANTO atributos físicos QUANTO stats de carreira
    for li in soup.find_all("li", class_="b-list__box-list-item_type_block"):
        txt = clean(li.get_text(separator=" "))

        if txt.startswith("Height:"):
            p["height_cm"]    = parse_height(txt.replace("Height:","").strip())
        elif txt.startswith("Weight:"):
            p["weight_kg"]    = parse_weight(txt.replace("Weight:","").strip())
        elif txt.startswith("Reach:"):
            p["reach_arm_cm"] = parse_reach(txt.replace("Reach:","").strip())
        elif txt.upper().startswith("STANCE:"):
            p["stance"]       = txt.split(":",1)[-1].strip()
        elif txt.startswith("DOB:"):
            p["dob"]          = parse_date(txt.replace("DOB:","").strip())


    # Histórico de lutas
    p["fight_history"] = _parse_fight_history(soup, url)
    return p


def _parse_fight_history(soup: BeautifulSoup, fighter_url: str) -> list[dict]:
    """
    Mapeamento REAL confirmado via debug (Carlos Prates):
      td[0]: resultado   flag win/loss/draw/nc
      td[1]: nomes       [este lutador, adversário]
      td[2]: KD          [este, adversário]
      td[3]: Sig.Str.hit [este, adversário]  ex: "104", "41"
      td[4]: TD hit      [este, adversário]
      td[5]: Sub.Att     [este, adversário]
      td[6]: Evento + Data  ex: "UFC Fight Night...", "May. 02, 2026"
      td[7]: Método      ex: "KO/TKO", "Decision - Unanimous"
      td[8]: Round       ex: "3"
      td[9]: Tempo       ex: "3:17"
    """
    history = []

    for row in soup.find_all("tr", attrs={"data-link": True}):
        tds = row.find_all("td")
        if len(tds) < 10:
            continue

        # Resultado
        flag = tds[0].find("i", class_="b-flag__text")
        result_txt = clean(flag.get_text()) if flag else ""
        result_map = {"win": "W", "loss": "L", "draw": "D", "nc": "NC"}
        result = result_map.get(result_txt.lower(), result_txt.upper())

        # KD — td[2], primeiro <p> = este lutador
        kd_ps = p_texts(tds[2])
        kd = pf(kd_ps[0]) if kd_ps else None

        # Sig. Strikes hit — td[3]
        sig_ps = p_texts(tds[3])
        sig_hit = pf(sig_ps[0]) if sig_ps else None

        # TD hit — td[4]
        td_ps = p_texts(tds[4])
        td_hit = pf(td_ps[0]) if td_ps else None

        # Sub. Att — td[5]
        sub_ps = p_texts(tds[5])
        sub_att_n = pf(sub_ps[0]) if sub_ps else None

        # Data — td[6], segundo <p>
        date = None
        date_ps = p_texts(tds[6])
        if len(date_ps) >= 2:
            date = parse_date(date_ps[1])

        if not date:
            continue

        # Método — td[7], primeiro <p>
        method_ps = p_texts(tds[7])
        method = method_ps[0] if method_ps else ""

        # Round — td[8]
        round_ps = p_texts(tds[8])
        rnd = None
        if round_ps:
            try: rnd = int(round_ps[0])
            except: pass

        # Tempo — td[9]
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
# 5. Stats acumuladas até uma data
# ─────────────────────────────────────────────────────────────

def calc_stats(profile: dict, before: datetime) -> dict:
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
        "sig_strikes_hit_total",
        "td_hit_total",
        "sub_att_total",
        "kd_total",
        "sig_strikes_per_min",
        "td_per_15min",
        "sub_per_15min",
        "avg_fight_time_sec",
    ]

    if not hist:
        for k in NULLS: s[k] = None
        return s

    total_sec = sum(fight_seconds(f["time"], f["round"]) for f in hist)
    total_min = max(total_sec / 60, 0.01)

    sig_h  = sum(f["sig_strikes_hit"] or 0 for f in hist)
    td_h   = sum(f["td_hit"]          or 0 for f in hist)
    sub    = sum(f["sub_att"]         or 0 for f in hist)
    kd     = sum(f["kd"]              or 0 for f in hist)

    s["sig_strikes_hit_total"]   = sig_h
    s["td_hit_total"]            = td_h
    s["sub_att_total"]           = sub
    s["kd_total"]                = kd
    s["sig_strikes_per_min"]     = round(sig_h/total_min, 3)
    s["td_per_15min"]            = round(td_h/total_min*15, 3)
    s["sub_per_15min"]           = round(sub/total_min*15, 3)
    s["avg_fight_time_sec"]      = round(total_sec/len(hist), 1)

    return s


# ─────────────────────────────────────────────────────────────
# 6. Pipeline principal
# ─────────────────────────────────────────────────────────────

def build_dataset(max_events: int = None, output: str = "ufc_dataset.csv"):
    fighter_cache: dict[str, dict] = {}
    events = get_all_events()
    if not events:
        log.error("Nenhum evento carregado.")
        return

    if max_events:
        events = events[:max_events]

    rows = []

    for ei, event in enumerate(events):
        log.info(f"[{ei+1}/{len(events)}] {event['name']} — {event['date'].date()}")
        time.sleep(DELAY)

        fights = get_fights_from_event(event)
        if not fights: continue

        for fight in fights:
            time.sleep(DELAY)
            details = get_fight_details(fight["fight_url"])
            if not details: continue

            winner = details.get("winner","")
            if winner not in ("A","B"):
                log.info(f"    Pulando (NC/Draw)")
                continue

            def load(url, name):
                if not url: return None
                if url not in fighter_cache:
                    log.info(f"    Perfil: {name}")
                    time.sleep(DELAY)
                    fighter_cache[url] = get_fighter_profile(url)
                return fighter_cache.get(url)

            pa = load(details.get("fighter_a_url"), details.get("fighter_a_name"))
            pb = load(details.get("fighter_b_url"), details.get("fighter_b_name"))
            if not pa or not pb: continue

            sa = calc_stats(pa, before=event["date"])
            sb = calc_stats(pb, before=event["date"])

            row = {
                "event_name": event["name"],
                "event_date": event["date"].date(),
                "fighter_a":  details["fighter_a_name"],
                "fighter_b":  details["fighter_b_name"],
                "winner":     winner,
                "label":      1 if winner == "A" else 0,
                "method":     details.get("method",""),
                "round":      details.get("round"),
                "time":       details.get("time",""),
            }
            for k, v in sa.items(): row[f"a_{k}"] = v
            for k, v in sb.items(): row[f"b_{k}"] = v

            rows.append(row)
            log.info(f"    ✓ {details['fighter_a_name']} vs {details['fighter_b_name']} → {winner}")

        if rows:
            pd.DataFrame(rows).to_csv(output, index=False)
            log.info(f"  💾 {len(rows)} lutas salvas em '{output}'")

    pd.DataFrame(rows).to_csv(output, index=False)
    log.info(f"\n✅ Concluído! {len(rows)} lutas → '{output}'")


# ─────────────────────────────────────────────────────────────
# 7. Entrada
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UFC Scraper v3")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--output", type=str, default="ufc_dataset.csv")
    parser.add_argument("--test", action="store_true", help="Apenas 3 eventos")
    args = parser.parse_args()

    build_dataset(
        max_events=3 if args.test else args.max_events,
        output=args.output
    )