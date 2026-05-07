"""
UFC Stats Scraper v2 — ufcstats.com
Seletores CSS corrigidos com base em projetos reais que funcionam.

Uso:
    python ufc_scraper_v2.py --test          # 3 eventos apenas
    python ufc_scraper_v2.py --max-events 20 # N eventos
    python ufc_scraper_v2.py                 # todos os eventos (horas)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE    = "http://ufcstats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
DELAY = 1.5  # segundos entre requisições


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
            log.warning(f"[{attempt+1}/3] Falhou: {url} — {e}")
            time.sleep(3)
    log.error(f"Desistindo de: {url}")
    return None


def clean(text: str) -> str:
    return " ".join(text.split())


def pf(val: str) -> Optional[float]:
    """Parse float — remove %, --, espaços."""
    v = clean(val).replace("%", "").replace("--", "").replace("---", "")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def parse_height(val: str) -> Optional[float]:
    """'6\' 1\"' → cm"""
    m = re.match(r"(\d+)'\s*(\d+)\"", val.strip())
    if m:
        return round(int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54, 1)
    return None


def parse_reach(val: str) -> Optional[float]:
    """'72\"' → cm"""
    m = re.search(r"([\d.]+)\"", val.strip())
    return round(float(m.group(1)) * 2.54, 1) if m else None


def parse_weight(val: str) -> Optional[float]:
    """'155 lbs.' → kg"""
    m = re.search(r"([\d.]+)\s*lbs", val.strip())
    return round(float(m.group(1)) * 0.453592, 1) if m else None


def parse_date(val: str) -> Optional[datetime]:
    val = val.strip()
    for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    return None


def split_of(val: str):
    """'10 of 25' → (10.0, 25.0)"""
    m = re.match(r"(\d+)\s+of\s+(\d+)", val.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ─────────────────────────────────────────────────────────────
# 1. Lista de eventos
# ─────────────────────────────────────────────────────────────

def get_all_events() -> list[dict]:
    url  = f"{BASE}/statistics/events/completed?page=all"
    soup = get(url)
    if not soup:
        log.error("Não conseguiu carregar a página de eventos.")
        return []

    events = []
    # Cada evento é um <tr> com classe b-statistics__table-row
    rows = soup.find_all("tr", class_="b-statistics__table-row")
    log.info(f"Linhas encontradas na tabela de eventos: {len(rows)}")

    for row in rows:
        # Link está dentro de <i class="b-statistics__table-content"> > <a>
        link = row.find("a", class_="b-link")
        if not link:
            continue
        name = clean(link.text)
        href = link.get("href", "")

        # Data está em <span class="b-statistics__date">
        span = row.find("span", class_="b-statistics__date")
        date_str = clean(span.text) if span else ""
        date     = parse_date(date_str)

        if name and href and date:
            events.append({"name": name, "url": href, "date": date})

    log.info(f"Total de eventos válidos: {len(events)}")
    return events


# ─────────────────────────────────────────────────────────────
# 2. Lutas de um evento
# ─────────────────────────────────────────────────────────────

def get_fights_from_event(event: dict) -> list[dict]:
    soup = get(event["url"])
    if not soup:
        return []

    fights = []
    # Linhas de luta têm a classe js-fight-details-click e data-link
    rows = soup.find_all(
        "tr",
        class_=lambda c: c and "js-fight-details-click" in c
    )
    log.info(f"  Lutas encontradas: {len(rows)}")

    for row in rows:
        fight_url = row.get("data-link", "").strip()
        if not fight_url:
            continue
        fights.append({
            "event_name": event["name"],
            "event_date": event["date"],
            "fight_url":  fight_url,
        })

    return fights


# ─────────────────────────────────────────────────────────────
# 3. Detalhes de uma luta
# ─────────────────────────────────────────────────────────────

def get_fight_details(fight_url: str) -> Optional[dict]:
    soup = get(fight_url)
    if not soup:
        return None

    d = {}

    # ── Nomes e links dos lutadores ──────────────────────────
    person_divs = soup.find_all("div", class_="b-fight-details__person")
    if len(person_divs) < 2:
        return None

    names, urls_f, results_raw = [], [], []
    for div in person_divs[:2]:
        a = div.find("a", class_="b-fight-details__person-link") or div.find("a")
        if a:
            names.append(clean(a.text))
            urls_f.append(a.get("href", ""))
        status = div.find("i", class_=lambda c: c and "b-fight-details__person-status" in c)
        results_raw.append(clean(status.text) if status else "")

    if len(names) < 2:
        return None

    d["fighter_a_name"] = names[0]
    d["fighter_b_name"] = names[1]
    d["fighter_a_url"]  = urls_f[0] if urls_f else None
    d["fighter_b_url"]  = urls_f[1] if len(urls_f) > 1 else None
    d["result_a"]       = results_raw[0]
    d["result_b"]       = results_raw[1] if len(results_raw) > 1 else ""

    if results_raw[0] == "W":
        d["winner"] = "A"
    elif len(results_raw) > 1 and results_raw[1] == "W":
        d["winner"] = "B"
    else:
        d["winner"] = results_raw[0] if results_raw else "NC"

    # ── Método, round, tempo ─────────────────────────────────
    for p in soup.find_all("p", class_="b-fight-details__text"):
        text = clean(p.get_text(separator=" "))
        if "Method:" in text:
            d["method"] = text.split("Method:")[-1].strip().split()[0:3]
            d["method"] = " ".join(d["method"])
        if "Round:" in text:
            m = re.search(r"Round:\s*(\d+)", text)
            d["round"] = int(m.group(1)) if m else None
        if "Time:" in text:
            m = re.search(r"Time:\s*([\d:]+)", text)
            d["time"] = m.group(1) if m else None

    # ── Stats totais da luta (tabela de totals) ──────────────
    tables = soup.find_all("table", class_=lambda c: c and "b-fight-details__table" in c)
    if tables:
        headers = [clean(th.text) for th in tables[0].find_all("th")]
        body_rows = tables[0].find_all(
            "tr",
            class_=lambda c: c and "b-fight-details__table-row" in c
        )
        for ri, brow in enumerate(body_rows[:2]):
            prefix = "a_" if ri == 0 else "b_"
            cells  = [clean(td.get_text(separator=" ")) for td in brow.find_all("td")]
            for h, v in zip(headers, cells):
                key = re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
                d[f"{prefix}fight_{key}"] = v

    return d


# ─────────────────────────────────────────────────────────────
# 4. Perfil do lutador
# ─────────────────────────────────────────────────────────────

def get_fighter_profile(url: str) -> Optional[dict]:
    soup = get(url)
    if not soup:
        return None

    p = {"url": url}

    # Nome
    name_tag = soup.find("span", class_="b-content__title-highlight")
    p["name"] = clean(name_tag.text) if name_tag else ""

    # Atributos físicos — lista de itens b-list__box-list-item
    for li in soup.find_all("li", class_="b-list__box-list-item"):
        text = clean(li.get_text(separator=" "))
        if text.startswith("Height:"):
            p["height_cm"] = parse_height(text[7:].strip())
        elif text.startswith("Weight:"):
            p["weight_kg"] = parse_weight(text[7:].strip())
        elif text.startswith("Reach:"):
            p["reach_arm_cm"] = parse_reach(text[6:].strip())
        elif text.startswith("STANCE:") or text.startswith("Stance:"):
            p["stance"] = text.split(":", 1)[-1].strip()
        elif text.startswith("DOB:"):
            p["dob"] = parse_date(text[4:].strip())

    # Stats de carreira (bloco de ícones numéricos no topo do perfil)
    career_items = soup.find_all("li", class_="b-list__box-list-item_type_block")
    for li in career_items:
        label_tag = li.find("i", class_="b-list__box-item-title")
        if label_tag:
            label = clean(label_tag.text).rstrip(":")
            value = clean(li.get_text(separator=" ")).replace(label, "").replace(":", "").strip()
            key   = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            p[f"career_{key}"] = pf(value)

    # Histórico de lutas
    p["fight_history"] = _parse_fight_history(soup)
    return p


def _parse_fight_history(soup: BeautifulSoup) -> list[dict]:
    history = []
    rows = soup.find_all(
        "tr",
        class_=lambda c: c and "b-fight-details__table-row__hover" in c
    )
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        def cell(i): return clean(cells[i].get_text(separator=" ")) if len(cells) > i else ""

        result   = cell(0)
        opponent = cell(1)
        event_td = cells[2] if len(cells) > 2 else None

        # Data — procura padrão "Month DD, YYYY" dentro da célula do evento
        date = None
        if event_td:
            for txt in event_td.stripped_strings:
                d = parse_date(txt)
                if d:
                    date = d
                    break

        if not date:
            continue

        kd       = pf(cell(4))
        sig_h, sig_a = split_of(cell(5))
        td_h,  td_a  = split_of(cell(7))
        sub_att  = pf(cell(9))
        method   = cell(8)
        rnd      = pf(cell(10))
        fight_time = cell(11)

        history.append({
            "date":             date,
            "result":           result,
            "opponent":         opponent,
            "method":           method,
            "round":            rnd,
            "time":             fight_time,
            "kd":               kd,
            "sig_strikes_hit":  sig_h,
            "sig_strikes_att":  sig_a,
            "td_hit":           td_h,
            "td_att":           td_a,
            "sub_att":          sub_att,
        })

    return sorted(history, key=lambda x: x["date"])


# ─────────────────────────────────────────────────────────────
# 5. Stats acumuladas até uma data
# ─────────────────────────────────────────────────────────────

def _fight_seconds(time_str: str, rnd) -> float:
    try:
        parts = str(time_str).split(":")
        mins, secs = int(parts[0]), int(parts[1])
        prev = max(0, int(rnd) - 1) if rnd else 0
        return prev * 300 + mins * 60 + secs
    except Exception:
        return 0


def calc_stats(profile: dict, before: datetime) -> dict:
    hist = [f for f in profile.get("fight_history", []) if f["date"] < before]

    s = {
        "height_cm":    profile.get("height_cm"),
        "weight_kg":    profile.get("weight_kg"),
        "reach_arm_cm": profile.get("reach_arm_cm"),
        "stance":       profile.get("stance"),
    }

    # Idade
    dob = profile.get("dob")
    s["age"] = round((before - dob).days / 365.25, 1) if isinstance(dob, datetime) else None

    wins   = [f for f in hist if f["result"].upper() == "W"]
    losses = [f for f in hist if f["result"].upper() == "L"]

    s["wins"]     = len(wins)
    s["losses"]   = len(losses)
    s["draws"]    = len(hist) - len(wins) - len(losses)
    s["wins_ko"]  = sum(1 for f in wins if re.search(r"KO|TKO", f.get("method",""), re.I))
    s["wins_sub"] = sum(1 for f in wins if re.search(r"SUB",     f.get("method",""), re.I))
    s["wins_dec"] = sum(1 for f in wins if re.search(r"DEC",     f.get("method",""), re.I))
    s["total_fights"] = len(hist)

    NONE_KEYS = [
        "sig_strikes_hit_total", "sig_strikes_att_total", "sig_strike_acc",
        "td_hit_total", "td_att_total", "td_acc",
        "sub_att_total", "kd_total",
        "sig_strikes_per_min", "td_per_15min", "sub_per_15min",
        "avg_fight_time_sec",
        # stats de defesa vêm do perfil de carreira
        "career_sig_str_absorbed_per_min",
        "career_sig_str_defense",
        "career_td_defense",
        "career_avg_td_per_15_min",
    ]

    if not hist:
        for k in NONE_KEYS:
            s[k] = None
        return s

    total_sec = sum(_fight_seconds(f["time"], f["round"]) for f in hist)
    total_min = max(total_sec / 60, 0.01)

    sig_h = sum(f["sig_strikes_hit"] or 0 for f in hist)
    sig_a = sum(f["sig_strikes_att"] or 0 for f in hist)
    td_h  = sum(f["td_hit"]          or 0 for f in hist)
    td_a  = sum(f["td_att"]          or 0 for f in hist)
    sub   = sum(f["sub_att"]         or 0 for f in hist)
    kd    = sum(f["kd"]              or 0 for f in hist)

    s["sig_strikes_hit_total"] = sig_h
    s["sig_strikes_att_total"] = sig_a
    s["sig_strike_acc"]        = round(sig_h / sig_a * 100, 1) if sig_a else None
    s["td_hit_total"]          = td_h
    s["td_att_total"]          = td_a
    s["td_acc"]                = round(td_h / td_a * 100, 1) if td_a else None
    s["sub_att_total"]         = sub
    s["kd_total"]              = kd
    s["sig_strikes_per_min"]   = round(sig_h / total_min, 3)
    s["td_per_15min"]          = round(td_h  / total_min * 15, 3)
    s["sub_per_15min"]         = round(sub   / total_min * 15, 3)
    s["avg_fight_time_sec"]    = round(total_sec / len(hist), 1)

    # Stats de defesa: vêm do perfil de carreira do ufcstats
    s["career_sig_str_absorbed_per_min"] = profile.get("career_sig_str_absorbed_per_min")
    s["career_sig_str_defense"]          = profile.get("career_sig_str_defense")
    s["career_td_defense"]               = profile.get("career_td_defense")
    s["career_avg_td_per_15_min"]        = profile.get("career_avg_td_per_15_min")

    return s


# ─────────────────────────────────────────────────────────────
# 6. Pipeline principal
# ─────────────────────────────────────────────────────────────

def build_dataset(max_events: int = None, output: str = "ufc_dataset.csv"):
    fighter_cache: dict[str, dict] = {}
    events = get_all_events()

    if not events:
        log.error("Nenhum evento carregado. Verifique sua conexão.")
        return

    if max_events:
        events = events[:max_events]

    rows = []

    for ei, event in enumerate(events):
        log.info(f"[{ei+1}/{len(events)}] {event['name']} — {event['date'].date()}")
        time.sleep(DELAY)

        fights = get_fights_from_event(event)
        if not fights:
            log.info("  Nenhuma luta encontrada, pulando.")
            continue

        for fight in fights:
            time.sleep(DELAY)
            details = get_fight_details(fight["fight_url"])
            if not details:
                continue

            winner = details.get("winner", "")
            if winner not in ("A", "B"):
                log.info(f"    Ignorando (sem vencedor claro): {winner}")
                continue

            def load_fighter(url, name):
                if not url:
                    return None
                if url not in fighter_cache:
                    log.info(f"    Perfil: {name}")
                    time.sleep(DELAY)
                    fighter_cache[url] = get_fighter_profile(url)
                return fighter_cache.get(url)

            pa = load_fighter(details.get("fighter_a_url"), details.get("fighter_a_name"))
            pb = load_fighter(details.get("fighter_b_url"), details.get("fighter_b_name"))

            if not pa or not pb:
                continue

            date  = event["date"]
            sa    = calc_stats(pa, before=date)
            sb    = calc_stats(pb, before=date)

            row = {
                "event_name":  event["name"],
                "event_date":  date.date(),
                "fighter_a":   details.get("fighter_a_name"),
                "fighter_b":   details.get("fighter_b_name"),
                "winner":      winner,
                "label":       1 if winner == "A" else 0,
                "method":      details.get("method"),
                "round":       details.get("round"),
                "time":        details.get("time"),
            }
            for k, v in sa.items():
                row[f"a_{k}"] = v
            for k, v in sb.items():
                row[f"b_{k}"] = v

            rows.append(row)
            log.info(f"    ✓ {details['fighter_a_name']} vs {details['fighter_b_name']} → {winner}")

        # Salva parcialmente após cada evento
        if rows:
            pd.DataFrame(rows).to_csv(output, index=False)
            log.info(f"  💾 Salvo: {len(rows)} lutas em '{output}'")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    log.info(f"\n✅ Concluído! {len(df)} lutas → '{output}'")
    return df


# ─────────────────────────────────────────────────────────────
# 7. Entrada
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UFC Scraper v2")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limite de eventos (padrão: todos)")
    parser.add_argument("--output", type=str, default="ufc_dataset.csv",
                        help="Arquivo CSV de saída")
    parser.add_argument("--test", action="store_true",
                        help="Modo teste: apenas 3 eventos")
    args = parser.parse_args()

    n = 3 if args.test else args.max_events
    build_dataset(max_events=n, output=args.output)