"""
UFC Stats Scraper v3 — seletores baseados no HTML real inspecionado.

Uso:
    python ufc_scraper_global.py --test            # 3 eventos
    python ufc_scraper_global.py --max-events 50   # N eventos
    python ufc_scraper_global.py                   # todos (horas)
"""

from bs4 import BeautifulSoup
import pandas as pd
import time, re, logging, threading, queue
import concurrent.futures as cf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE        = "http://ufcstats.com"
DELAY       = 1
MAX_WORKERS = 6


# ─────────────────────────────────────────────────────────────
# Playwright — thread dedicada com fila de requests
#
# O sync_playwright usa greenlets e NÃO pode ser chamado de
# threads diferentes da que o criou. Solução: uma única thread
# "playwright-loop" consome a fila; as demais threads submetem
# URLs e aguardam o resultado via concurrent.futures.Future.
# ─────────────────────────────────────────────────────────────

_request_queue: queue.Queue = queue.Queue()

def _playwright_loop():
    pw      = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    )
    page: Page = context.new_page()

    while True:
        item = _request_queue.get()
        if item is None:          # sinal de encerramento
            _request_queue.task_done()
            break
        url, fut = item
        try:
            time.sleep(DELAY)
            page.goto(url, wait_until="networkidle", timeout=30_000)
            soup = BeautifulSoup(page.content(), "html.parser")
            if not fut.done():
                fut.set_result(soup)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
        finally:
            _request_queue.task_done()

    page.close()
    browser.close()
    pw.stop()

_pw_thread = threading.Thread(target=_playwright_loop, daemon=True, name="playwright-loop")
_pw_thread.start()

import atexit
def _shutdown():
    _request_queue.put(None)   # sinaliza encerramento
    _pw_thread.join(timeout=10)
atexit.register(_shutdown)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get(url: str) -> Optional[BeautifulSoup]:
    """Submete URL para a thread do Playwright e aguarda o resultado."""
    for attempt in range(3):
        fut: cf.Future = cf.Future()
        _request_queue.put((url, fut))
        try:
            return fut.result(timeout=60)
        except Exception as e:
            log.warning(f"[{attempt+1}/3] {url} — {e}")
            time.sleep(3)
    return None

def _throttled_get(url: str) -> Optional[BeautifulSoup]:
    """Alias de get() — delay já está dentro da thread do Playwright."""
    return get(url)

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

def get_fight_details(soup: BeautifulSoup) -> Optional[dict]:
    """Recebe soup já carregado — sem request extra."""
    d = {}

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

    if results_raw[0] == "W":   d["winner"] = "A"
    elif results_raw[1] == "W": d["winner"] = "B"
    else:                       d["winner"] = "NC"

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

    name_tag = soup.find("span", class_="b-content__title-highlight")
    p["name"] = clean(name_tag.text) if name_tag else ""

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

    p["fight_history"] = _parse_fight_history(soup, url)
    return p


def _parse_fight_history(soup: BeautifulSoup, fighter_url: str) -> list[dict]:
    """
    Mapeamento REAL confirmado via debug (Carlos Prates):
      td[0]: resultado   flag win/loss/draw/nc
      td[1]: nomes       [este lutador, adversário]
      td[2]: KD          [este, adversário]
      td[3]: Sig.Str.hit [este, adversário]
      td[4]: TD hit      [este, adversário]
      td[5]: Sub.Att     [este, adversário]
      td[6]: Evento + Data
      td[7]: Método
      td[8]: Round
      td[9]: Tempo
    """
    history = []

    for row in soup.find_all("tr", attrs={"data-link": True}):
        tds = row.find_all("td")
        if len(tds) < 10:
            continue

        flag = tds[0].find("i", class_="b-flag__text")
        result_txt = clean(flag.get_text()) if flag else ""
        result_map = {"win": "W", "loss": "L", "draw": "D", "nc": "NC"}
        result = result_map.get(result_txt.lower(), result_txt.upper())

        kd_ps     = p_texts(tds[2]); kd        = pf(kd_ps[0])     if kd_ps     else None
        kd_abs    = pf(kd_ps[1]) if len(kd_ps) > 1 else None
        
        sig_ps    = p_texts(tds[3]); sig_hit   = pf(sig_ps[0])    if sig_ps    else None
        sig_abs   = pf(sig_ps[1]) if len(sig_ps) > 1 else None
        
        td_ps     = p_texts(tds[4]); td_hit    = pf(td_ps[0])     if td_ps     else None
        td_abs    = pf(td_ps[1]) if len(td_ps) > 1 else None
        
        sub_ps    = p_texts(tds[5]); sub_att_n = pf(sub_ps[0])    if sub_ps    else None
        sub_abs   = pf(sub_ps[1]) if len(sub_ps) > 1 else None

        date = None
        date_ps = p_texts(tds[6])
        if len(date_ps) >= 2:
            date = parse_date(date_ps[1])
        if not date:
            continue

        method_ps = p_texts(tds[7]); method    = method_ps[0]     if method_ps else ""
        round_ps  = p_texts(tds[8]); rnd       = None
        if round_ps:
            try: rnd = int(round_ps[0])
            except: pass
        time_ps   = p_texts(tds[9]); fight_time = time_ps[0]      if time_ps   else ""

        history.append({
            "date": date, "result": result, "method": method,
            "round": rnd, "time": fight_time,
            "kd": kd, "sig_strikes_hit": sig_hit,
            "td_hit": td_hit, "sub_att": sub_att_n,
            "kd_absorbed": kd_abs, "sig_strikes_absorbed": sig_abs,
            "td_absorbed": td_abs, "sub_absorbed": sub_abs,
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

    s["wins"]         = len(wins)
    s["losses"]       = len(losses)
    s["draws"]        = len([f for f in hist if f["result"] == "D"])
    s["wins_ko"]      = sum(1 for f in wins if re.search(r"KO|TKO", f.get("method",""), re.I))
    s["wins_sub"]     = sum(1 for f in wins if re.search(r"Sub",    f.get("method",""), re.I))
    s["wins_dec"]     = sum(1 for f in wins if re.search(r"Dec",    f.get("method",""), re.I))
    s["losses_ko"]    = sum(1 for f in losses if re.search(r"KO|TKO", f.get("method",""), re.I))
    s["losses_sub"]   = sum(1 for f in losses if re.search(r"Sub",    f.get("method",""), re.I))
    s["losses_dec"]   = sum(1 for f in losses if re.search(r"Dec",    f.get("method",""), re.I))
    s["total_fights"] = len(hist)

    NULLS = [
        "sig_strikes_hit_total", "td_hit_total", "sub_att_total", "kd_total",
        "sig_strikes_per_min", "td_per_15min", "sub_per_15min", "avg_fight_time_sec",
        "kd_absorbed_total", "sig_strikes_absorbed_total", "td_absorbed_total", "sub_absorbed_total",
        "sig_strikes_absorbed_per_min", "td_absorbed_per_15min", "sub_absorbed_per_15min"
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
    
    sig_a = sum(f.get("sig_strikes_absorbed") or 0 for f in hist)
    td_a  = sum(f.get("td_absorbed")          or 0 for f in hist)
    sub_a = sum(f.get("sub_absorbed")         or 0 for f in hist)
    kd_a  = sum(f.get("kd_absorbed")          or 0 for f in hist)

    s["sig_strikes_hit_total"] = sig_h
    s["td_hit_total"]          = td_h
    s["sub_att_total"]         = sub
    s["kd_total"]              = kd
    s["sig_strikes_per_min"]   = round(sig_h/total_min, 3)
    s["td_per_15min"]          = round(td_h/total_min*15, 3)
    s["sub_per_15min"]         = round(sub/total_min*15, 3)
    s["avg_fight_time_sec"]    = round(total_sec/len(hist), 1)
    
    s["kd_absorbed_total"]            = kd_a
    s["sig_strikes_absorbed_total"]   = sig_a
    s["td_absorbed_total"]            = td_a
    s["sub_absorbed_total"]           = sub_a
    s["sig_strikes_absorbed_per_min"] = round(sig_a/total_min, 3)
    s["td_absorbed_per_15min"]        = round(td_a/total_min*15, 3)
    s["sub_absorbed_per_15min"]       = round(sub_a/total_min*15, 3)

    return s


# ─────────────────────────────────────────────────────────────
# 6. Pipeline principal
# ─────────────────────────────────────────────────────────────

def build_dataset(max_events: int = None, output: str = "ufc_dataset.csv"):
    fighter_cache: dict[str, dict] = {}
    cache_lock = threading.Lock()
    rows: list[dict] = []
    rows_lock  = threading.Lock()
    save_lock  = threading.Lock()

    events = get_all_events()
    if not events:
        log.error("Nenhum evento carregado.")
        return

    if max_events:
        events = events[:max_events]

    def load_profile(url: str, name: str) -> Optional[dict]:
        if not url: return None
        with cache_lock:
            if url in fighter_cache:
                return fighter_cache[url]
        log.info(f"    Perfil: {name}")
        profile = get_fighter_profile(url)
        with cache_lock:
            fighter_cache.setdefault(url, profile)
        return fighter_cache.get(url)

    def process_fight(fight: dict, event: dict) -> Optional[dict]:
        soup_fight = _throttled_get(fight["fight_url"])
        if not soup_fight: return None

        details = get_fight_details(soup_fight)
        if not details: return None

        winner = details.get("winner", "")
        if winner not in ("A", "B"):
            log.info(f"    Pulando (NC/Draw): {fight['fight_url']}")
            return None

        url_a  = details.get("fighter_a_url", "")
        url_b  = details.get("fighter_b_url", "")
        name_a = details.get("fighter_a_name", "")
        name_b = details.get("fighter_b_name", "")

        with ThreadPoolExecutor(max_workers=2) as inner:
            fut_a = inner.submit(load_profile, url_a, name_a)
            fut_b = inner.submit(load_profile, url_b, name_b)
            pa = fut_a.result()
            pb = fut_b.result()

        if not pa or not pb: return None

        sa = calc_stats(pa, before=event["date"])
        sb = calc_stats(pb, before=event["date"])

        row = {
            "event_name": event["name"],
            "event_date": event["date"].date(),
            "fighter_a":  name_a,
            "fighter_b":  name_b,
            "winner":     winner,
            "label":      1 if winner == "A" else 0,
            "method":     details.get("method", ""),
            "round":      details.get("round"),
            "time":       details.get("time", ""),
        }
        for k, v in sa.items(): row[f"a_{k}"] = v
        for k, v in sb.items(): row[f"b_{k}"] = v

        log.info(f"    ✓ {name_a} vs {name_b} → {winner}")
        return row

    for ei, event in enumerate(events):
        log.info(f"[{ei+1}/{len(events)}] {event['name']} — {event['date'].date()}")
        time.sleep(DELAY)

        fights = get_fights_from_event(event)
        if not fights: continue

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process_fight, f, event): f for f in fights}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    with rows_lock:
                        rows.append(result)

        with save_lock:
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