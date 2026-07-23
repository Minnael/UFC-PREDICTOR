import json
import logging
import time
import requests
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser as date_parser

# Importamos as lógicas do scraper atual para não reescrever a roda
from ufc_scraper_global import (
    _playwright_loop, _request_queue, get_fights_from_event, process_fight, DELAY, MAX_WORKERS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WORKER] %(message)s")
log = logging.getLogger(__name__)

def start_worker():
    # 1. Obter o IP do Host
    host_ip = input("Digite o IP do Host (ex: 192.168.1.10) ou aperte Enter para 'localhost': ").strip()
    if not host_ip:
        host_ip = "127.0.0.1"
    
    base_url = f"http://{host_ip}:5000"
    
    # 2. Ligar o navegador Playwright
    log.info("Iniciando navegador silencioso do Playwright...")
    threading.Thread(target=_playwright_loop, daemon=True).start()
    
    # 3. Loop infinito buscando tarefas
    while True:
        try:
            log.info("Solicitando evento para o Host...")
            response = requests.get(f"{base_url}/next_event")
            if response.status_code != 200:
                log.error("Servidor Host retornou erro. Tentando de novo em 5s...")
                time.sleep(5)
                continue
                
            data = response.json()
            if data.get("done"):
                log.info("🎉 Nenhum evento restante! O Host confirmou que o dataset está completo. Desligando...")
                break
                
            event = data["event"]
            # Converter a string ISO de volta para datetime
            event["date"] = date_parser.parse(event["date"])
            
            log.info(f"Recebido: {event['name']} ({event['date'].date()})")
            time.sleep(DELAY)
            
            # Buscar lutas
            fights = get_fights_from_event(event)
            if not fights:
                log.info("O evento não tem lutas publicadas.")
                continue
                
            rows = []
            rows_lock = threading.Lock()
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_fight, f, event): f for f in fights}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        with rows_lock:
                            rows.append(result)
                            
            # Enviar de volta
            if rows:
                log.info(f"Enviando {len(rows)} lutas para o Host...")
                requests.post(f"{base_url}/submit_event", json={"rows": rows})
            else:
                log.info("Nenhuma luta válida para enviar.")
                
        except requests.exceptions.ConnectionError:
            log.error(f"Host {host_ip}:5000 não encontrado. Ele está ligado? Tentando de novo em 5s...")
            time.sleep(5)
        except Exception as e:
            log.error(f"Erro inesperado: {e}")
            time.sleep(5)
            
    # Sinalizar para o Playwright morrer
    _request_queue.put(None)

if __name__ == '__main__':
    start_worker()
