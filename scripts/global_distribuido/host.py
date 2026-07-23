import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import pandas as pd

from ufc_scraper_global import get_all_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s [HOST] %(message)s")
log = logging.getLogger(__name__)

DATASET_FILE = "ufc_dataset.csv"
PORT = 5000

event_queue = []
queue_lock = threading.Lock()
save_lock = threading.Lock()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class HostHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/next_event':
            with queue_lock:
                if event_queue:
                    event = event_queue.pop(0)
                    response = {"done": False, "event": event}
                    log.info(f"Enviando evento para Worker: {event['name']}")
                else:
                    response = {"done": True}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, default=str).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/submit_event':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                rows = data.get("rows", [])
                
                if rows:
                    with save_lock:
                        df = pd.DataFrame(rows)
                        write_header = not os.path.exists(DATASET_FILE)
                        df.to_csv(DATASET_FILE, mode='a', header=write_header, index=False)
                        log.info(f"Salvo {len(rows)} lutas de um Worker!")
                        
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except Exception as e:
                log.error(f"Erro ao salvar dados do Worker: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_host():
    global event_queue
    
    processed_events = set()
    if os.path.exists(DATASET_FILE):
        try:
            df = pd.read_csv(DATASET_FILE)
            if "event_name" in df.columns:
                processed_events = set(df["event_name"].unique())
                log.info(f"Arquivo CSV encontrado com {len(processed_events)} eventos.")
        except Exception as e:
            log.error(f"Erro ao ler CSV existente: {e}")
            
    log.info("Buscando lista completa de eventos do UFC...")
    all_events = get_all_events()
    
    for event in all_events:
        if event["name"] not in processed_events:
            event_copy = event.copy()
            event_copy["date"] = event["date"].isoformat()
            event_queue.append(event_copy)
            
    log.info(f"Host pronto! {len(event_queue)} eventos restantes na fila.")
    
    server = ThreadingHTTPServer(('0.0.0.0', PORT), HostHandler)
    log.info(f"Servidor aguardando Workers na porta {PORT}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Encerrando Host...")
        server.server_close()

if __name__ == '__main__':
    start_host()
