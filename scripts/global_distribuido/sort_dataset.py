import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SORT] %(message)s")
log = logging.getLogger(__name__)

def sort_csv():
    csv_path = os.path.join(os.path.dirname(__file__), "ufc_dataset.csv")
    
    if not os.path.exists(csv_path):
        log.error(f"Arquivo não encontrado: {csv_path}")
        return
        
    log.info("Carregando o dataset...")
    df = pd.read_csv(csv_path)
    log.info(f"Total de lutas carregadas: {len(df)}")
    
    # Converter a coluna event_date para DateTime para ter garantia que vai ordenar igual calendário
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Ordena do evento mais novo para o mais antigo (Decrescente)
    df = df.sort_values(by='event_date', ascending=False)
    
    log.info("Salvando dataset ordenado...")
    df.to_csv(csv_path, index=False)
    
    log.info("Sucesso! Seu dataset foi perfeitamente ordenado de forma cronológica.")

if __name__ == '__main__':
    sort_csv()
