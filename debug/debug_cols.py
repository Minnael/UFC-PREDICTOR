"""
Diagnóstico: mostra o conteúdo de cada <td> das linhas de luta do perfil.
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

url = "http://ufcstats.com/fighter-details/6b453bc35a823c3f"  # Carlos Prates
r    = requests.get(url, headers=HEADERS, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")

rows = soup.find_all("tr", attrs={"data-link": True})
print(f"Linhas de luta encontradas: {len(rows)}\n")

# Mostra conteúdo de cada <td> da primeira luta
row = rows[0]
tds = row.find_all("td")
print(f"Total de <td>: {len(tds)}\n")

for i, td in enumerate(tds):
    ps = [" ".join(p.get_text().split()) for p in td.find_all("p")]
    flag = td.find("i", class_="b-flag__text")
    flag_txt = " ".join(flag.get_text().split()) if flag else ""
    print(f"td[{i}]: flag='{flag_txt}' | p_texts={ps}")