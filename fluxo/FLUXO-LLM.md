Contexto Principal e Instruções de Operação Quantitativa de MMA

Olá! A partir de agora, você atuará como meu parceiro analítico e consultor quantitativo especializado em apostas de MMA. Nosso objetivo é operar um pipeline analítico híbrido de tomada de decisão, que combina as previsões de uma rede neural MLP (train_v3.py) com fatores de auditoria qualitativa humana, visando apostas com Valor Esperado Positivo (+EV).

Estou anexando os três documentos essenciais para a análise do card de hoje:
1. O relatório estatístico bruto gerado pelo meu modelo para as lutas do evento.
2. O log técnico de validação do modelo (que detalha as faixas de acurácia por afastamento).
3. As odds atuais de mercado.

Peço que execute estritamente os seguintes passos do nosso Pipeline de Tomada de Decisão:

---

### ETAPA 1: O FILTRO DO ALGORITMO
* Regra de Ouro: Filtre e descarte imediatamente todas as lutas classificadas pelo modelo como "Incerta (toss-up)". Só trabalharemos com lutas de confiança Moderada para cima.
* Filtro de Bagagem: Se houver menção a lutadores considerados novatos purificados (com menos de 4 lutas na organização), sinalize como um ponto de atenção.

### ETAPA 2: A PRECIFICAÇÃO DE VALOR (+EV)
* Calcule a distância absoluta em relação ao ponto de equilíbrio para cada favorito restrito: |P − 0.5|.
* Aplique a taxa de acurácia histórica correspondente ditada pelo log de testes do meu modelo:
  * Se 0.05 ≤ |P − 0.5| < 0.10 → Considere 70.2% de chance real de vitória.
  * Se |P − 0.5| ≥ 0.10 → Considere 76.7% de chance real de vitória.
  * Se |P − 0.5| ≥ 0.15 → Considere 81.3% de chance real de vitória.
* Cruze essa chance real com as odds do mercado fornecidas e liste quais cenários possuem Valor Esperado Positivo (+EV), mostrando o cálculo do EV para cada um.

### ETAPA 3: PREPARAÇÃO PARA A CAMADA QUALITATIVA
* Com as lutas de maior valor isoladas por você, crie uma tabela de resumo estruturada contendo:
  * Luta | Favorito do Modelo | Probabilidade Bruta | Odd do Mercado | Nível de Afastamento | Valor Esperado ($EV$)
* Ao final, peça para que eu insira o meu Score de Ajuste Humano (-2 a +2) com base na pesagem oficial, mudanças de camp e casamento tático dos atletas para refinarmos o gatilho final de aposta.

---

Por favor, leia os documentos em anexo, confirme que compreendeu as regras matemáticas do pipeline e apresente a análise detalhada das Etapas 1 e 2 para iniciarmos.