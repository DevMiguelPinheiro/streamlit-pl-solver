# 🎯 Solver de Programação Linear - Lachtermacher 3.1.1

Dashboard interativo em Streamlit para resolver problemas de **Programação Linear (PL)** com duas variáveis (x₁ e x₂), baseado no **Problema 3.1.1 do livro "Pesquisa Operacional na Tomada de Decisões" de Gerson Lachtermacher**.

## 🚀 Funcionalidades

### ✅ Entrada de Dados Interativa
- **Tipo de objetivo**: Maximizar ou Minimizar
- **Coeficientes da função objetivo**: Valores para x₁ e x₂
- **Restrições**: Configuração flexível de até 6 restrições
- **Tipos de desigualdade**: ≤, ≥, =

### ✅ Resolução Automática
- **Algoritmo**: SciPy (`scipy.optimize.linprog`)
- **Resultados**: Valor ótimo, valores das variáveis, status da solução
- **Tratamento de erros**: Problemas inviáveis, ilimitados, etc.

### ✅ Visualização Gráfica
- **Região viável**: Visualização interativa da área de soluções
- **Linhas de restrições**: Cada restrição com cor diferente
- **Ponto ótimo**: Destacado com estrela dourada
- **Linha de isoprofit**: Função objetivo no ponto ótimo

### ✅ Exportação de Resultados
- **CSV**: Dados tabulares dos resultados
- **Excel (.xlsx)**: Planilha com resultados e gráfico
- **PDF**: Relatório formatado
- **PNG**: Imagem do gráfico da região viável

## 📊 Exemplo: Problema 3.1.1 do Lachtermacher

**Problema:** Maximizar Z = 3x₁ + 2x₂

**Sujeito a:**
- x₁ + x₂ ≤ 10
- 2x₁ + x₂ ≤ 15  
- x₁ ≤ 8
- x₁, x₂ ≥ 0

**Solução:** x₁ = 5, x₂ = 5, Z = 25

## 🛠️ Como Usar

1. **Defina o tipo de objetivo** (Maximizar ou Minimizar)
2. **Insira os coeficientes** da função objetivo
3. **Configure as restrições** (coeficientes, tipo de desigualdade e RHS)
4. **Clique em 'Resolver Problema'**
5. **Visualize os resultados** e o gráfico da região viável
6. **Exporte os resultados** nos formatos desejados

## 🚀 Deploy no Streamlit Cloud

Este projeto está pronto para deploy no **Streamlit Community Cloud**:

1. Faça upload dos arquivos para o GitHub
2. Conecte seu repositório ao Streamlit Cloud
3. O deploy será automático com as dependências do `requirements.txt`

## 📦 Dependências

- `streamlit`: Interface web
- `numpy`: Computação numérica
- `scipy`: Otimização linear
- `pandas`: Manipulação de dados
- `matplotlib`: Gráficos básicos
- `plotly`: Gráficos interativos
- `fpdf`: Geração de PDF
- `openpyxl`: Manipulação de Excel
- `kaleido`: Exportação de gráficos Plotly

## 📁 Estrutura do Projeto

```
.
├── app.py                # Código principal do Streamlit
├── requirements.txt      # Dependências para deploy
└── README.md            # Este arquivo
```

## 🎓 Baseado em

**Lachtermacher, G. (2017). Pesquisa Operacional na Tomada de Decisões.**  
*Problema 3.1.1 - Programação Linear com Duas Variáveis*

---

**Desenvolvido para fins educacionais e de pesquisa operacional.** 