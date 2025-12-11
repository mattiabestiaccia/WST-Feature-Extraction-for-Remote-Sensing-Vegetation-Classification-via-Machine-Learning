# PIANO DI REVISIONE 
## Multiscale Feature Extraction with Wavelet Scattering Transform for Remote Sensing Vegetation Classification via Machine Learning

**Data Revisione:** 2025-11-23
**Rivista Target:** Ecological Informatics (Elsevier)
**Autore Principale:** Mattia Bruscia
**Revisori:** Claude Code + Gemini AI (doppia validazione)

---

✅ **Validazione reciproca su 6 problemi bloccanti:**
1. Stile bibliografico errato (plain → author-year)
2. Funding statement incompleto (placeholder)
4. Document class non ottimale (article → elsarticle)


## 🔴 PROBLEMI BLOCCANTI - PRIORITÀ MASSIMA


### PROBLEMA 3: Funding Statement Incompleto

**Stato Attuale:** Line 716 contiene placeholder `[FUNDING AGENCY NAME] [grant number XXXX]`

**Azione Richiesta:**

**Opzione 1 - Se NON hai finanziamenti specifici (RACCOMANDATO):**

```latex
\section*{Funding}

This research did not receive any specific grant from funding agencies in
the public, commercial, or not-for-profit sectors.
```

**Opzione 2 - Se hai finanziamenti universitari:**

```latex
\section*{Funding}

This work was supported by the University of Maryland Center for
Environmental Science.
```

**Opzione 3 - Se hai grant specifici:**

```latex
\section*{Funding}

This work was supported by the National Science Foundation [grant number
NSF-XXXXXXX]; and the Maryland Sea Grant [grant number NA18OAR4170090].
```

**Formato Richiesto da Elsevier (Lines 194-206):**
> List funding sources in this standard way: "This work was supported by
> the [Agency Name] [grant numbers xxxx, yyyy]..."

**⚠️ IMPORTANTE:** Se usi Opzione 1, **commenta o rimuovi** completamente le righe 716-718:

```latex
% RIMUOVI QUESTO:
% This work was supported by [FUNDING AGENCY NAME] [grant number XXXX].
```

**Tempo stimato:** 5 minuti (se hai le info), 1 giorno (se devi verificare con co-autori)
**Priorità:** 🔴 ALTA - Sistema di submission potrebbe bloccarsi

---

## 🟡 PROBLEMI STRUTTURALI - PRIORITÀ MEDIA

### PROBLEMA 7: Document Class Non Ottimale

**Stato Attuale:** `\documentclass{article}` con formattazione manuale

**Implicazioni:**
- Formattazione non automatica
- Potenziali conflitti durante production
- Richiede più lavoro manuale


```latex
\documentclass[preprint,12pt,authoryear]{elsarticle}

% Pacchetti essenziali
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{subcaption}
\usepackage{hyperref}

\journal{Ecological Informatics}

\begin{document}

\begin{frontmatter}

\title{Multiscale Feature Extraction with Wavelet Scattering Transform
for Remote Sensing Vegetation Classification via Machine Learning}

% Autori con email
\author[umces]{Mattia Bruscia\corref{cor1}}
\ead{mbruscia@umces.edu}
\orcid{0000-0003-0910-6445}

\author[umces]{William Nardin}
\ead{wnardin@umces.edu}
\orcid{0000-0002-5490-879X}

\author[umces]{Xiaoxu Guo}
\ead{xguo@umces.edu}

\author[umces]{Limin Sun}
\ead{lsun@umces.edu}

\author[salisbury]{Giulia Franchi}
\ead{gfranchi@salisbury.edu}

\cortext[cor1]{Corresponding author}

% Affiliazioni
\affiliation[umces]{organization={University of Maryland Center for
    Environmental Science, Horn Point Laboratory},
    addressline={2020 Horns Point Road},
    city={Cambridge},
    postcode={MD 21613},
    country={USA}}

\affiliation[salisbury]{organization={Department of Computer Science,
    Salisbury University},
    addressline={1101 Camden Avenue},
    city={Salisbury},
    postcode={MD 21801},
    country={USA}}

% Abstract (max 250 parole)
\begin{abstract}
This study evaluates the Wavelet Scattering Transform (WST) for Random
Forest-based land cover classification under limited datasets and noisy
imagery...
[Usa la versione condensata da Problema 5]
\end{abstract}

% Keywords (max 7)
\begin{keyword}
Feature extraction \sep Wavelet Scattering Transform \sep Vegetation
Classification \sep Ecological Restoration \sep Wetland Monitoring \sep
UAV Remote Sensing \sep Machine Learning
\end{keyword}

\end{frontmatter}

% =========================================
% CORPO DEL DOCUMENTO (invariato)
% =========================================

\section{Introduction}
% ... tutto il contenuto attuale ...

% ... tutte le altre sezioni ...

% =========================================
% BACK MATTER
% =========================================

\section*{Supplementary Materials}
% ... contenuto attuale ...

\section*{Data Availability Statement}
% ... contenuto attuale ...

\section*{Declaration of Competing Interest}
% ... contenuto attuale ...

\section*{Funding}
This research did not receive any specific grant from funding agencies in
the public, commercial, or not-for-profit sectors.

\section*{Author Contributions}
\textbf{Mattia Bruscia:} Conceptualization, Methodology, Software, Formal
analysis, Investigation, Data curation, Writing -- original draft,
Visualization.

\textbf{William Nardin:} Conceptualization, Methodology, Resources,
Writing -- review \& editing, Supervision, Project administration.

\textbf{Xiaoxu Guo:} Software, Methodology, Writing -- review \& editing.

\textbf{Limin Sun:} Formal analysis, Writing -- review \& editing.

\textbf{Giulia Franchi:} Methodology, Software, Validation, Writing --
review \& editing.

\section*{Declaration of Generative AI and AI-assisted Technologies in
the Writing Process}

During the preparation of this work the author(s) used Claude (Anthropic)
in order to assist with data organization, software development and
coding, manuscript formatting, compliance checking with journal
guidelines, and editorial revisions. After using this tool/service, the
author(s) reviewed and edited the content as needed and take(s) full
responsibility for the content of the published article.

% Bibliografia
\bibliographystyle{elsarticle-harv}
\bibliography{bibliography}

\end{document}
```

**Pacchetti da RIMUOVERE (gestiti da elsarticle):**
- ❌ `authblk` (affiliazioni gestite da elsarticle)
- ❌ `geometry` (page layout gestito da elsarticle)
- ❌ `titlesec` (section formatting gestito da elsarticle)
- ❌ `setspace` (spacing gestito da elsarticle)
- ❌ `helvet` (font gestiti da elsarticle)

**Tempo stimato:**
- Approccio A: 15 minuti
- Approccio B: 3-4 ore (include testing completo)

---

### PROBLEMA 8: Highlights Mancanti

**Requisito Elsevier:** 3-5 highlights, max 85 caratteri ciascuno

**Stato Attuale:** File `highlights.txt` esiste ✅, ma serve formato corretto per submission

**Contenuto Attuale (già conforme):**

```
• Hybrid WST+statistics features improve F1 by 1.7% vs statistics alone (p=0.039)
• All methods retain >94% accuracy with only 5 images/class (extreme scarcity)
• Hybrid degrades 131% slower than statistics under increasing noise intensity
• WST alone shows no significant advantage; benefits only when combined
• 1,512 experiments systematically tested across 6 noise types and 3 dataset sizes
```

**Verifica Lunghezza:**
- Bullet 1: 78 caratteri ✅
- Bullet 2: 76 caratteri ✅
- Bullet 3: 73 caratteri ✅
- Bullet 4: 71 caratteri ✅
- Bullet 5: 82 caratteri ✅

**Formato per Submission:**

**Con `elsarticle` class:**
```latex
% All'interno di \begin{frontmatter}...\end{frontmatter}

\begin{highlights}
\item Hybrid WST+statistics features improve F1 by 1.7\% vs statistics
      alone (p=0.039)
\item All methods retain >94\% accuracy with only 5 images/class
      (extreme scarcity)
\item Hybrid degrades 131\% slower than statistics under increasing
      noise intensity
\item WST alone shows no significant advantage; benefits only when combined
\item 1,512 experiments systematically tested across 6 noise types and
      3 dataset sizes
\end{highlights}
```

**File Separato per Submission System:**

Crea file `highlights.docx` o `highlights.tex` con:

```
HIGHLIGHTS

• Hybrid WST+statistics features improve F1 by 1.7% vs statistics alone (p=0.039)

• All methods retain >94% accuracy with only 5 images/class (extreme scarcity)

• Hybrid degrades 131% slower than statistics under increasing noise intensity

• WST alone shows no significant advantage; benefits only when combined

• 1,512 experiments systematically tested across 6 noise types and 3 dataset sizes
```

**Tempo stimato:** 10 minuti


### Prepara File per Submission (30 minuti)

**Checklist File da Preparare:**

1. **Main Manuscript:**
   - [ ] `main.tex` (con tutte le correzioni)
   - [ ] `main.pdf` (compilato)

2. **Supplementary Materials:**
   - [ ] `supplementary_materials.pdf` (verifica compilazione)
   - [ ] Rinomina in `Bruscia_2025_SupplementaryMaterials.pdf`

3. **Highlights:**
   - [ ] Crea `Bruscia_2025_Highlights.docx`
   - [ ] Contiene i 5 bullet points (verificati <85 char)

4. **Figures (separate):**
   - [ ] Tutte le figure come file individuali
   - [ ] Nome logico: `Figure_1.png`, `Figure_2.png`, etc.
   - [ ] Verifica risoluzione

5. **Bibliography:**
   - [ ] `bibliography.bib` (verifica formato)

6. **Cover Letter:**
   - [ ] Crea `CoverLetter.pdf`
   - [ ] Include: titolo, rilevanza per rivista, originalità

---

### Fase 4 (OPZIONALE): Migrazione a elsarticle (6-8 ore)

**Solo se vuoi massima compliance**

1. Scarica template: `wget https://www.elsevier.com/__data/assets/file/.../elsarticle-template.zip`
2. Usa template completo fornito in "PROBLEMA 7 - Approccio B"
3. Copia contenuto sezioni da `main.tex` attuale
4. Test compilazione completa
5. Verifica formattazione

---


## 🎯 CHECKLIST FINALE PRE-SUBMISSION

### Metadati Completi
- [ ] **Email per TUTTI gli autori** presenti (non solo corresponding)
- [ ] ORCID ID dove disponibili
- [ ] Affiliazioni complete con indirizzo postale e paese
- [ ] Corresponding author chiaramente indicato

### Contenuto Conforme
- [ ] Abstract ≤250 parole (conta con `texcount` o Word)
- [ ] Esattamente 1-7 keywords (attualmente: 7 ✅)
- [ ] Highlights: 3-5 bullet points, ciascuno <85 caratteri

### Bibliografia
- [ ] Citazioni in formato Author-Year: (Mallat, 2012)
- [ ] Bibliografia in ordine alfabetico
- [ ] Nessuna citazione numerica [1], [2], etc.
- [ ] DOI inclusi dove disponibili

### Dichiarazioni Complete
- [ ] **Funding statement** senza placeholder
- [ ] **Author contributions** completo per TUTTI (incluso Giulia Franchi)
- [ ] Competing interests dichiarati
- [ ] Data availability statement presente
- [ ] AI usage disclosure presente (già OK ✅)

### Figure e Tabelle
- [ ] Tutte le figure citate nel testo
- [ ] Tutte le tabelle citate nel testo
- [ ] Figure ad alta risoluzione (300-500 dpi)
- [ ] Didascalie descrittive
- [ ] File separati per ogni figura

### File per Submission
- [ ] `main.tex` (source file)
- [ ] `main.pdf` (compiled)
- [ ] `bibliography.bib`
- [ ] `supplementary_materials.pdf`
- [ ] `highlights.docx` o `.tex`
- [ ] Figure individuali (PNG/TIFF ad alta risoluzione)
- [ ] `CoverLetter.pdf`

### Test Compilazione
- [ ] Compila senza errori con `pdflatex + bibtex`
- [ ] PDF generato correttamente
- [ ] Tutte le figure visibili
- [ ] Riferimenti bibliografici risolti
- [ ] Nessun warning critico nel log

---
