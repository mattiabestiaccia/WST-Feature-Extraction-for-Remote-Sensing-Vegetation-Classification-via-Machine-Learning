You are an expert scientific editor and AI code assistant. 
Your task is to deeply analyze my LaTeX scientific paper and compare it against the Elsevier author guidelines provided in a separate Markdown document, `/home/brusc/Projects/random_forest/new_root/elsevier_author_guide.md`  

## HIGH-LEVEL GOAL
Your goal is to:
1. Read and understand **all Elsevier instructions** (formatting, sections, abstract, references, figures, tables, language style, length, etc.).
2. Scan **every section, subsection and LaTeX file** in `/paper/`.
3. Identify *all* mismatches, issues, or missing elements preventing submission.
4. Produce:
   - A **comprehensive revision plan**  
     (bullet points + detailed explanations)
   - A **structured list of all necessary changes**  
     (formatting, style, structure, figures, references, equations, captions, metadata, etc.)

## REQUIRED OUTPUT STRUCTURE
Do NOT modify any LaTeX source directly.

Instead, write **all your findings and proposed modifications** inside a new file named:

`revision_plan.md`

The file must include:

### 1. High-level compliance assessment  
- Summary of adherence to Elsevier requirements  
- Critical issues blocking submission  
- Minor issues  
- Optional improvements  

### 2. Section-by-section deep analysis  
For every section of the paper, include:
- What matches the guidelines  
- What violates the guidelines  
- Missing required elements  
- Suggested rewrites  
- Structural corrections  
- Improvements to clarity, conciseness, and academic tone  

### 3. Formatting compliance  
Check and document in detail:
- Title page requirements  
- Author names & affiliations  
- Abstract formatting  
- Keyword formatting  
- Figure and table placement  
- Figure resolution and source images  
- Citation and reference style (Elsevier-specific)  
- Equations formatting  
- Page limits (if applicable)  
- Required LaTeX packages for Elsevier style  
- Any conflict with the journal’s .cls file  

### 4. List of required actions  
A numbered list, with each item containing:
- **Action title**
- **Why it is required** (cite the specific guideline from elsevier_instructions.md)
- **Priority**: HIGH / MEDIUM / LOW
- **Suggested implementation strategy**  

### 5. Optional improvements  
Recommendations that are not mandatory, but beneficial for acceptance.

## IMPORTANT RULES
- Think step-by-step and reason extensively before writing the revision_plan.md file.
- Be exhaustive: review every subfile, every subsection, and every figure/table.
- Do not modify LaTeX files until explicitly asked.
- Do not skip any part of the Elsevier guidelines.
- The revision_plan.md must be complete, organized, and actionable.

When ready, create the file `revision_plan.md` and write all findings there.
