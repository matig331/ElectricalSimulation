# HOWTO — figura "tre esiti" di una cultura simulata (culture_states_hpc.py)

## A cosa serve
Disegna, per **una cultura simulata** e **uno spessore dello slab**, tre pannelli:

1. in **verde** i neuroni che hanno sparato in un qualunque momento della simulazione;
2. in **rosso** (gradiente di ΔVm) i neuroni **depolarizzati** a t₀ (fine dell'impulso);
3. in **blu** (gradiente di ΔVm) i neuroni **iperpolarizzati** a t₀.

Gli altri neuroni sono puntini grigio chiaro; le **morfologie** sono in grigio sottile.
Regole: *lo spike vince* (un neurone che spara non conta come depolarizzato/iperpolarizzato);
altrimenti ΔVm ≥ +1 mV → depolarizzato, ΔVm ≤ −1 mV → iperpolarizzato, il resto è neutro.
Le etichette del CSV (solo segno, senza soglia) **non** vengono usate.

**Non serve rifare nessun merge, né `config.py`, né NEURON, né altri script del progetto.** Si parte dal `merged_*.csv` che esiste già, oppure da un CSV della campagna.
Solo `numpy`, `pandas`, `matplotlib` e `morphio`. Nel repository lo script è nella cartella della pipeline e il job è `jobs/culture_states.pbs` (Passo 6).

**Tempi e memoria (misurati su 1.2 milioni di righe, 1700 neuroni):** lettura del file ~3 s (≈ 10 s per i 3.7 milioni della campagna), figura completa con le 6 morfologie **40–90 secondi** su un core, circa **350 MB** di memoria. Il PNG pesa ~6 MB, il PDF < 1 MB. Va bene il nodo di login o un job da 30 minuti.

## Cosa serve avere sotto mano
| Cosa | Dove |
|---|---|
| `culture_states_hpc.py` | nella cartella della pipeline (repository) |
| **un** `merged_*.csv` della campagna | `<cartella statistiche>/activation/merged_activation.csv` di `culture_statistics`: per il job di analisi `analysis/<run>/stats/activation/merged_activation.csv` (**basta uno solo** dei tre merged: contengono gli stessi neuroni e gli stessi esiti) |
| oppure un `culture_P*.csv` | di `culture_merge.py` (`results_<modello>/<job>/`, `merged_<modello>/`, `analysis/<run>/merged/`) o di `culture_export.py` |
| oppure un CSV **grezzo** | un pezzo di un job, `results_<modello>/parts_<job>/part_NNN.csv` (va bene anche per un job ancora in corso) |
| le morfologie | `eyal_archive/` nella cartella della pipeline: contiene `specimen_*/morphology.asc` (6 specimen: 60308, 130303, 60303, 60311, 130305, 130306) |

Trovare il file (dalla cartella della pipeline): `find . -name "merged_activation.csv" 2>/dev/null` (oppure `find . -name "culture_Pactivation*.csv" | head`).

**Controllo veloce che il merged abbia il `seed`** (serve per le morfologie; se c'è, non va dato a mano):
```bash
head -1 /percorso/merged_activation.csv | tr ',' '\n' | grep -n -i "^seed$\|culture_global"
```
Deve stampare `culture_global` e, di solito, `seed`. Se `seed` non c'è: vedere "Problemi comuni".

## Passo 1 — Ambiente Python
Servono Python ≥ 3.9 e `numpy pandas matplotlib morphio`.
- Sul cluster, nella cartella della pipeline, l'ambiente dei job li ha già: `conda activate neuron_env` (lo stesso di `jobs/env_setup.sh`).
- Altrove, se esiste un ambiente della pipeline (NEURON/LFPy), di solito li ha già:
  ```bash
  python -c "import numpy, pandas, matplotlib, morphio; print('ok')"
  ```
- Altrimenti:
  ```bash
  python3 -m venv ~/venv_figs && source ~/venv_figs/bin/activate
  pip install numpy pandas matplotlib morphio
  ```
  (se il cluster usa i moduli: prima `module load python`, il nome dipende dal cluster).

## Passo 2 — Test (meno di un minuto, nessun dato necessario)
```bash
python culture_states_hpc.py --selftest
```
Deve finire con `14/14 passed`. Se `morphio` manca, i test delle morfologie vengono saltati e lo dicono: allora installarlo (serve per le morfologie).

Nel repository c'è anche il controllo contro il codice della campagna (meno di un minuto, niente NEURON):
```bash
python smoke_test_culture_states.py
```
Deve finire con `All smoke tests passed.`: verifica che le copie interne dello script (estrazioni casuali, quadrato dei somata, regole degli esiti, elettrodi, slicing) coincidano con `culture_export`, `culture_merge`, `culture_statistics`, `field` e `slicer`.

## Passo 3 — Scegliere cultura e spessore
```bash
python culture_states_hpc.py --input /percorso/merged_activation.csv --list
```
Stampa un riepilogo (`720 cultures, layers [40, 80, 120], neurons per (culture, layer): [1700]`) e le prime righe come `full_active_S1000_C0  layer 80: 1700 neurons`. **Il nome della cultura** è `<modello>_S<seed>_C<n>` nei merged, `S<seed>_C<n>` nei `culture_P*.csv` e nei pezzi grezzi (`n` = indice della cultura nel suo job: nei file di `culture_merge` è la colonna `local_culture`, non `culture`, che lì è rinumerata): copiarlo da qui in `--culture`. Nella campagna ftd il job `ftdNN` ha seed 50000 + 1000·(NN − 1) e culture 0–47 (es. `full_tuned_S50000_C0` = cultura 0 di ftd01). Scegliere una cultura e uno spessore (40, 80 o 120).

## Passo 4 — Fare la figura (tutte e 6 le morfologie)
```bash
python culture_states_hpc.py \
    --input /percorso/merged_activation.csv \
    --layer 80 --culture NOME_DELLA_CULTURA \
    --morph-dir eyal_archive --require-all-morph \
    --out figures/culture_states_C0_L80
```
- `--morph-dir` = cartella che contiene `specimen_*/morphology.asc` (nel repository: `eyal_archive`). Controllo:
  `find eyal_archive -path "*specimen_*" -name morphology.asc | wc -l` deve stampare **6**.
- `--require-all-morph` fa **fermare** il programma se manca anche un solo specimen (invece di una figura incompleta).
- Seed, neuroni per cultura, numero di specimen e quadrato dei somata (semilato S e centro) vengono **ricavati dal CSV** e stampati, per esempio:
  `parameters: neurons per culture N=2000, specimens=6, half side S=300 um, centre (0, -30) um  (inferred)`.
  Si possono forzare con `--seed --n-neurons --n-morph --span --center`. Va bene anche una cultura incompleta (il pezzo di un job ancora in corso): il numero di neuroni per cultura viene cercato e verificato sulle rotazioni.
- Senza morfologie: togliere `--morph-dir --require-all-morph`.

**Output:** `figures/culture_states_C0_L80.png` e `.pdf` (300 dpi) e, a schermo, la riga con i conteggi:
`Culture NOME -- layer 80 um: 2000 neurons -> activated …, depolarized …, hyperpolarized …, neutral …`

Con 1700–2000 neuroni le morfologie e i somata si adattano da soli (morfologie più chiare, somata più piccoli). Per vedere meglio la zona attorno agli elettrodi: **`--half 300`**.

Altre opzioni: `--neu 1` (soglia mV), `--dvm-max 10` (scala colori), `--half 500` (semilato mappa), `--soma-size`, `--morph-alpha`, `--pool` (tutte le culture insieme, senza morfologie).

## Passo 5 — Controllare che sia credibile
- La riga `parameters:` deve dare i valori della campagna:

  | campagna | N | S | centro |
  |---|---|---|---|
  | ftd (full_tuned, dal 2026-09-29) | 2000 | 300 µm | (0, −30) µm = centro del dipolo |
  | precedenti al 2026-09-28 (720 culture; soma_only, full_active, ftc) | 1700 | 500 µm | (0, 0) |

  Se N, S o il centro sono diversi, è la run sbagliata.
- `morphologies drawn for [6 nomi]` e nessun `NOT available`.
- Nel titolo di ogni pannello: "k of **N** neurons" con l'N della tabella sopra.
- Se compare `NOTE: morphologies not drawn (... do not match ...)`, il programma ha verificato che le posizioni o le rotazioni ricostruite **non** coincidono con quelle del CSV (`x_um`, `y_um`, `theta_orient_deg`) e si rifiuta di disegnare morfologie sbagliate. Mandare il messaggio completo.

## Passo 6 (facoltativo) — Come job PBS
Dalla cartella della pipeline (il job usa l'ambiente di `jobs/env_setup.sh`, un core, 30 minuti):
```bash
qsub -v INPUT=analysis/<run>/stats/activation/merged_activation.csv,CULTURE=full_tuned_S50000_C0,LAYER=80 jobs/culture_states.pbs
qsub -v JOB=ftd01,CULTURE=0 jobs/culture_states.pbs        # cultura 0 di ftd01, dal merge del job
```
Opzioni (`-v`, separate da virgole): `INPUT`, `JOB` (+ `MODEL`), `CULTURE`, `LAYER` (80), `MORPH_DIR` (`eyal_archive`; `none` = solo somata), `OUT` (`figures/culture_states_<CULTURE>_L<LAYER>`), `NEU` (1), `DVM_MAX` (10), `HALF`, e solo se servono `SEED` (file senza colonna `seed`), `N_NEURONS` e `NO_ORIENT_CHECK=1` (vedere "Problemi comuni").
Log: `logs/estim_culture_states.log`. Righe che **devono** comparire: `14/14 passed`, `parameters: …` con i valori del Passo 5, `morphologies drawn for […]` con 6 nomi, i conteggi, `done: <OUT>.png`. Se le morfologie non sono state disegnate, alla fine compare `WARNING: morphologies NOT drawn`; con `NO_ORIENT_CHECK=1` compare sempre `WARNING: rotations NOT checked on theta_orient_deg`.

## Cosa rimandare indietro
I file `figures/*.png` e `*.pdf`, e le righe stampate a schermo (`parameters: …`, `morphologies drawn …`, i conteggi).

## Problemi comuni
| Messaggio | Causa e rimedio |
|---|---|
| `missing columns [...]` | non è una tabella della campagna (`merged_*.csv`, `culture_P*.csv` o `part_*.csv`) |
| `no rows for layer 80 / culture ...` | cultura o spessore sbagliati: usare `--list` |
| `no `seed` column in this file` | file senza colonna `seed` (export vecchio o merged senza `seed`): dare `--seed SEED` (il seed della campagna), oppure cercarlo con `--find-seed --n-neurons 1700 --span 500 --n-morph 6` (il centro del quadrato non serve per la ricerca) |
| `could not infer the soma square` | nessun numero di neuroni per cultura riproduce posizioni e rotazioni: il seed non è quello della campagna o il file non è una tabella della campagna (con un seed sbagliato la ricerca dura ~10 s prima di arrendersi) |
| `... the positions are reproduced but never the rotations ...` | la colonna `theta_orient_deg` del file segue un'altra definizione (export più vecchio di `culture_export.rel_orientation_deg`): `--no-orient-check` (job: `NO_ORIENT_CHECK=1`) controlla solo le posizioni. Giusto per culture complete, **sbagliato** per culture tagliate (job in corso o interrotti): lo dice la riga `NOTE: rotations checked through the positions only` |
| `NOTE: ... the last row is incomplete` | il file è il pezzo di un job ancora in corso: l'ultima riga, scritta a metà, viene scartata (normale) |
| `STOP: no .asc found for specimen(s) [...]` | mancano morfologie sotto `--morph-dir` |
| `NOTE: morphologies not drawn (... morphio ...)` | `pip install morphio` |
| `WARNING: morphologies NOT drawn` (nel log del job) | leggere la riga `NOTE: morphologies not drawn (...)` subito sopra |
| `MemoryError` | non dovrebbe succedere (lettura a blocchi): provare `--chunksize 100000` |
