# Video della stimolazione — cosa fanno, come, come si lanciano su HPC

Allineati al codice HPC attuale (`culture_export.py` riscritto, con sham e `CellPool`) e parallelizzati. Sostituisce `VIDEO_PIPELINE_stato_e_piano.md`. `make_video_exporter.py` e `culture_export_video.py` sono **obsoleti**: non usarli.

---

## 1. I tre video

| Video | Cosa si vede | Messaggio |
|---|---|---|
| **1 — probabilità** (tutte le culture) | mappa del piano; colore = miscela pesata di P(attivato) verde, P(depolarizzato) rosso, P(iperpolarizzato) blu, neutro bianco, grigio = nessun dato; contorni P(attivazione) = 0.1 e 0.5; a lato la corrente e la **% attesa dell'area** in ciascuno stato nel tempo | come il campo bifasico sposta la probabilità: un lato in fase 1, l'altro in fase 2, poi attivazione e rilassamento |
| **2 e 3 — una cultura ciascuno** (un layer) | campo extracellulare Ve come gradiente (viola negativo, arancio positivo), elettrodi con la polarità del momento, morfologie colorate per specimen, ogni soma colorato con il suo ΔV(t) (rosso depolarizzato, blu iperpolarizzato), stella verde = attivato; a lato la frazione di questa cultura contro quella attesa dall'insieme, e P(stato \| θ) con intervalli di Wilson | neuroni deterministici di una cultura indipendente seguono la regola stimata sull'insieme |

---

## 2. Come funziona

**Tre script, in catena:**

| Script | Cosa fa | NEURON |
|---|---|---|
| `video_frames.py` | simula i neuroni e scrive ΔV(t) nel tempo | sì |
| `make_prob_videos.py` | video 1 | no |
| `make_culture_videos.py` | video 2 e 3 | no |

`jobs/video_frames.pbs` li lancia tutti e tre in un solo job.

**`video_frames.py` usa il codice della campagna:**
- **stesse culture**: `culture_draws(seed, c, …)` → stesse morfologie, posizioni e rotazioni;
- **stesse cellule**: un `CellPool` per processo (una cellula viva alla volta), stesso riposo e stato di leak, stesso stimolo (`config.i0_uA`, `spikes_at`);
- **stesso riferimento**: la simulazione sham (I = 0) della stessa cellula;
- **stessa etichetta**: `phase2_end_outcome` su ΔV_end.

Cambiano solo due cose, entrambe necessarie per un video, e nessuna delle due modifica la dinamica simulata, solo ciò che viene restituito:
- la traccia registrata parte **1 ms prima** della fine dell'impulso (la campagna la tiene da t₀ in poi): così i fotogrammi durante l'impulso sono veri;
- la coda dopo l'impulso dura `--tmax-ms` (default 200 ms), con la stessa finestra lunga della campagna (CVODE).

**Controllo automatico:** per ogni neurone, il ΔV_end del video viene confrontato con quello della campagna (V a fine fase 2 meno lo sham della campagna). A fine corsa lo script stampa la differenza massima, e se non è zero esce con errore e il job **non** disegna i video.

**Grandezze nei video:**
- ΔV(t) = V_stim(t) − V_sham(t);
- attivato = il soma ha superato 0 mV entro t;
- depolarizzato / iperpolarizzato = non attivato e |ΔV(t)| ≥ **1 mV** (la soglia di `culture_statistics.py`).

Probabilità nel video 1: media dei neuroni vicini con un kernel gaussiano di larghezza σ; la % di area è l'integrale di P sulla regione simulata.

**Parallelismo:** i neuroni di tutte le culture vengono divisi in pezzi (`--chunk`, default 50) e distribuiti su tutti i core del nodo. Ogni processo ha la sua cellula. I pezzi vengono poi riuniti, un file per cultura. Il risultato è identico riga per riga a una corsa seriale (verificato nello smoke test).

---

## 3. Come si lancia su HPC

Dal cluster, nella cartella del repository aggiornato:

**Video della campagna** (le culture di un job della campagna, per esempio ftd01):

```bash
qsub -v JOB=ftd01,TMAX=1000 jobs/video_frames.pbs   # culture 0-4 di ftd01, layer 80, 1000 ms
```

Con `JOB` il seed viene letto da `results_<modello>/parts_<JOB>/`, numero di neuroni e quadrato dei somata sono quelli di `config.py` (gli stessi della campagna). Due controlli: prima di simulare le posizioni delle culture scelte vengono rigenerate e confrontate con le righe della campagna; dopo, ogni neurone x layer del video viene confrontato con la sua riga (specimen, posizione, spike, ΔV_end entro 2e-6 mV). Il job disegna i video solo se tutto coincide (`[campaign check] OK` nel log). Scegliere un job **finito**; `ALLOW_MISSING=1` per un job ancora in corsa.

Senza `JOB`:

```bash
git pull                       # deve contenere i file della sezione 6
qsub jobs/video_frames.pbs     # 5 culture di config (seed, neuroni e quadrato di config.py), layer 80, 200 ms
```

Per cambiare i parametri, con `-v` (senza spazi dopo le virgole):

```bash
qsub -v NEURONS=1700,SPAN=571.75,N_CULTURES=5,LAYERS=80,TMAX=200 jobs/video_frames.pbs
qsub -v MODEL=soma_only jobs/video_frames.pbs          # un altro modello di cellula
qsub -v RENDER=0 jobs/video_frames.pbs                 # solo i dati, i video dopo
```

Il log è in `logs/estim_video_frames.log`, i risultati in `video_run_<modello>_<data>_<ora>/`.

**Disegnare i video dopo, o sul Mac** (non serve NEURON; copia la cartella dei risultati):

```bash
python make_prob_videos.py    --frames "CARTELLA/culture_frames_S*_C*.csv" --half 600 --sigma 20 --outdir CARTELLA
python make_culture_videos.py --frames "CARTELLA/culture_frames_S*_C*.csv" --list
python make_culture_videos.py --frames "CARTELLA/culture_frames_S*_C*.csv" --cultures 0 1 --layer 80 --half 600 --outdir CARTELLA
```

(`CARTELLA` = il nome vero della cartella dei risultati.)

---

## 4. Parametri da toccare

**Nel job (`qsub -v ...`):**

| Variabile | Default | Cosa cambia |
|---|---|---|
| `NEURONS` | vuoto = `config.py` | neuroni per cultura (tutte uguali); con `JOB` non va dato |
| `SPAN` | vuoto = `config.py` | metà lato del quadrato dei somata, in µm (centro da `config.placement_centre`); con `JOB` non va dato |
| `N_CULTURES`, `FIRST` | 5, 0 | quante culture e da quale indice |
| `LAYERS` | 80 | uno solo costa un terzo; `40,80,120` per tutti e tre |
| `TMAX` | 200 | ms dopo l'impulso; più lungo = più fotogrammi e più costo |
| `MODEL` | vuoto = `config.cell_model` | deve essere lo stesso modello delle mappe mostrate in tesi |
| `JOB` | vuoto | un job della campagna (es. `ftd01`): le sue culture, controllate riga per riga |
| `ALLOW_MISSING` | 0 | 1 = accetta neuroni che un job ancora in corsa non ha scritto |
| `SEED` | vuoto = `config.seed`, o quello di `JOB` | culture diverse per job diversi |
| `CHUNK` | 50 | neuroni per pezzo (bilanciamento del carico) |
| `CULTURE_IDS` | `0 1` | le due culture dei video 2 e 3 |
| `HALF` | vuoto = quadrato + 20 µm | estensione della mappa nei video (µm) |
| `SIGMA` | 20 | lisciamento del video 1 (µm) |
| `RENDER` | 1 | 0 = solo dati |

Risorse nell'intestazione del `.pbs`: `ppn=32`, `walltime=04:00:00`, coda `cpu`. Adattale al cluster.

**Nei renderer (se li lanci a mano):**

| Opzione | Default | Cosa cambia |
|---|---|---|
| `--neu` | 1.0 | soglia depol/iper in mV (quella della statistica) |
| `--fps`, `--hold-end` | 12, 2 s | velocità del video e pausa finale |
| `--n-morph` | 15 | morfologie disegnate nei video 2 e 3 |
| `--soma-color` | `dvm` | `state` per i tre stati discreti |
| `--panel` | `theta` | `r` per il pannello in funzione della distanza |
| `--metric` | `area` | `neurons` per la % di neuroni nel video 1 |

---

## 5. Uscite e controlli

In `video_run_…/`:

| File | Contenuto |
|---|---|
| `culture_frames_S<seed>_C<c>.csv` | una riga per neurone × layer × fotogramma |
| `culture_video_neurons_S<seed>.csv` | una riga per neurone × layer: attivato, ΔV_end, etichetta, r, θ |
| `electrodes.csv`, `video_manifest_S<seed>.json` | geometria; tutti i parametri e il controllo di coerenza |
| `prob_video_all.mp4`, `culture_video_<id>_L<layer>.mp4` | i tre video (GIF se manca `ffmpeg`) |
| `prob_snapshots_all.png`, `culture_snapshots_*.png` | istantanee per le slide |

**Prima di usare un video:**
- nel log: `consistency: max |DeltaV_end(...)| = 0.00e+00 mV`;
- nessun messaggio "pulse frames are FROZEN";
- fase 1 e fase 2 con polarità invertite; elettrodi grigi dopo l'impulso;
- l'attivazione sta sugli elettrodi, come nelle mappe HPC;
- nei video 2 e 3 i punti della cultura cadono per lo più dentro gli intervalli di Wilson attorno all'insieme.

**Costo:** circa 20 s per neurone × layer con finestra di 800 ms (misurato in `jobs/vm_examples.pbs`); con 200 ms costa meno. 5 × 1700 × 1 layer = 8500 simulazioni, circa 1.5 h su 32 core nel caso peggiore. Con 3 layer, il triplo.

**Prova locale prima del cluster** (Mac, con NEURON):

```bash
python smoke_video_frames.py                                     # senza NEURON: ALL PASSED
python video_frames.py --n-cultures 1 --neurons 20 --layers 80 --tmax-ms 20 --processes 4 --out video_test
```

---

## 6. File da mettere nel repository

`video_frames.py`, `smoke_video_frames.py`, `make_prob_videos.py`, `smoke_prob_videos.py`, `make_culture_videos.py`, `smoke_culture_videos.py`, `jobs/video_frames.pbs`, `video_campaign_check.py`, `smoke_video_campaign.py`.

`make_prob_videos.py` non usa più scipy (il cluster non lo ha): lo smoothing gaussiano è in numpy, identico a `scipy.ndimage.gaussian_filter(mode="constant")` (controllato da `smoke_video_campaign.py`).

Usano i file già presenti: `culture_export.py`, `rich_footprint.py`, `bump_kinetics.py`, `config.py`, `field.py`, `morphologies.py`, `slicer.py`. Il job controlla all'inizio che ci siano e che siano le versioni nuove.
