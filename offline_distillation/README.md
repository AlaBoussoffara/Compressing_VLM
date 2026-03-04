# Offline Knowledge Distillation — Vision-Language

## Principe

La **distillation offline** transfère les connaissances d'un gros modèle (teacher) vers un petit modèle (student) en deux phases :

1. **Phase 1 — Génération des logits** : Le teacher (7B) fait de l'inférence sur le dataset. On sauvegarde les top-K logits (distributions de probabilité) et l'index de chaque sample dans le dataset, sur disque.

2. **Phase 2 — Entraînement du student** : Le student (2B) est entraîné à reproduire les distributions du teacher. Pour chaque sample, il **re-charge l'image depuis le dataset HuggingFace** original grâce au `dataset_index` sauvegardé en Phase 1, puis la re-traite avec son propre processor. Il reçoit donc les tokens texte du teacher ET les pixels de l'image.

La loss finale combine deux objectifs :
- **KL Divergence** (soft labels) : le student imite la distribution complète du teacher, y compris ses "hésitations" entre tokens
- **Cross-Entropy** (hard labels) : le student apprend aussi les bonnes réponses

```
L = α × T² × KL(student || teacher) + (1 − α) × CE(student, labels)
```

## Modèles

| Rôle | Modèle | Architecture | Params |
|---|---|---|---|
| Teacher | `letxbe/qwen2-7b-BoundingDocs-rephrased` | Qwen2-VL-7B fine-tuné | ~8B |
| Student | `Qwen/Qwen2-VL-2B-Instruct` | Qwen2-VL-2B | ~2B |

Le teacher est un Qwen2-VL-7B fine-tuné sur BoundingDocs avec des questions reformulées. Le student est le plus petit Qwen2-VL disponible (il n'existe pas de 1.5B VL officiel).

Les deux modèles partagent le même tokenizer (vocabulaire ~152k tokens), ce qui permet de réutiliser les `input_ids` du teacher pour le student.

## Installation

```bash
pip install torch transformers datasets accelerate bitsandbytes tqdm numpy
pip install qwen-vl-utils Pillow
```

## Usage

### Phase 1 : Générer les logits du teacher

```bash
python offline_distillation_vl_v2.py --phase generate_logits \
    --teacher_model letxbe/qwen2-7b-BoundingDocs-rephrased \
    --dataset letxbe/BoundingDocs \
    --logits_dir ./cached_logits_vl \
    --max_samples 5000
```

Ajouter `--load_in_4bit` si la VRAM est limitée (<16 Go).

**Ressources nécessaires :**
- VRAM : ~14 Go en float16, ~5 Go avec `--load_in_4bit`
- Disque : ~20-50 Go selon le nombre de samples (logits uniquement, pas de pixels stockés)
- Temps : ~1-2 secondes par sample sur un GPU type A100

### Phase 2 : Distiller le student

```bash
python offline_distillation_vl_v2.py --phase distill \
    --student_model Qwen/Qwen2-VL-2B-Instruct \
    --dataset letxbe/BoundingDocs \
    --logits_dir ./cached_logits_vl \
    --output_dir ./distilled_docexplainer \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --temperature 2.0 \
    --alpha 0.5
```

**Important** : le dataset HuggingFace doit être accessible en Phase 2 car les images sont re-chargées à la volée depuis le dataset original (elles ne sont pas stockées sur disque). Le `--dataset` doit pointer vers le même dataset que celui utilisé en Phase 1.

**Ressources nécessaires :**
- VRAM : ~12-16 Go (gradient checkpointing activé, bfloat16)
- Le dataset HuggingFace doit être accessible (téléchargé ou en cache)

## Hyperparamètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `temperature` | 2.0 | Plus élevé → distribution plus douce, plus de transfert de "dark knowledge". Plage typique : 1.5–4.0 |
| `alpha` | 0.5 | Poids de la KL loss vs CE loss. 0.7–0.9 favorise le teacher, 0.3–0.5 équilibre les deux |
| `top_k` | 64 | Nombre de logits conservés par position (codé en dur). 32–128 est un bon compromis taille/qualité |
| `lr` | 1e-5 | Learning rate. 1e-5 à 5e-5 selon la taille du dataset |
| `batch_size` | 2 | Micro batch size. Le batch effectif = `batch_size × gradient_accumulation_steps` |
| `gradient_accumulation_steps` | 16 | Batch effectif = 2 × 16 = 32 |
| `epochs` | 3 | Nombre de passes sur le dataset |
| `save_every` | 500 | Sauvegarde un checkpoint tous les N optimizer steps |

## Nombre de samples recommandé

| Samples | Usage |
|---|---|
| 100–500 | Debug : vérifier que le pipeline tourne, que la loss descend |
| 2 000–5 000 | Premier run exploratoire, signal de distillation visible |
| 10 000–20 000 | Run sérieux pour un student 2B |
| 30 000+ | Optimal (le dataset BoundingDocs train contient ~38.5k samples) |

## Structure des fichiers de logits

```
cached_logits_vl/
├── metadata.json              # Config du teacher, dataset, nombre de samples
├── processor/                 # Processor sauvegardé (tokenizer + image processor)
├── sample_000000.npz          # Logits compressés du sample 0
├── sample_000001.npz
└── ...
```

Chaque `.npz` contient :
- `input_ids` (int32) : tokens d'entrée (texte + placeholders image)
- `top_k_values` (float16) : valeurs des 64 logits les plus élevés par position
- `top_k_indices` (int32) : indices vocabulaire correspondants
- `attention_mask` (int8) : masque d'attention (1 = token réel, 0 = padding)
- `dataset_index` (int64) : indice du sample dans le dataset HuggingFace (pour retrouver l'image en Phase 2)
- `answer` (object) : réponse attendue (pour debug/évaluation)

## Architecture du pipeline

```
Phase 1 (inférence teacher)
────────────────────────────
Dataset HF ──→ format_boundingdocs_sample() ──→ messages chat Qwen2-VL
                                                       │
                    processor (7B) ◄────────────────────┘
                         │
                 input_ids + pixel_values + image_grid_thw
                         │
                    teacher 7B ──→ logits (seq_len, 152k)
                         │
                    torch.topk(64) ──→ top_k_values + top_k_indices
                         │
                    .npz sur disque (+ dataset_index)


Phase 2 (entraînement student)
──────────────────────────────
.npz ──→ input_ids, attention_mask, labels, teacher top-k
  +
dataset_index ──→ Dataset HF ──→ image PIL ──→ processor (2B) ──→ pixel_values + image_grid_thw
                                                                          │
                                           student 2B ◄──────────────────┘
                                                │
                                           student logits
                                                │
                              ┌─────────────────┴─────────────────┐
                              │                                   │
                     KL(student || teacher)              CE(student, labels)
                              │                                   │
                              └────── α × KL + (1-α) × CE ───────┘
                                                │
                                           loss.backward()
```

## Conseils

- **Plus de données = meilleur student.** Utilisez le maximum de samples pour la Phase 1.
- **Temperature** : commencez à 2.0, montez à 4.0 si le student sous-performe sur les cas ambigus.
- **Alpha** : si le student diverge, baissez alpha (moins de poids sur la KL).
- **Gradient checkpointing** : activé par défaut, réduit la VRAM d'environ 40%.
- **num_workers** : augmentez à 4–8 sur un cluster si le re-processing d'images est un bottleneck.
- **Reproductibilité** : ajoutez `torch.manual_seed(42)` au début de Phase 2 pour fixer le split train/val et le shuffle.
- **Multi-GPU** : wrappez avec `accelerate launch` pour du data parallelism.
- **Cache HF** : sur un cluster partagé, définissez `HF_HOME` vers un répertoire commun pour éviter que chaque nœud re-télécharge le dataset.

## Évaluation post-distillation

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("./distilled_docexplainer/best")
processor = AutoProcessor.from_pretrained("./distilled_docexplainer/best")
# Évaluation sur le test split de BoundingDocs
```