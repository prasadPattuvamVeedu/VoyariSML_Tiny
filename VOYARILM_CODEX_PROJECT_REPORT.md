# VoyariLM Tiny — Codex Project Report

**Inspection date:** 2026-09-03  
**Workspace inspected:** `D:\voyari_sml`  
**Method:** read-only filesystem inspection, static code tracing, line/record counts, SHA-256 comparison of likely duplicate files, tokenizer-JSON inspection, and two non-writing runtime smoke checks. `.venv/`, `.git/`, and `__pycache__/` were excluded. No data was changed and nothing was retrained.

## Executive summary

VoyariLM Tiny is currently a **data-preparation project with a working pretraining text-to-sequence path, a working (but candidate-only) SFT normalizer, and two trained tokenizer artifacts**. It is not yet an end-to-end language-model project. There is no usable decoder model, training loop, SFT loss/collator, checkpoint implementation, generation code, evaluation runner, or RAG implementation.

The most important state inconsistency is in the frozen training manifest. `training_dataset_manifest_v1.json` declares five approved pretraining corpora, including `03_india_wikipedia_corpus.txt`, but that file is missing from `06_FINAL_TRAINING/pretraining/`. The live reader and tokenizer script list only the four files that actually exist. Thus the actual runnable pretraining set is four corpora (165,508 documents), not the five-corpus/396,712,893-character set claimed by the manifest.

Other high-impact findings:

- The authoritative runtime tokenizer is `artifacts/tokenizer/voyari_tokenizer_16k_v2.json`: BPE, 16,000 tokens, ByteLevel pre-tokenization, nine special tokens with IDs 0–8.
- A common data config already exists at `src/data/voyari_data_config.py`, but no other module imports it. Live modules duplicate paths and context length.
- The normalized SFT candidate counts exactly match the expected counts in the request: 11,980 / 8,425 / 31,624 / 8,114 / 45.
- Those five normalized files are **candidates, not approved final training data**. Only Voyari V9 train (32,667) and the finalized SGD clarification supplement (3,320) are approved.
- TravelPlanner normalization excludes `reference_information`, but its `annotated_plan` is stored as a Python-repr string in the source. The normalizer does not parse that string, so the resulting assistant answer is a quoted, escaped raw structure rather than the intended readable day-by-day format.
- `src/model/token_embedding.py` is nonfunctional: it imports a nonexistent `config` package using wrong-case variable names, and `forward` is nested inside `__init__` instead of being a class method.
- There are no dependency lock/declaration files. Installed packages cannot make the project reproducible.

## 1. Project structure

### Requested top-level areas

| Path | Current contents and purpose |
|---|---|
| `src/` | Eight Python files. `data/` contains the pretraining readers/formatter/sequence builder, SFT normalizer, and a proposed common data config. `tokenizer/` contains tokenizer training and token counting. `model/` contains only a broken token embedding stub. `evaluation/`, `inference/`, `sft/`, and `training/` are empty. |
| `scripts/` | Four source scripts: final-boundary auditing, ALIA cleaning, formatted pretraining token counting, and a Kaggle downloader for Top Indian Places. Cached bytecode was ignored. |
| `configs/` | One file, `model_config.py`, containing the planned model dimensions. Nothing successfully consumes it. |
| `tests/` | Empty. There is no automated test suite. |
| `artifacts/` | Two tokenizer JSON artifacts in `artifacts/tokenizer/`; no model artifacts. |
| `data/VoyariLM_DATA/` | 191 files, about 2.4 GB. It contains a registry, raw sources, cleaned pretraining corpora, place/RAG candidates, instruction datasets, evaluation sets, and the frozen final-training boundary. |
| `checkpoints/` | Empty. |
| `logs/` | Empty. |

### Important source files

| File | What it currently does |
|---|---|
| `src/data/pretraining_documents.py` | Hardcodes four approved final pretraining files, reads blank-line-delimited documents, and yields `(source, document)` pairs. |
| `src/data/pretraining_formatter.py` | Loads tokenizer V2 at import time, tokenizes one document, and manually adds BOS/EOS. |
| `src/data/pretraining_sequences.py` | Packs formatted documents into a shared token buffer and yields 1,024-token `x`/`y` Python lists for next-token prediction. |
| `src/data/sft_normalizer.py` | Converts five raw SFT datasets to a common messages schema and globally deduplicates them. It writes only to `04_INSTRUCTION/normalized_candidates/`. |
| `src/data/voyari_data_config.py` | Proposed common path/config registry for final, candidate, RAG, and evaluation data plus tokenizer and context length. It exists but is unused by other modules. |
| `src/tokenizer/train_voyari_tokenizer.py` | Trains/saves the V2 16K ByteLevel BPE tokenizer from explicitly listed final pretraining and final instruction files. It executes training at module import/run time; there is no `main` guard. It was not run during this inspection. |
| `src/tokenizer/count_voyari_tokens.py` | Counts content tokens line-by-line across the same four live pretraining corpora. It does not include formatter-added BOS/EOS. |
| `src/model/token_embedding.py` | Intended token embedding, but currently cannot import and has an incorrectly nested `forward`. |
| `configs/model_config.py` | Planned values: vocabulary 16,000; context 1,024; model width 384; 14 layers; 6 heads; 64 head dimension; FFN 1,024; tied embeddings; no bias. These are configuration only, not an implemented ~23M model. |
| `scripts/00_audit_training_boundaries.py` | Scans final `.txt`/`.jsonl` files for literal special-token strings, line statistics, malformed JSON, and roles. |
| `scripts/clean_alia_tourism.py` | Cleans the ALIA raw JSONL into blank-line-separated English documents, removing Markdown/URLs/promotional fragments and documents shorter than 200 characters. |
| `scripts/count_formatted_pretraining_tokens.py` | Calls the real document reader and formatter to count exact formatted tokens per source. |
| `scripts/download_travel_datasets.py` | Top Indian Places Kaggle downloader. It imports `kagglehub`, which is not installed in the inspected environment. |

`data/VoyariLM_DATA/01_RAW_SOURCES/india_wikipedia/` also contains 13 numbered exploratory, downloader, resume, finalization, and audit scripts. They are not imported by the live pipeline. Many point to an old `D:\travelBuddy\voyari_llm\VoyariLM_DATA` tree, so they are historical/ad hoc utilities rather than runnable current-project modules without edits.

## 2. Dataset categories

Counts below are actual nonblank JSONL rows, CSV data rows, JSON array entries, or blank-line-delimited text documents where practical.

| Dataset | Exact relevant path(s) | Format | Category | State | Count | Used by current training path? |
|---|---|---|---|---|---:|---|
| Wikivoyage | `02_CLEAN_PRETRAINING/wikivoyage/wikivoyage_articles.jsonl`; `.../wikivoyage_corpus.txt`; frozen copy `06_FINAL_TRAINING/pretraining/01_wikivoyage_corpus.txt` | JSONL + text corpus | PRETRAINING | Cleaned, final | 30,036 documents | **Yes**, reader/formatter/sequences and tokenizer-training list |
| SimpleWiki | `02_CLEAN_PRETRAINING/simplewiki/simplewiki_articles.jsonl`; `.../simplewiki_corpus.txt`; frozen copy `06_FINAL_TRAINING/pretraining/02_simplewiki_corpus.txt` | JSONL + text corpus | PRETRAINING | Cleaned, final | 133,230 documents | **Yes** |
| India Wikipedia | `01_RAW_SOURCES/india_wikipedia/auto_v1/india_wikipedia_articles.jsonl` and `india_wikipedia_corpus.txt` | JSONL + text | PRETRAINING | Raw-area generated candidate/history; manifest claims a cleaned/frozen version | 4,472 article rows; manifest/audit literature refers to 4,304 retained | **No.** `02_CLEAN_PRETRAINING/india_wikipedia/` and final `03_india_wikipedia_corpus.txt` are missing |
| Wikidata India Travel | `02_CLEAN_PRETRAINING/wikidata_india_travel/wikidata_india_travel_articles.jsonl`; corpus; frozen `04_wikidata_india_travel_corpus.txt` | JSONL + text corpus | PRETRAINING | Cleaned, final | 2,197 documents | **Yes** |
| UNESCO India Heritage | raw `01_RAW_SOURCES/unesco_india_world_heritage/v1` and `v2_fixed`; clean `02_CLEAN_PRETRAINING/unesco_india_heritage`; frozen `05_unesco_india_heritage_corpus.txt` | JSONL + text + reports | PRETRAINING | V1 JSONL empty; V2 fixed/clean/frozen final | 45 documents | **Yes** |
| ALIA Tourism | raw `01_RAW_SOURCES/alia_tourism/.../tourism-public-en-md.base.jsonl`; clean `02_CLEAN_PRETRAINING/alia_tourism/alia_tourism_english_clean.txt` | JSONL + text | PRETRAINING | Cleaned candidate, not final | 380 raw rows / 380 cleaned documents | **No**; listed only in `PRETRAINING_CANDIDATES` |
| Voyari V9 | `04_INSTRUCTION/v9_final/{train,validation,test,all}.jsonl`; frozen train copy `06_FINAL_TRAINING/instruction/01_voyari_v9_train.jsonl` | JSONL messages | SFT / INSTRUCTION and EVALUATION | Final split/frozen train | 32,667 train; 2,106 validation; 1,820 test; 36,593 all | **Train only is approved**; tokenizer-training script reads it, but no SFT trainer exists |
| Schema Guided Dialogue / SGD | official ZIP under `04_INSTRUCTION/schema_guided_dialogue/official`; successive candidate folders; final `sgd_travel_clarification_sft_v1/train.jsonl`; frozen copy | ZIP, JSONL, JSON/TXT reports | SFT / INSTRUCTION | Raw official source, old candidates, and finalized supplement | 5,981 V1 candidates; 5,977 safe V2; 3,320 final | **Final 3,320 only is approved** and is read by tokenizer training |
| India Travel Itineraries | `04_INSTRUCTION/india_travel_itineraries/india_travel_itineraries.jsonl`; normalized candidate | JSONL | SFT / INSTRUCTION | Raw copy + normalized candidate | 13,800 raw; 11,980 normalized | **No**, candidate only |
| MultiWOZ 2.2 | `04_INSTRUCTION/multiwoz_2_2/multiwoz_2_2_train.jsonl`; normalized candidate | JSONL | SFT / INSTRUCTION | Raw copy + normalized candidate | 8,437 raw; 8,425 normalized | **No**, candidate only |
| Bitext Travel | `04_INSTRUCTION/bitext_travel/bitext_travel_train.jsonl`; normalized candidate | JSONL | SFT / INSTRUCTION | Raw copy + normalized candidate | 31,658 raw; 31,624 normalized | **No**, candidate only |
| Taskmaster-2 Travel | `04_INSTRUCTION/taskmaster_2/taskmaster_2_travel.jsonl`; normalized candidate | JSONL | SFT / INSTRUCTION | Raw copy + normalized candidate | 8,114 raw; 8,114 normalized | **No**, candidate only |
| TravelPlanner | train in `04_INSTRUCTION/travelplanner/train.jsonl`; validation/test in `05_EVALUATION/travelplanner/`; normalized train candidate | JSONL | SFT / INSTRUCTION + EVALUATION | Train candidate; validation/test eval-only | 45 train; 180 validation; 1,000 test | **No**, normalized train is candidate only |
| Wanderlust destinations | `01_RAW_SOURCES/wanderlust_ai/destinations.jsonl` | JSONL | RAG/recommendation pattern data | Synthetic | 3,370 | No |
| Wanderlust POIs | raw copy and `03_PLACES/wanderlust_unverified/wanderlust_pois.jsonl` | JSONL | RAG | Synthetic, explicitly unverified | 23,260 | No |
| GeoNames India | `03_PLACES/geonames_india.jsonl` | JSONL | RAG | Structured factual place data | 660,026 | No; registered as RAG only |
| Explore India | `03_PLACES/explore_india_destinations/Expanded_Indian_Travel_Dataset.{csv,json,xml}` | CSV, line-delimited JSON despite `.json`, XML | RAG | Structured place data; duplicated from raw | 110 | No |
| Indian Tourism Detailed | `03_PLACES/indian_tourism_detailed/india_tourism_dataset.json` | JSON array | RAG / recommendation | Detailed structured recommendation profiles; provenance should be verified | 100 | No |
| Top Indian Places | `03_PLACES/top_indian_places/Top Indian Places to Visit.csv` | CSV | RAG / recommendation | Structured attractions/review fields; duplicated from raw | 325 | No |
| Gold Eval V1 | `05_EVALUATION/gold_eval_v1/FINAL/gold_eval_v1_final.jsonl` plus drafts, batches, candidates, reports | JSONL + JSON/TXT | EVALUATION | Frozen final evaluation plus old build artifacts | 900 final cases | Never; manifest explicitly forbids training |

The CSV registry contains only 13 rows and is not comprehensive: it omits the four already-final pretraining corpora, Voyari V9, SGD, GeoNames, and Gold Eval. No code imports the CSV registry. For actual pretraining, `src/data/pretraining_documents.py` is the live source list.

## 3. Tokenizer

### Current implementation and artifacts

- **Training script:** `src/tokenizer/train_voyari_tokenizer.py`.
- **Actually loaded by preprocessing:** `artifacts/tokenizer/voyari_tokenizer_16k_v2.json` from `src/data/pretraining_formatter.py`.
- **Type:** Hugging Face `tokenizers` BPE model with `<unk>` as the unknown token.
- **Vocabulary size:** exactly 16,000 in both artifacts.
- **Normalizer:** none.
- **Post-processor:** none. BOS/EOS are not inserted automatically by the tokenizer artifact; the pretraining formatter inserts them manually.
- **ByteLevel pre-tokenizer:** `add_prefix_space=false`, `trim_offsets=true`, `use_regex=true`; the trainer supplies `ByteLevel.alphabet()` as the initial alphabet.
- **ByteLevel decoder:** `add_prefix_space=true`, `trim_offsets=true`, `use_regex=true`.
- **Trainer settings visible in source:** `vocab_size=16000`, `min_frequency=2`, nine ordered special tokens.

| Token | ID |
|---|---:|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |
| `<system>` | 4 |
| `<user>` | 5 |
| `<assistant>` | 6 |
| `<tool>` | 7 |
| `<eot>` | 8 |

### Versions present

| Artifact | Size | Status |
|---|---:|---|
| `artifacts/tokenizer/voyari_tokenizer_16k.json` | 1,093,668 bytes | Older artifact; no source file currently references it |
| `artifacts/tokenizer/voyari_tokenizer_16k_v2.json` | 1,093,744 bytes | **Authoritative/current runtime artifact** |

The two tokenizers are materially different, not duplicate files: each has 323 vocabulary entries absent from the other, 15,217 shared tokens have different IDs, and their merge tables differ. Both retain the same special-token IDs and top-level configuration.

### Hardcoded tokenizer paths and duplication

The V2 path is independently hardcoded in:

1. `src/data/pretraining_formatter.py`
2. `src/data/voyari_data_config.py`
3. `src/tokenizer/count_voyari_tokens.py`
4. `src/tokenizer/train_voyari_tokenizer.py` (output path)

The tokenizer training script also duplicates the approved training file lists rather than importing the data config. Its dictionary/list content branch is incorrectly nested beneath `if isinstance(content, str)`, so only non-empty string message contents are yielded; dictionary/list message content cannot reach the `elif`. No artifact-side provenance records the exact training inputs or package version, so the repository strongly indicates V2 is current but cannot prove which historical source snapshot produced it.

## 4. Common configuration

### Definitions found

| Value/search term | Locations | Runtime authority |
|---|---|---|
| `CONTEXT_LENGTH` / 1,024 | `configs/model_config.py` (`1_024`); `src/data/voyari_data_config.py` (`1024`); `src/data/pretraining_sequences.py` (`1024`). Comments in the sequence builder repeat 1,024. `FFN_DIM=1_024` is unrelated to context length. | For sequence production, the local constant in `pretraining_sequences.py` is authoritative. For the planned model, `configs/model_config.py` is intended but not consumed. |
| `VOCAB_SIZE` / 16,000 | `configs/model_config.py` (`16_000`); tokenizer trainer `vocab_size=16000`; both JSON artifacts contain 16,000 entries. | The loaded tokenizer artifact controls actual token IDs; model config merely agrees numerically. |
| `TOKENIZER_PATH` / V2 | Four files listed in section 3. `pretraining_formatter.py` additionally aliases it as both `Tokenizer_path` and `TOKENIZER_PATH`. | `pretraining_formatter.py` for live pretraining formatting. |
| `MAX_TOKENS` | **Not found** in project source/config. | None. |
| `voyari_tokenizer_16k` | Only the old artifact filename exists; current Python source refers to V2. | Not authoritative. |

`src/data/voyari_data_config.py` is a genuine common configuration attempt and should be extended/reused rather than replaced. However, it currently has **zero importers**, so it is not operationally authoritative. It also lacks `VOCAB_SIZE` and model dimensions, while `configs/model_config.py` holds model constants. The repository therefore has configuration files, but not a single enforced common configuration.

All eight core `src/` files/scripts plus several utilities hardcode `D:\voyari_sml`. Many historical India Wikipedia scripts and both frozen manifest renderings instead hardcode `D:\travelBuddy\voyari_llm\VoyariLM_DATA`, which no longer matches this workspace.

## 5. Pretraining pipeline

### Actual call/data flow

1. **Source selection:** `src/data/pretraining_documents.py::yield_pretraining_documents()` calls `yield_blank_separated_documents(path)` for four explicitly hardcoded frozen corpora.
2. **Document reading:** `yield_blank_separated_documents()` strips each line, accumulates nonblank lines, emits a document on a blank line, and resets the accumulator. Consecutive blank lines do not create empty documents. After EOF it emits a final accumulated document even if no trailing blank line exists.
3. **Formatting/tokenization:** caller `src/data/pretraining_sequences.py::yield_training_sequences()` calls `src/data/pretraining_formatter.py::format_pretraining_document(document)`. The formatter strips outer whitespace, calls the loaded V2 tokenizer, and returns `[BOS_ID] + encoding.ids + [EOS_ID]`.
4. **Packing:** `yield_training_sequences()` extends one shared Python-list buffer with every formatted document. It does not reset the buffer at document or source boundaries.
5. **Next-token windows:** while at least 1,025 tokens are buffered, it takes `window[:1025]`, emits `x=window[:-1]` and `y=window[1:]`, then removes 1,024 buffer entries. The last token of the old window remains as the first token of the next input window.

The runtime smoke check successfully read the first Wikivoyage document, produced a BOS-first/EOS-last formatted list, and produced `x` and `y` lists of length 1,024 with the correct one-token shift.

### Behavioral details and status

| Step | File/function | Important values | Status |
|---|---|---|---|
| Approved source list | `pretraining_documents.py::yield_pretraining_documents` | Four existing frozen files | **IMPLEMENTED / WORKING**, but conflicts with five-source frozen manifest |
| Blank-line document parsing | `yield_blank_separated_documents` | Blank line terminates current nonempty document | **IMPLEMENTED / WORKING**; final unterminated document is retained |
| Tokenizer load | module scope in `pretraining_formatter.py` | V2 path | **IMPLEMENTED / WORKING** |
| BOS/EOS | `format_pretraining_document` | BOS=1, EOS=2 | **IMPLEMENTED / WORKING** |
| Buffer packing | `yield_training_sequences` | Shared buffer across documents/sources | **IMPLEMENTED / WORKING** |
| Fixed windows and targets | same | context 1,024; needs 1,025 tokens | **IMPLEMENTED / WORKING** for full windows |
| Final short buffer | same | no padding or final yield | **PARTIAL**; leftover tokens below 1,025 are silently discarded |
| Tensors/dataset/shuffling/batching | absent | — | **NOT IMPLEMENTED** |

Different documents **are packed together**. A boundary in the shared stream is `... document A tokens, <eos>, <bos>, document B tokens ...`. A window may cross that boundary; EOS/BOS preserve the boundary semantically, but packing is continuous.

The actual live source total is 165,508 documents: 30,036 Wikivoyage + 133,230 SimpleWiki + 2,197 Wikidata India Travel + 45 UNESCO. ALIA and India Wikipedia are not included.

## 6. SFT / instruction pipeline

`src/data/sft_normalizer.py` implements normalization only. It does not train a model.

### Source-to-normalized mappings

| Source | Source schema observed | Mapping to normalized schema |
|---|---|---|
| India Travel Itineraries | `{instruction, input, output}` | `user = clean(instruction) + "\n\n" + clean(input)` (nonempty parts); `assistant = clean(output)`; no source metadata |
| Bitext Travel | `{instruction, intent, category, tags, response}` | `user=instruction`; `assistant=response`; metadata retains `intent`, `category`, `tags` |
| MultiWOZ 2.2 | `{dialogue_id, services, turns}` where `turns` may be dictionary-of-lists or list-of-dicts | Speakers normalized to user/assistant; utterances become messages; metadata retains `dialogue_id` and `services` |
| Taskmaster-2 | `{conversation_id, instruction_id, utterances, voyari_source, voyari_domain}` | Each utterance speaker/text becomes a message; metadata retains `conversation_id` and `voyari_domain` (not `instruction_id`) |
| TravelPlanner | `{org, dest, days, ..., query, level, annotated_plan, reference_information}` | `user=query`; intended assistant day plan from `annotated_plan`; metadata retains origin/destination/days/people/budget/level and safety flags |

All successful records use:

```json
{
  "id": "first-16-hex-chars-of-sha256",
  "source": "dataset_name",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {}
}
```

### Implemented behavior

- `clean_text`: converts values to strings, normalizes CRLF/CR to LF, collapses whitespace within each line, removes blank lines, and rejoins retained lines with LF. It does not remove URLs, HTML, templated placeholders, personally identifying data, or factual staleness.
- `normalize_role`: maps numeric `0/1` and common `USER/HUMAN/CUSTOMER` vs `ASSISTANT/SYSTEM/AGENT/BOT` labels. Notably, a dataset speaker labeled `SYSTEM` becomes the assistant role, not Voyari's special system role.
- `merge_adjacent_messages`: drops unsupported roles/empty contents and joins consecutive same-role messages with a newline.
- `prepare_messages`: removes leading messages until the first user, removes trailing messages until the last assistant, requires at least two messages and at least one of each role. After adjacent-role merging, retained conversations alternate by construction.
- `conversation_hash`: lowercases message content, collapses all whitespace, includes roles, and returns SHA-256. `build_record` uses the first 16 hex characters as the ID.
- `write_normalized_datasets`: uses one `seen_hashes` set across all five sources, so deduplication is both within-source and cross-source, in source iteration order. It overwrites each normalized candidate output when run.
- Output files are the five JSONLs under `04_INSTRUCTION/normalized_candidates/`.
- TravelPlanner explicitly does **not** read `reference_information`; metadata says `training_use=planning_structure_only` and `needs_fact_verification=true`.

### Partial or missing behavior

- **TravelPlanner is PARTIAL.** The source stores `annotated_plan` as a string containing Python literals. `format_travelplanner_plan()` only recursively walks actual dictionaries/lists; it does not parse this string. The fallback JSON-encodes the entire raw Python-repr string. The inspected normalized answer begins as an extra-quoted escaped structure and includes trailing empty dictionaries, rather than the intended `Day 1:` format.
- **Token-length filtering: NOT IMPLEMENTED.** The normalizer neither loads the tokenizer nor references `CONTEXT_LENGTH`.
- **Context-length enforcement/truncation: NOT IMPLEMENTED.**
- **Chat serialization with `<system>/<user>/<assistant>/<eot>`: NOT IMPLEMENTED.**
- **Assistant-only loss masking: NOT IMPLEMENTED.** There is no collator or SFT training loop.
- **Dataset approval/promotion: NOT IMPLEMENTED in this script.** It deliberately writes candidates, and no code promotes them into `06_FINAL_TRAINING`.

## 7. Normalized SFT data

| Actual file | Rows | Malformed JSON | Top-level schema check | Status |
|---|---:|---:|---|---|
| `04_INSTRUCTION/normalized_candidates/india_travel_itineraries.jsonl` | 11,980 | 0 | All checked records use `id,source,messages,metadata` | Candidate |
| `04_INSTRUCTION/normalized_candidates/multiwoz_2_2.jsonl` | 8,425 | 0 | Same | Candidate |
| `04_INSTRUCTION/normalized_candidates/bitext_travel.jsonl` | 31,624 | 0 | Same | Candidate |
| `04_INSTRUCTION/normalized_candidates/taskmaster_2.jsonl` | 8,114 | 0 | Same | Candidate |
| `04_INSTRUCTION/normalized_candidates/travelplanner.jsonl` | 45 | 0 | Same, but assistant content has the parsing defect above | Candidate |

The counts exactly match the approximate expectations supplied in the request. The inspection also found no row with fewer than two messages in these files. This structural result does not imply approval, token-length safety, factual quality, or correct assistant-only training behavior.

## 8. Final training data

### Final pretraining data actually present and read

1. `06_FINAL_TRAINING/pretraining/01_wikivoyage_corpus.txt` — 30,036 documents
2. `06_FINAL_TRAINING/pretraining/02_simplewiki_corpus.txt` — 133,230 documents
3. `06_FINAL_TRAINING/pretraining/04_wikidata_india_travel_corpus.txt` — 2,197 documents
4. `06_FINAL_TRAINING/pretraining/05_unesco_india_heritage_corpus.txt` — 45 documents

Their hashes match the corresponding cleaned copies and the hashes recorded in the SHA manifest. `03_india_wikipedia_corpus.txt` is **missing**, even though both metadata manifests list it and the text manifest says five pretraining sources.

### Final instruction data actually approved

1. `06_FINAL_TRAINING/instruction/01_voyari_v9_train.jsonl` — 32,667 rows
2. `06_FINAL_TRAINING/instruction/02_sgd_travel_clarification_train.jsonl` — 3,320 rows

These frozen copies exactly match the corresponding `04_INSTRUCTION` source files by SHA-256. The approved total is 35,987 SFT rows.

### New candidates not approved

All five files in `04_INSTRUCTION/normalized_candidates/` are unapproved. ALIA in `02_CLEAN_PRETRAINING/alia_tourism/` is also only a pretraining candidate. Earlier SGD V1/V2 files, SGD schema-alignment analysis artifacts, raw/copy datasets in `04_INSTRUCTION`, and Voyari `all.jsonl`, validation, and test are not approved training inputs.

## 9. RAG / places

| Dataset | Format and content | Trust/use classification |
|---|---|---|
| GeoNames India | 660,026 JSONL rows with names, alternate names, coordinates, feature/admin codes, population, timezone, source/license | Structured factual place index; appropriate as a retrieval candidate, not prose pretraining |
| Explore India | 110 records in CSV, line-delimited JSON (misnamed `.json`), and XML; destination/state/region/category/attraction/access fields | Small structured factual candidate; raw and places copies are exact duplicates |
| Indian Tourism Detailed | 100-element JSON array with extensive recommendation, season, budget, safety, itinerary, accommodation, and provenance-ish fields | Recommendation data; factual freshness/provenance needs validation |
| Top Indian Places | 325-row CSV with attraction type, ratings/review counts, fees, access, closure, and visit-time fields | Recommendation data with time-sensitive fields; header has an unnamed first column |
| Wanderlust | 3,370 synthetic destination profiles in raw; 23,260 POIs duplicated into `wanderlust_unverified` | Synthetic recommendation patterns / explicitly unverified facts |
| `india_tourism_ogd` | Empty directories in both raw and places trees | Missing dataset |

Searches found no embedding generator, embedding artifact, FAISS, Chroma, pgvector, vector database, retriever, ranking logic, indexing job, or runtime grounding code. `RAG_SOURCES` is only a path dictionary.

**RAG pipeline: NOT IMPLEMENTED.**

## 10. Evaluation and leakage risk

### Evaluation data present

- **Gold Eval V1:** frozen `05_EVALUATION/gold_eval_v1/FINAL/gold_eval_v1_final.jsonl`, 900 cases, manifest status `FROZEN`, `evaluation_only=true`, `training_allowed=false`. Numerous 40/100-case batches, drafts, candidates, review batches, and semantic-audit artifacts remain beside it.
- **TravelPlanner:** `05_EVALUATION/travelplanner/validation.jsonl` (180) and `test.jsonl` (1,000), exact copies of the raw-source validation/test files.
- **Voyari V9:** `05_EVALUATION/v9_final/validation.jsonl` (2,106) and `test.jsonl` (1,820), exact copies of the held-out files under `04_INSTRUCTION/v9_final/`.
- Gold Eval metadata reports zero exact overlap with V9 held-out data; the final training manifest reports zero exact instruction overlap with Gold Eval. These are recorded audit claims, not recomputed semantic-leakage guarantees in executable training code.

### Leakage assessment

Current explicit lists are reasonably safe: pretraining reads only the four final `.txt` files; tokenizer training lists only final pretraining plus the two final instruction train files. No current code reads `05_EVALUATION` for training.

Risks remain high for future broad/glob-based code:

1. TravelPlanner validation/test also live under `01_RAW_SOURCES`, not only `05_EVALUATION`.
2. V9 `all.jsonl` (which contains train + validation + test) lives beside `train.jsonl` under `04_INSTRUCTION/v9_final`.
3. V9 validation/test also live under the instruction category as well as evaluation.
4. Gold Eval has many complete-looking drafts/candidates/review batches in addition to the frozen final file.
5. Candidate and final SFT files coexist under `04_INSTRUCTION`; folder-wide discovery would double-count source examples and could ingest held-out data.
6. There is no trainer enforcing the final manifest, hash allowlist, or `training_allowed=false` metadata.

The safest rule is to train only from an existence-checked, hash-verified allowlist under `06_FINAL_TRAINING`, never by scanning `01_RAW_SOURCES`, `04_INSTRUCTION`, or `05_EVALUATION`.

## 11. Model implementation

| Component | Status | Exact evidence/path |
|---|---|---|
| TokenEmbedding | **PARTIAL / NONFUNCTIONAL** | `src/model/token_embedding.py`; intended `nn.Embedding`, but import fails (`No module named 'config'`), requested names `vocab_size`/`D_model` do not match uppercase config names, and `forward` is nested inside `__init__` |
| RMSNorm | **NOT IMPLEMENTED** | No implementation found |
| RoPE / rotary embeddings | **NOT IMPLEMENTED** | No implementation found |
| Causal self-attention | **NOT IMPLEMENTED** | No implementation found |
| SwiGLU | **NOT IMPLEMENTED** | No implementation found |
| Transformer block | **NOT IMPLEMENTED** | No implementation found |
| Decoder-only language model | **NOT IMPLEMENTED** | No implementation found |
| Training loop | **NOT IMPLEMENTED** | `src/training/` empty; no backward/optimizer logic |
| Optimizer | **NOT IMPLEMENTED** | No Adam/AdamW or optimizer construction |
| Scheduler | **NOT IMPLEMENTED** | None found |
| Checkpointing | **NOT IMPLEMENTED** | Project `checkpoints/` empty; an old dataset-fetch script's download checkpoint is unrelated |
| Generation/inference | **NOT IMPLEMENTED** | `src/inference/` empty |

The ~23M architecture is therefore a plan represented only by numeric constants in `configs/model_config.py`.

## 12. Dependencies

No `requirements.txt`, `pyproject.toml`, lock file, Conda environment file, Pipfile, `setup.py`, or `setup.cfg` exists. Dependencies are **not reproducible** from the repository.

The currently selected Python environment reports:

| Package | Installed state |
|---|---|
| Python | 3.13.5 (Anaconda build) |
| PyTorch | 2.8.0 installed, but not declared |
| tokenizers | 0.22.2 installed, but not declared |
| datasets | 2.14.6 installed, but not declared |
| transformers | 5.2.0 installed, but not declared |
| NumPy | 2.3.2 installed, but not declared |
| kagglehub | Not installed; required by `download_travel_datasets.py` |
| accelerate | Not installed |

Historical India Wikipedia scripts additionally import packages such as `requests`, `mwparserfromhell`, and `pyarrow`; none is declared by the project.

## 13. Duplicates, old files, and risks

### Configuration/code duplication

- Context length 1,024 appears in three configuration/runtime locations.
- Vocabulary size 16,000 appears in model config, tokenizer trainer, and artifacts.
- The V2 tokenizer path appears in four Python files.
- Final pretraining and instruction file lists are duplicated between `voyari_data_config.py`, tokenizer scripts, and the actual reader.
- `PROJECT_ROOT` is hardcoded repeatedly instead of deriving from the repository or importing common config.
- `voyari_data_config.py` is not used, so changing it alone changes no live behavior.

### Exact duplicate data copies

SHA-256 comparisons confirmed exact copies between raw/category/final trees for Bitext, India itineraries, MultiWOZ, Taskmaster, TravelPlanner train/validation/test, Explore India CSV/JSON/XML, Indian Tourism Detailed files, Top Indian Places, Wanderlust POIs, the four live final pretraining corpora, V9 train/validation/test, and SGD finalized train. The duplication appears intentional for category/freeze boundaries, but it makes broad file discovery dangerous.

### Old/candidate/confusing artifacts

- Two materially different tokenizer artifacts exist; only the `_v2` file is current.
- Multiple SGD generations (`v1_candidate`, `v2_voyari_candidate`, final clarification, V3 schema analysis) remain together.
- Gold Eval retains shells, draft, V1/V2 complete candidates, review batches, and final frozen data.
- The India Wikipedia directory contains exploratory test scripts, two collectors (original/resume), audit/finalize scripts, raw outputs, and stale old-root paths.
- `scripts/download_travel_datasets.py` is a one-off downloader for data already present and is not usable in the current environment because `kagglehub` is absent.
- Python bytecode caches exist in the repository data/script areas, although they were excluded from analysis as requested.
- Empty directories (`fineweb_edu`, `india_tourism_ogd`, model training/inference/evaluation folders, tests, logs, checkpoints) can imply planned capabilities that do not exist.

### Highest risks

1. Frozen manifest says five pretraining sources while the actual final directory and code contain four.
2. Model configuration/import naming is inconsistent (`configs` vs `config`, uppercase vs mixed/lowercase).
3. No allowlist enforcement or hash verification exists inside a training entry point because no trainer exists.
4. Evaluation files and aggregate `all.jsonl` are physically close to training candidates.
5. TravelPlanner candidate output is structurally valid JSONL but semantically malformed for the intended readable-plan conversion.
6. Unverified/synthetic RAG data could be treated as factual if future code consumes all of `03_PLACES` indiscriminately.

## 14. Actual current pipeline

```text
RAW SOURCES                                      [WORKING as stored data]
    |
    +--> source-specific cleaning/finalization   [PARTIAL]
    |      - ALIA cleaner exists
    |      - many corpora were produced historically
    |      - no unified reproducible build pipeline
    v
CLEAN / NORMALIZE
    +--> final pretraining text                  [WORKING, 4 live corpora]
    +--> SFT normalized_candidates               [WORKING/PARTIAL]
           (5 outputs valid; TravelPlanner content defect; not approved)
    |
    v
TOKENIZER (16K ByteLevel BPE V2)                 [WORKING]
    |
    v
PRETRAINING DOCUMENTS -> BOS/EOS -> BUFFER
-> 1,024-token x/y sequences                     [WORKING/PARTIAL]
    |                                             (lists only; tail dropped)
    v
MODEL                                             [NOT IMPLEMENTED]
    |
    v
PRETRAINING                                      [NOT IMPLEMENTED]
    |
    v
SFT                                              [NOT IMPLEMENTED]
    |                                             (normalization only)
    v
EVALUATION                                       [NOT IMPLEMENTED]
    |                                             (datasets exist; no runner)
    v
RAG / RUNTIME                                    [NOT IMPLEMENTED]
```

## 15. Authoritative current state

| Item | Authoritative current file/value |
|---|---|
| Tokenizer | `artifacts/tokenizer/voyari_tokenizer_16k_v2.json` |
| Vocabulary size | 16,000 (artifact truth; agrees with config/trainer) |
| Special tokens | `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`, `<system>=4`, `<user>=5`, `<assistant>=6`, `<tool>=7`, `<eot>=8` |
| Context length | 1,024; live sequence authority is `src/data/pretraining_sequences.py::CONTEXT_LENGTH` |
| Pretraining source registry | Operational: hardcoded list in `src/data/pretraining_documents.py`; descriptive frozen manifest is inconsistent because source 03 is missing |
| Pretraining document reader | `src/data/pretraining_documents.py::yield_blank_separated_documents` and `yield_pretraining_documents` |
| Pretraining formatter | `src/data/pretraining_formatter.py::format_pretraining_document` |
| Pretraining sequence builder | `src/data/pretraining_sequences.py::yield_training_sequences` |
| SFT normalizer | `src/data/sft_normalizer.py` |
| Normalized SFT directory | `data/VoyariLM_DATA/04_INSTRUCTION/normalized_candidates/` (candidate only) |
| Final pretraining directory | `data/VoyariLM_DATA/06_FINAL_TRAINING/pretraining/` (four actual files; manifest's fifth file missing) |
| Final instruction directory | `data/VoyariLM_DATA/06_FINAL_TRAINING/instruction/` (two approved files, 35,987 rows) |
| RAG directory | `data/VoyariLM_DATA/03_PLACES/`; data only, no implementation |
| Evaluation directory | `data/VoyariLM_DATA/05_EVALUATION/`; frozen Gold Eval plus held-out/candidate artifacts |
| Model implementation | None usable; `src/model/token_embedding.py` is a broken stub |
| Training implementation | Not implemented; `src/training/` empty |
| Inference implementation | Not implemented; `src/inference/` empty |

## 16. Next 5 safest steps

1. **Reconcile and re-freeze the training boundary before any model work.** Decide whether India Wikipedia is intentionally excluded or restore the exact hash-listed `03_india_wikipedia_corpus.txt`; then make the manifest, SHA list, common config, document reader, tokenizer provenance, and totals agree. Do not silently change the frozen V1 definition.
2. **Make the existing common config operational instead of creating another config.** Have readers/counters/trainers import one source allowlist, tokenizer path, and context length from `src/data/voyari_data_config.py` (with model constants consistently exposed from `configs/model_config.py`). Remove hardcoded roots in live code by deriving the project root safely.
3. **Add read-only boundary validation and tests before training.** Tests should require every approved path and hash, reject anything under `05_EVALUATION`, reject V9 `all/validation/test`, reject `normalized_candidates` and RAG sources, and verify blank-line EOF behavior, BOS/EOS packing, 1,024 shifts, and final-buffer policy.
4. **Finish and approve SFT preprocessing deliberately.** Parse TravelPlanner's Python-literal strings safely, keep `reference_information` excluded, serialize conversations with the existing V2 tokenizer, filter to the chosen context policy, and implement/test assistant-only labels. Review/deduplicate candidates against final V9, SGD, and all evaluation prompts before a versioned promotion into `06_FINAL_TRAINING`.
5. **Implement the model and trainers in small verified layers.** First fix package/config naming and implement/test embedding, RMSNorm, RoPE, causal attention, SwiGLU, blocks, and decoder LM; then add tensor datasets, optimizer/scheduler, checkpointing, pretraining, SFT, generation, and evaluation. Keep RAG indexing/runtime separate and allowlist only verified place sources; never train on raw/unverified RAG data by directory scan.
