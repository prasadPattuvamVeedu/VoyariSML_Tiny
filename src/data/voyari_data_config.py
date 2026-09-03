from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. MAIN DATA ROOT
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "VoyariLM_DATA"
)


# ============================================================
# 3. DATA CATEGORIES
# ============================================================

RAW_ROOT = (
    DATA_ROOT
    / "01_RAW_SOURCES"
)

CLEAN_PRETRAINING_ROOT = (
    DATA_ROOT
    / "02_CLEAN_PRETRAINING"
)

PLACES_ROOT = (
    DATA_ROOT
    / "03_PLACES"
)

INSTRUCTION_ROOT = (
    DATA_ROOT
    / "04_INSTRUCTION"
)

EVALUATION_ROOT = (
    DATA_ROOT
    / "05_EVALUATION"
)

FINAL_TRAINING_ROOT = (
    DATA_ROOT
    / "06_FINAL_TRAINING"
)


# ============================================================
# 4. TOKENIZER
# ============================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "voyari_tokenizer_16k_v2.json"
)


# ============================================================
# 5. MODEL CONTEXT
# ============================================================

CONTEXT_LENGTH = 1024


# ============================================================
# 6. FINAL PRETRAINING SOURCES
# ============================================================

FINAL_PRETRAINING_ROOT = (
    FINAL_TRAINING_ROOT
    / "pretraining"
)

PRETRAINING_SOURCES = {

    "wikivoyage":
        FINAL_PRETRAINING_ROOT
        / "01_wikivoyage_corpus.txt",

    "simplewiki":
        FINAL_PRETRAINING_ROOT
        / "02_simplewiki_corpus.txt",

    "wikidata_india_travel":
        FINAL_PRETRAINING_ROOT
        / "04_wikidata_india_travel_corpus.txt",

    "unesco_india_heritage":
        FINAL_PRETRAINING_ROOT
        / "05_unesco_india_heritage_corpus.txt",
}


# ============================================================
# 7. PRETRAINING CANDIDATES
# ============================================================

PRETRAINING_CANDIDATES = {

    "alia_tourism":
        CLEAN_PRETRAINING_ROOT
        / "alia_tourism"
        / "alia_tourism_english_clean.txt",
}


# ============================================================
# 8. EXISTING FINAL SFT SOURCES
# ============================================================

FINAL_INSTRUCTION_ROOT = (
    FINAL_TRAINING_ROOT
    / "instruction"
)

FINAL_SFT_SOURCES = {

    "voyari_v9":
        FINAL_INSTRUCTION_ROOT
        / "01_voyari_v9_train.jsonl",

    "sgd_travel":
        FINAL_INSTRUCTION_ROOT
        / "02_sgd_travel_clarification_train.jsonl",
}


# ============================================================
# 9. NEW NORMALIZED SFT CANDIDATES
# ============================================================

NORMALIZED_SFT_ROOT = (
    INSTRUCTION_ROOT
    / "normalized_candidates"
)

SFT_CANDIDATES = {

    "india_travel_itineraries":
        NORMALIZED_SFT_ROOT
        / "india_travel_itineraries.jsonl",

    "multiwoz_2_2":
        NORMALIZED_SFT_ROOT
        / "multiwoz_2_2.jsonl",

    "bitext_travel":
        NORMALIZED_SFT_ROOT
        / "bitext_travel.jsonl",

    "taskmaster_2":
        NORMALIZED_SFT_ROOT
        / "taskmaster_2.jsonl",

    "travelplanner":
        NORMALIZED_SFT_ROOT
        / "travelplanner.jsonl",
}


# ============================================================
# 10. RAG / PLACE SOURCES
# ============================================================

RAG_SOURCES = {

    "geonames_india":
        PLACES_ROOT
        / "geonames_india.jsonl",

    "explore_india":
        PLACES_ROOT
        / "explore_india_destinations"
        / "Expanded_Indian_Travel_Dataset.csv",

    "indian_tourism_detailed":
        PLACES_ROOT
        / "indian_tourism_detailed"
        / "india_tourism_dataset.json",

    "top_indian_places":
        PLACES_ROOT
        / "top_indian_places"
        / "Top Indian Places to Visit.csv",
}


# ============================================================
# 11. EVALUATION SOURCES
# ============================================================

EVAL_SOURCES = {

    "travelplanner_validation":
        EVALUATION_ROOT
        / "travelplanner"
        / "validation.jsonl",

    "travelplanner_test":
        EVALUATION_ROOT
        / "travelplanner"
        / "test.jsonl",

    "v9_validation":
        EVALUATION_ROOT
        / "v9_final"
        / "validation.jsonl",

    "v9_test":
        EVALUATION_ROOT
        / "v9_final"
        / "test.jsonl",
}