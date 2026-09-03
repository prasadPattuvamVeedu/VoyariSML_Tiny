from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

project_root = Path(r"D:\voyari_sml")

TRAINING_ROOT = (
    project_root
    / "data"
    / "VoyariLM_DATA"
    / "06_FINAL_TRAINING"
)

PRETRAINING_DIR = TRAINING_ROOT / "pretraining"
INSTRUCTION_DIR = TRAINING_ROOT / "instruction"

OUTPUT_DIR = project_root/ "artifacts" / "tokenizer"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
PRETRAINING_FILES = [
    PRETRAINING_DIR / "01_wikivoyage_corpus.txt",
    PRETRAINING_DIR / "02_simplewiki_corpus.txt",
    PRETRAINING_DIR / "04_wikidata_india_travel_corpus.txt",
    PRETRAINING_DIR / "05_unesco_india_heritage_corpus.txt",
]


INSTRUCTION_FILES = [
    INSTRUCTION_DIR / "01_voyari_v9_train.jsonl",
    INSTRUCTION_DIR / "02_sgd_travel_clarification_train.jsonl",
]

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<eot>",
]

def yield_pretrainig_text():
    for path in PRETRAINING_FILES:
         print(f"Reading pretraining: {path.name}")
         with open (path,"r", encoding="utf-8", errors="replace",) as file:
             for line in file:
                 text = line.strip()
                 if text:
                     yield text


def yield_instruction_text():
    for path in INSTRUCTION_FILES:
        print(f"Reading instruction: {path.name}")
        with open(path, "r", encoding="utf-8", errors="replace") as file:
           for line in file:
               if not line.strip():
                   continue
               row = json.loads(line)
               messages = row.get("messages", [])

               for message in messages:
                   content = message.get("content", "")
                   if isinstance(content, str):
                       if content.strip():
                           yield content
                       elif isinstance(
                        content,
                        (dict, list),
                    ):

                            yield json.dumps(
                                content,
                                ensure_ascii=False,
                            )

def training_iterator():

    yield from yield_pretrainig_text()

    yield from yield_instruction_text()


tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),

)

print()
print("Starting Voyari tokenizer training...")
print()

tokenizer.train_from_iterator(
    training_iterator(),
    trainer=trainer,
)

TOKENIZER_PATH = (
    OUTPUT_DIR
    / "voyari_tokenizer_16k_v2.json"
)

tokenizer.save(
    str(TOKENIZER_PATH)
)



# testing

print()
print("Special token IDs:")
print()

for token in SPECIAL_TOKENS:
    token_id = tokenizer.token_to_id(token)
    print(f"{token:12} -> {token_id}")


test_text = "Plan a three day trip from Kochi to Munnar."

encoding = tokenizer.encode(test_text)

print()
print("Test text:")
print(test_text)

print()

print("Tokens:")
print(encoding.tokens)

print()

print("Token IDs:")
print(encoding.ids)


decoded_text = tokenizer.decode(
    encoding.ids
)

print()
print("Decoded text:")
print(decoded_text)


print()
print("Vocabulary size:")
print(tokenizer.get_vocab_size())

print()

print("Tokenizer saved to:")
print(TOKENIZER_PATH)