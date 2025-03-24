from pathlib import Path
import random

NOUN_FILE_PATH = Path(__file__).parent / 'assets' / 'nouns.txt'
ADJECTIVE_FILE_PATH = Path(__file__).parent / 'assets' / 'adjectives.txt'

def _load_words(file_path: Path) -> list[str]:
    with open(file_path, 'r') as f:
        return f.readlines()
    

NOUNS = _load_words(NOUN_FILE_PATH)
ADJECTIVES = _load_words(ADJECTIVE_FILE_PATH)


def generate_name() -> str:
    adjective = random.choice(ADJECTIVES).strip()
    noun = random.choice(NOUNS).strip()
    return f'{adjective}-{noun}'