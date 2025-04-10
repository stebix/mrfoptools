import enum
from pathlib import Path
import random

NOUN_FILE_PATH = Path(__file__).parent / 'assets' / 'nouns.txt'
ADJECTIVE_FILE_PATH = Path(__file__).parent / 'assets' / 'adjectives.txt'
ANIMALS_FILE_PATH = Path(__file__).parent / 'assets' / 'animals.txt'

class NameElement(enum.Enum):
    """
    Enum to specify the type of element for the name generator.
    """
    NOUN = 'noun'
    ADJECTIVE = 'adjective'
    ANIMAL = 'animal'

def _load_words(file_path: Path) -> list[str]:
    with open(file_path, 'r') as f:
        return f.readlines()

NOUNS = _load_words(NOUN_FILE_PATH)
ADJECTIVES = _load_words(ADJECTIVE_FILE_PATH)
ANIMALS_FILE_PATH = _load_words(ANIMALS_FILE_PATH)

ELEMENTS: dict[NameElement, list[str]] = {
    NameElement.ADJECTIVE: ADJECTIVES,
    NameElement.NOUN: NOUNS,
    NameElement.ANIMAL: ANIMALS_FILE_PATH
}


def generate_name(
    order: tuple[NameElement, ...] = (NameElement.ADJECTIVE, NameElement.NOUN, NameElement.ANIMAL),
    ) -> str:
    """
    Generate a random, but human-friendly (compared to UUID et al.)
    name consisting of various elements.
    """
    parts = [
        ELEMENTS[element][random.randint(0, len(ELEMENTS[element]) - 1)].strip().lower()
        for element in order
    ]
    name = '-'.join(parts)
    return name