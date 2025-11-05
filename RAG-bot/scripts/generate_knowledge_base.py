import os
import json
import re
import requests
from pathlib import Path
from tqdm import tqdm
from random import choice
import string


RAW_DIR = Path("../raw_text")
KB_DIR = Path("../knowledge_base")
RAW_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)


pokemon_names = [
    "pikachu", "charizard", "bulbasaur", "squirtle", "mewtwo",
    "eevee", "snorlax", "gengar", "lapras", "jigglypuff",
    "dragonite", "alakazam", "gyarados", "machamp", "vaporeon",
    "jolteon", "flareon", "espeon", "umbreon", "scyther",
    "togekiss", "typhlosion", "blaziken", "sceptile", "swampert",
    "gardevoir", "salazzle", "incineroar", "zeraora", "zacian"
]


def random_name(length=6):
    return ''.join(choice(string.ascii_letters) for _ in range(length)).capitalize()


terms_map = {name: random_name() for name in pokemon_names}


def fetch_pokemon_data(name):
    """Берет данные о покемоне: типы, способности, статы, атаки и flavor text"""
    try:
        
        species_url = f"https://pokeapi.co/api/v2/pokemon-species/{name.lower()}/"
        r1 = requests.get(species_url, timeout=10)
        r1.raise_for_status()
        species_data = r1.json()
        flavor_texts = [entry["flavor_text"].replace("\n", " ").replace("\x0c", " ")
                        for entry in species_data["flavor_text_entries"]
                        if entry["language"]["name"] == "en"][:3]
        pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}/"
        r2 = requests.get(pokemon_url, timeout=10)
        r2.raise_for_status()
        poke_data = r2.json()
        types = [t["type"]["name"] for t in poke_data["types"]]
        abilities_poke = [a["ability"]["name"] for a in poke_data["abilities"]][:3]  # максимум 3
        stats = {s["stat"]["name"]: s["base_stat"] for s in poke_data["stats"]}
        moves = [m["move"]["name"] for m in poke_data["moves"]][:5]  # максимум 5 атак

        return {
            "types": types,
            "abilities": abilities_poke,
            "stats": stats,
            "moves": moves,
            "flavor_text": flavor_texts
        }
    except Exception as e:
        print(f"Error fetching {name}: {e}")
        return None

def replace_terms(text, mapping):
    for original, fake in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"(?<!\w){re.escape(original)}(?!\w)"
        text = re.sub(pattern, fake, text, flags=re.IGNORECASE)
    return text

print("Downloading and converting Pokemon texts...")
for i, name in enumerate(tqdm(pokemon_names, desc="Pokemons")):
    data = fetch_pokemon_data(name)
    if not data:
        continue

    for atk in data['moves']:
        if atk not in terms_map:
            terms_map[atk] = random_name()
    for ab in data['abilities']:
        if ab not in terms_map:
            terms_map[ab] = random_name()

    text_parts = [
        f"Name: {name}",
        f"Types: {', '.join(data['types'])}",
        f"Abilities: {', '.join(data['abilities'])}",
        f"Stats: {data['stats']}",
        f"Moves: {', '.join(data['moves'])}",
        "Flavor Texts:",
    ] + [f"- {ft}" for ft in data['flavor_text']]

    full_text = "\n".join(text_parts)

    raw_path = RAW_DIR / f"{i+1:03d}_{name}.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    replaced_text = replace_terms(full_text, terms_map)
    kb_path = KB_DIR / f"{i+1:03d}_{name}_kb.txt"
    with open(kb_path, "w", encoding="utf-8") as f:
        f.write(replaced_text)

with open(KB_DIR / "terms_map.json", "w", encoding="utf-8") as f:
    json.dump(terms_map, f, indent=2)

print("\nPokemon knowledge base created!")
print(f"Original texts in {RAW_DIR}/")
print(f"Knowledge base texts in {KB_DIR}/")
