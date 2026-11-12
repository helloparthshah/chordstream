"""E33: cache 16 kHz int16 audio (data/raw16k) for every training and test song."""
import zlib
from pathlib import Path

from chordstream.data.bollywood import trusted_ids_v2
from chordstream.data.dataset import split_files
from chordstream.rawfront import prepare_raw

train, _ = split_files("data/cache_ext_bolly3x2")
stems = {f.stem if f.stem.startswith("billboard__") else "bolly__" + f.stem.split("__")[1] for f in train}
stems |= {f.stem for f in split_files("data/cache_ext")[1]}
ok = trusted_ids_v2("align_cache_bolly3.jsonl")
stems |= {f.stem for f in Path("data/cache_bolly3").glob("*.npz") if zlib.crc32(f.stem.encode()) % 1000 < 300 and f.stem.split("__")[-1] in ok}
for i, s in enumerate(sorted(stems)):
    sid = s.split("__")[-1]
    prepare_raw(s, Path("data/billboard") / f"{sid}.flac" if s.startswith("billboard__") else Path("data/bollywood/audio") / f"{sid}.flac")
    if i % 100 == 0:
        print(i, len(stems), flush=True)
print("done", len(stems))
