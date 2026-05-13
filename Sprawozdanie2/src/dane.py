import pandas as pd

BINARNE_KOLUMNY = ["P90", "PAPIEROSY", "NADCISNIENIE", "MIGRENA", "CUKRZYCA"]
WYMAGANE_KOLUMNY = ["P90", "WIEK", "PAPIEROSY", "CUKRZYCA"]
MAPOWANIE_BINARNE = {"Tak": 1, "Nie": 0, True: 1, False: 0, 1: 1, 0: 0}


def _sprawdz_kolumny(df, wymagane_kolumny):
    brakujace = [kol for kol in wymagane_kolumny if kol not in df.columns]
    if brakujace:
        raise ValueError(f"Brakuje wymaganych kolumn: {', '.join(brakujace)}")


def _mapuj_kolumne_binarna(df, kolumna):
    wynik = df[kolumna].map(MAPOWANIE_BINARNE)

    nieznane_mask = df[kolumna].notna() & wynik.isna()
    if nieznane_mask.any():
        nieznane = sorted(df.loc[nieznane_mask, kolumna].astype(str).unique().tolist())
        raise ValueError(
            f"Kolumna '{kolumna}' ma nieobslugiwane wartosci: {', '.join(nieznane)}"
        )

    df[f"{kolumna}_bin"] = wynik


def wczytaj_i_przygotuj_dane(filepath, wymagane_kolumny=None):
    """Wczytuje dane TSV, waliduje kolumny i tworzy *_bin dla zmiennych binarnych."""
    df = pd.read_csv(filepath, sep="\t")

    if wymagane_kolumny is None:
        wymagane_kolumny = WYMAGANE_KOLUMNY
    _sprawdz_kolumny(df, wymagane_kolumny)

    for kolumna in BINARNE_KOLUMNY:
        if kolumna in df.columns:
            _mapuj_kolumne_binarna(df, kolumna)

    return df
