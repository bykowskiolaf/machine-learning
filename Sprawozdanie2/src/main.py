import logging
from pathlib import Path

from dane import wczytaj_i_przygotuj_dane
from Zad03 import Zadanie3
from Zad04 import Zadanie4


logger = logging.getLogger(__name__)


def _sekcja(tytul):
    logger.info("%s", "#" * 60)
    logger.info("%s", tytul)
    logger.info("%s", "#" * 60)


def main():
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    sciezka_danych = src_dir / "data" / "regresja_logistyczna_zadanie_proj_dane_PREP.csv"
    sciezka_wykresow_zad3 = project_dir / "outputs" / "plots"

    _sekcja("START: Wczytywanie i przygotowywanie danych")
    df = wczytaj_i_przygotuj_dane(sciezka_danych)
    logger.info("Wczytano rekordy: %d", len(df))

    _sekcja("ZADANIE 3")
    zad3 = Zadanie3(df)
    # TODO odkomentować
    #zad3.krzywa_logistyczna(output_path=sciezka_wykresow_zad3)
    #zad3.roc(output_path=sciezka_wykresow_zad3)
    #zad3.odds_ratios(output_path=sciezka_wykresow_zad3)
    #zad3.smokers(output_path=sciezka_wykresow_zad3)
    zad3.confusion(output_path=sciezka_wykresow_zad3)
    #zad3.graph(output_path=sciezka_wykresow_zad3)

    # TODO odkomentować
    #_sekcja("ZADANIE 4")
    #zad4 = Zadanie4(df)

    #form_bazowa = "P90_bin ~ WIEK"
    #form_2_zmienne = "P90_bin ~ WIEK + CUKRZYCA_bin"
    #form_3_zmienne = "P90_bin ~ WIEK + CUKRZYCA_bin + PAPIEROSY_bin"

    #logger.info("Porownanie modeli LRT: WIEK vs WIEK + CUKRZYCA")
    #zad4.zadanie_4c_test_lrt(form_prostsza=form_bazowa, form_zlozona=form_2_zmienne)

    #logger.info("Porownanie modeli LRT: WIEK + CUKRZYCA vs WIEK + CUKRZYCA + PAPIEROSY")
    #zad4.zadanie_4c_test_lrt(form_prostsza=form_2_zmienne, form_zlozona=form_3_zmienne)

    #logger.info("Ewaluacja modelu wielowymiarowego (train/test)")
    #zad4.zadanie_4e_f_ewaluacja(formula=form_3_zmienne, cut_off=0.5)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
