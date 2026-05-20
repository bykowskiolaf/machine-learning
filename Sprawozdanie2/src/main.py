import logging
from pathlib import Path

from dane import wczytaj_i_przygotuj_dane
from Zad03 import Zadanie3
from Zad04 import Zadanie4
from Zad05 import Zadanie5

logger = logging.getLogger(__name__)

DATA_FILE = "regresja_logistyczna_zadanie_proj_dane_PREP.csv"
FORM_BAZOWA = "P90_bin ~ WIEK"
FORM_CUKRZYCA = "P90_bin ~ WIEK + CUKRZYCA_bin"
FORM_PELNA = "P90_bin ~ WIEK + CUKRZYCA_bin + PAPIEROSY_bin"

class DynamicNameWidthFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", min_width=0):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._name_width = min_width

    def format(self, record):
        self._name_width = max(self._name_width, len(record.name))
        record.name_padded = f"{record.name:<{self._name_width}}"
        return super().format(record)

def _configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        DynamicNameWidthFormatter(
            # fmt="%(asctime)s | %(levelname)-8s | %(name_padded)s | %(message)s",
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            min_width=len(__name__),
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def _sekcja(tytul):
    logger.info("%s", "#" * 60)
    logger.info("%s", tytul)
    logger.info("%s", "#" * 60)

def _sciezki_projektu():
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    sciezka_danych = src_dir / "data" / DATA_FILE
    
    # Dodane ścieżki do nowych wykresów z zadania 4
    sciezka_wykresu_zad3 = project_dir / "outputs" / "plots" / "zad03_krzywa_logistyczna.png"
    sciezka_wykresu_zad4_chi2 = project_dir / "outputs" / "plots" / "zad04_rozklad_chi2.png"
    sciezka_wykresu_zad4_cm = project_dir / "outputs" / "plots" / "zad04_macierz_pomylek.png"
    sciezka_wykresu_zad5 = project_dir / "outputs" / "plots" / "zad05_liniowosc_logitu.png"
    
    return sciezka_danych, sciezka_wykresu_zad3, sciezka_wykresu_zad4_chi2, sciezka_wykresu_zad4_cm, sciezka_wykresu_zad5


def _uruchom_zadanie_4(df, sciezka_chi2, sciezka_cm):
    _sekcja("ZADANIE 4")
    zad4 = Zadanie4(df)

    logger.info("Porownanie modeli LRT: WIEK vs WIEK + CUKRZYCA")
    zad4.zadanie_4c_test_lrt(form_prostsza=FORM_BAZOWA, form_zlozona=FORM_CUKRZYCA)

    logger.info("Porownanie modeli LRT: WIEK + CUKRZYCA vs WIEK + CUKRZYCA + PAPIEROSY")
    # Zapisujemy wynik drugiego testu, aby narysować dla niego wykres
    lrt_2 = zad4.zadanie_4c_test_lrt(form_prostsza=FORM_CUKRZYCA, form_zlozona=FORM_PELNA)
    
    # Rysujemy wykres gęstości chi-kwadrat dla decydującego testu LRT
    zad4.rysuj_rozklad_chi2(lrt_stat=lrt_2['lrt_stat'], df=lrt_2['df_diff'], output_path=sciezka_chi2)

    logger.info("Ewaluacja modelu wielowymiarowego (train/test)")
    zad4.zadanie_4e_f_ewaluacja(formula=FORM_PELNA, cut_off=0.5)

    logger.info("Wyznaczanie optymalnego cut-off dla modelu wielowymiarowego")
    cutoff_result = zad4.wyznacz_optymalny_cutoff(formula=FORM_PELNA)
    
    # Zapisujemy wynik ewaluacji z optymalnym cut-offem, by dostać macierz pomyłek
    eval_result = zad4.zadanie_4e_f_ewaluacja(formula=FORM_PELNA, cut_off=cutoff_result["best_cutoff"])
    
    # Rysujemy piękną macierz pomyłek (Heatmap)
    zad4.rysuj_macierz_pomylek(cm_list=eval_result['confusion_matrix'], output_path=sciezka_cm)


def _uruchom_zadanie_5(df, sciezka_wykresu):
    _sekcja("ZADANIE 5")
    zad5 = Zadanie5(df)
    
    # Krok 1: Sprawdzenie współliniowości dla naszego najlepszego modelu
    zad5.sprawdz_wspolliniowosc(formula=FORM_PELNA)
    
    # Krok 2: Sprawdzenie liniowości logitu dla zmiennej ciągłej (WIEK)
    # Wykonujemy test Boxa-Tidwella w obecności pozostałych predyktorów
    zad5.test_box_tidwell(zmienna_ciagla='WIEK', zmienne_dodatkowe=['CUKRZYCA_bin', 'PAPIEROSY_bin'])
    
    # Krok 3: Wygenerowanie wykresu pomocniczego do sprawozdania
    zad5.wykres_empiryczny_logit(zmienna_ciagla='WIEK', output_path=sciezka_wykresu)

def main():
    sciezka_danych, sciezka_wykresu_zad3, sciezka_wykresu_zad4_chi2, sciezka_wykresu_zad4_cm, sciezka_wykresu_zad5 = _sciezki_projektu()

    _sekcja("START: Wczytywanie i przygotowywanie danych")
    df = wczytaj_i_przygotuj_dane(sciezka_danych)
    logger.info("Wczytano rekordy: %d", len(df))

    _sekcja("ZADANIE 3")
    zad3 = Zadanie3(df)
    zad3.zadanie_3_krzywa_logistyczna(output_path=sciezka_wykresu_zad3, show_plot=False)

    _uruchom_zadanie_4(df, sciezka_wykresu_zad4_chi2, sciezka_wykresu_zad4_cm)
    _uruchom_zadanie_5(df, sciezka_wykresu_zad5)

if __name__ == "__main__":
    _configure_logging()
    main()
