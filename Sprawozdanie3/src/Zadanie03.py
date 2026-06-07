import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import levene, shapiro, ttest_rel

class Zad03:
    def __init__(self):
        src_dir = Path(__file__).resolve().parent
        data_dir = src_dir / "data" / "zad03.csv"
        self.df = pd.read_csv(data_dir, sep='\t') # 22 wiersze

    def run(self):
        tetno_1, tetno_2 = self.df['Tętno I'], self.df['Tętno II']
        # normalność rozkładu
        # Test Shapiro-Wilka: H_0 - dane mają rozkład normalny
        result_1 = shapiro(tetno_1)
        result_2 = shapiro(tetno_2)
        print("p-wartości testu Shapiro-Wilka:")
        print("Tętno I:", result_1.pvalue)
        print("Tętno II:", result_2.pvalue)
        # Obie wartości > .05 => obie zmienne mają rozkład normalny

        # równość wariancji
        # Test Levene'a: H_0 - wariancje są równe
        result = levene(tetno_1, tetno_2)
        print("p-wartość testu Levene'a:")
        print(result.pvalue)
        # Wartość > .05 => wariancje są równe

        # Test t-studenta dla zmiennych powiązanych - ponieważ wyniki obserwacji są zestawione w pary, istnieją dwie serie wyników dla tych samych elementów. H_0 - średnie są równe
        #result = ttest_rel(tetno_1, tetno_2)
        #print("p-wartość testu t-Studenta:")
        #print(result.pvalue)
        # Wartość b. niska (< .01) => średnie nie są równe
        # ^^^ zakomentowałem, bo to chyba niewłaściwy test


        # wartość krytyczna na poziomie istotności 0.05
            # n - 1 = 22 - 1 = 21 stopni swobody
            # na podst.: https://www.stat.purdue.edu/~lfindsen/stat503/t-Dist.pdf
        # alpha = 1.721
        # różnice
        n = 22
        d = tetno_2 - tetno_1
        mean = np.mean(d)
        stddev = np.std(d)
        # wartość testu ze wzoru ze slajdu
        t = mean/stddev * np.sqrt(n)
        print("Wartość testu:", t) # t > alpha => odrzucamy H_0 na poziomie istotności 0.05 => średnie nie są równe
        # oznacza to, że wyniki NIE PRZECZĄ hipotezie, że wysiłek wpływa na przyspieszenie tętna
