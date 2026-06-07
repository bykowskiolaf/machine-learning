import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

class Zad07:
    def __init__(self):
        src_dir = Path(__file__).resolve().parent
        data_dir = src_dir / "data" / "Chi2_przykład2.csv"
        self.df = pd.read_csv(data_dir, sep='\t')

    def run(self):
        tab = pd.crosstab(index=self.df['Choroba wieńcowa'], columns=self.df['Ciśnienie'])
        print(tab)
        print()

        chi2_stat, p_value, dof, expected = chi2_contingency(tab, correction=True)
        # Przed zastosowaniem poprawki:
        # Chi^2 statistic: 26.23497454415329
        # p-value: 3.022977172210045e-07
        print("Chi^2 statistic:", chi2_stat)
        print("p-value:", p_value)
        # Wysoka p-wartość => brak przesłanek do stwierdzenia korelacji
        print("Degrees of freedom:", dof)
        print("expected frequencies:\n", expected)

        # Wartość krytyczna na poziomie istotności 0.05 przy 1 stopniu swobody = 3.84146
        # wart. z testu > wart. kryt => są przesłanki do odrzucenia H_0 na założonym poziomie istotności, czyli do stwierdzenia korelacji

        # miary siły związku
        # V-Cramera [0, 1]:
        N = 100
        # wierszy i kolumn tyle samo, zakładamy r - 1 = 1
        V = np.sqrt((chi2_stat/N))
        print("Cramer's V =", V)
        # interpretacja: https://www.statology.org/interpret-cramers-v/
        # ok. 0.48 => dla 2 stopni swobody taka wartość świadczy o umiarkowanie silnej korelacji
        # współczynnik kontyngencji Pearsona C [-1, 1]:
        # służy do weryfikowania, czy zmienne są liniowo skorelowane, dodatkowo określa czy korelacja jest pozytywna, czy negatywna
        C = np.sqrt(chi2_stat/(N + chi2_stat))
        print("Pearsons's C =", C)
        # ok. 0.44 => świadczy o umiarkowanym skorelowaniu wprost proporcjonalnym
