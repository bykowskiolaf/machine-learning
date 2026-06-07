import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, chi2_contingency

def save_image(fig, filename):
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    output_path = project_dir / "images" / filename
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print("Zapisano wykres:", output)
    plt.close(fig)


class Zad06:
    def __init__(self):
        src_dir = Path(__file__).resolve().parent
        data_dir = src_dir / "data" / "Chi2_przykład1.csv"
        self.df = pd.read_csv(data_dir, sep='\t') # 1500 wierszy

    def run(self):
        # tabela wielodzielcza
        tab = pd.crosstab(index=self.df['Zmiany_płuca'], columns=self.df['Papierosy'])
        print(tab)
        print()

        # test chi^2 na podstawie wyznaczonej tabeli
        chi2_stat, p_value, dof, expected = chi2_contingency(tab, correction=False)
        print("Chi^2 statistic:", chi2_stat)
        print("p-value:", p_value)
        print("Degrees of freedom:", dof)
        # ekstremalnie niska p-wartość => odrzucamy H_0
        # Wartości oczekiwane - zgodne z policzonymi "ręcznie" w Excelu
        print("expected frequencies:\n", expected)
        # brak wartości <5 -> świadczy o tym, że aproksymacja chi^2 precyzyjnie oddaje faktyczny rozkład danych
        # WNIOSEK: cechy są silnie zależne

        # Wartość krytyczna na poziomie istotności 0.05 przy 2 stopniach swobody = 5.99146
        # Wartość uzyskana z testu = 631.66518
        # wart. z testu > wart. kryt => odrzucamy H_0 na założonym poziomie istotności

        # narysowany rozkład Chi^2 dla odpowiedniej ilości stopni swobody, wartość krytyczna i wartość otrzymana z danych
        x = np.arange(0, 640, 1)
        fig, ax = plt.subplots(figsize=(16, 5))
        plt.plot(x, chi2.pdf(x, df=2))
        plt.axvline(x=5.99146, color='red')
        plt.axvline(x=chi2_stat, color='purple')
        plt.xlim(0, x[-1])
        save_image(fig, "zad06_distrib.png")

        # miary siły związku
        # V-Cramera [0, 1]:
        N = 1500
        # mniej jest wierszy, r - 1 = 1
        V = np.sqrt((chi2_stat/N))
        print("Cramer's V =", V)
        # interpretacja: https://www.statology.org/interpret-cramers-v/
        # ok. 0.64 => dla 2 stopni swobody taka wartość świadczy o bardzo silnej korelacji
        # współczynnik kontyngencji Pearsona C [-1, 1]:
        # służy do weryfikowania, czy zmienne są liniowo skorelowane, dodatkowo określa czy korelacja jest pozytywna, czy negatywna
        C = np.sqrt(chi2_stat/(N + chi2_stat))
        print("Pearsons's C =", C)
        # ok. 0.54 => świadczy o umiarkowanym skorelowaniu wprost proporcjonalnym
        
