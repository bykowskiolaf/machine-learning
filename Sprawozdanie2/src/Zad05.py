import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

class Zadanie5:
    def __init__(self, df):
        self.df = df.copy()

    def _zmienne_z_formuly(self, formula):
        """Wydobywa nazwy zmiennych niezależnych z formuły"""
        prawa_strona = formula.split("~")[1]
        return [z.strip() for z in prawa_strona.replace(" ", "").split("+")]

    def sprawdz_wspolliniowosc(self, formula):
        """Oblicza Variance Inflation Factor (VIF) dla zmiennych w modelu."""
        logger.info("%s", "=" * 50)
        logger.info("ZADANIE 5: Sprawdzenie współliniowości (VIF)")
        logger.info("Model: %s", formula)
        logger.info("%s", "=" * 50)

        zmienne = self._zmienne_z_formuly(formula)
        df_clean = self.df.dropna(subset=zmienne)

        # Do VIF potrzebujemy wyodrębnić tylko zmienne objaśniające (X) i dodać stałą
        X = df_clean[zmienne]
        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Zmienna"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        for index, row in vif_data.iterrows():
            if row['Zmienna'] != 'const':
                logger.info("Zmienna: %-15s | VIF: %.4f", row['Zmienna'], row['VIF'])
        
        # Wartość VIF < 5 oznacza brak problemu ze współliniowością
        max_vif = vif_data[vif_data['Zmienna'] != 'const']['VIF'].max()
        if max_vif < 5:
            logger.info("Wniosek: Brak problemu współliniowości (wszystkie VIF < 5).")
        else:
            logger.warning("Wniosek: Możliwy problem ze współliniowością! VIF przekracza 5.")

        return vif_data

    def test_box_tidwell(self, zmienna_ciagla='WIEK', zmienne_dodatkowe=None):
        """Wykonuje test Boxa-Tidwella sprawdzający założenie o liniowości logitu."""
        logger.info("%s", "=" * 50)
        logger.info("ZADANIE 5: Liniowość logitu (Test Boxa-Tidwella)")
        logger.info("Zmienna ciągła: %s", zmienna_ciagla)
        logger.info("%s", "=" * 50)

        zmienne_dodatkowe = zmienne_dodatkowe or []
        wszystkie_zmienne = ['P90_bin', zmienna_ciagla] + zmienne_dodatkowe
        df_clean = self.df.dropna(subset=wszystkie_zmienne).copy()

        # Obliczanie członu interakcji: zmienna * log(zmienna)
        interakcja_nazwa = f"{zmienna_ciagla}_ln_{zmienna_ciagla}"
        df_clean[interakcja_nazwa] = df_clean[zmienna_ciagla] * np.log(df_clean[zmienna_ciagla])

        # Budowa formuły dla testu Boxa-Tidwella
        formula_bt = f"P90_bin ~ {zmienna_ciagla} + {interakcja_nazwa}"
        if zmienne_dodatkowe:
            formula_bt += " + " + " + ".join(zmienne_dodatkowe)

        model_bt = smf.logit(formula_bt, data=df_clean).fit(disp=False)
        p_value_interakcji = model_bt.pvalues[interakcja_nazwa]

        logger.info("p-value dla członu interakcji (%s): %.6f", interakcja_nazwa, p_value_interakcji)
        
        if p_value_interakcji < 0.05:
            logger.warning("Wniosek: Odrzucamy H0 (p < 0.05). ZŁAMANE założenie o liniowości dla %s.", zmienna_ciagla)
        else:
            logger.info("Wniosek: Brak podstaw do odrzucenia H0 (p >= 0.05). Założenie o liniowości SPEŁNIONE.")

        return p_value_interakcji

    def wykres_empiryczny_logit(self, zmienna_ciagla='WIEK', output_path=None):
        """Wizualizuje liniowość logitu dzieląc zmienną ciągłą na koszyki."""
        df_clean = self.df.dropna(subset=[zmienna_ciagla, 'P90_bin']).copy()

        # Dzielimy dane na równe kwartyle/koszyki (np. 5 koszyków)
        df_clean['Koszyk'] = pd.qcut(df_clean[zmienna_ciagla], q=5, duplicates='drop')

        grupy = df_clean.groupby('Koszyk', observed=True).agg({
            zmienna_ciagla: 'mean',
            'P90_bin': ['sum', 'count']
        })
        grupy.columns = ['Srodek_koszyka', 'Liczba_1', 'Liczba_obserwacji']
        grupy['Prawdopodobienstwo'] = grupy['Liczba_1'] / grupy['Liczba_obserwacji']

        # Unikamy log(0)
        eps = 1e-4
        grupy['Prawdopodobienstwo'] = np.clip(grupy['Prawdopodobienstwo'], eps, 1 - eps)
        grupy['Logit'] = np.log(grupy['Prawdopodobienstwo'] / (1 - grupy['Prawdopodobienstwo']))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(x=grupy['Srodek_koszyka'], y=grupy['Logit'], ax=ax, scatter_kws={'s': 100})
        ax.set_title(f"Zadanie 5: Empiryczny logit dla zmiennej {zmienna_ciagla}")
        ax.set_xlabel(f"{zmienna_ciagla} (Środek koszyka)")
        ax.set_ylabel("Logit (log-odds)")

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            logger.info("Zapisano wykres liniowości logitu: %s", output)

        plt.close(fig)