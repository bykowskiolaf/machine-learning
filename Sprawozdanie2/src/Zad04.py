import logging

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class Zadanie4:
    def __init__(self, df, random_state=42):
        self.df = df
        self.random_state = random_state

    def _zmienne_z_formuly(self, formula):
        return [z.strip() for z in formula.replace("~", "+").split("+")]

    def _przygotuj_dane(self, formula):
        zmienne = self._zmienne_z_formuly(formula)
        return self.df.dropna(subset=zmienne)

    def _trenuj_i_predykcja(self, formula, test_size):
        df_clean = self._przygotuj_dane(formula)
        train_data, test_data = train_test_split(
            df_clean,
            test_size=test_size,
            random_state=self.random_state,
        )
        model = smf.logit(formula, data=train_data).fit(disp=False)
        y_true = test_data["P90_bin"]
        y_pred_prob = model.predict(test_data)
        return train_data, test_data, model, y_true, y_pred_prob

    def zadanie_4c_test_lrt(self, form_prostsza, form_zlozona):
        """
        Zadanie 4c i 4g: Test Ilorazu Wiarygodności (LRT).
        Sprawdza, czy bardziej złożony model istotnie poprawia dopasowanie.
        """
        logger.info("%s", "=" * 50)
        logger.info("ZADANIE 4c: Test LRT")
        logger.info("Model Prostszy: %s", form_prostsza)
        logger.info("Model Złożony:  %s", form_zlozona)
        logger.info("%s", "=" * 50)
        
        zmienne = sorted(
            set(self._zmienne_z_formuly(form_prostsza) + self._zmienne_z_formuly(form_zlozona))
        )
        df_lrt = self.df.dropna(subset=zmienne)

        model1 = smf.logit(form_prostsza, data=df_lrt).fit(disp=False)
        model2 = smf.logit(form_zlozona, data=df_lrt).fit(disp=False)
        
        L1 = model1.llf
        L2 = model2.llf
        
        LRT_stat = 2 * (L2 - L1)
        df_diff = model2.df_model - model1.df_model
        p_value = chi2.sf(LRT_stat, df_diff)
        
        logger.info("Log-Likelihood (Model 1): %.4f", L1)
        logger.info("Log-Likelihood (Model 2): %.4f", L2)
        logger.info("Statystyka LRT: %.4f", LRT_stat)
        logger.info("Stopnie swobody (df): %s", df_diff)
        logger.info("p-value: %.6f", p_value)
        
        if p_value < 0.05:
            logger.info("Wniosek: Odrzucamy H0. Bardziej złożony model wnosi istotną poprawę.")
        else:
            logger.info(
                "Wniosek: Brak podstaw do odrzucenia H0. Dodatkowe zmienne nie poprawiają modelu w sposób istotny."
            )

        return {
            "formula_simple": form_prostsza,
            "formula_complex": form_zlozona,
            "n_obs": int(len(df_lrt)),
            "ll_simple": float(L1),
            "ll_complex": float(L2),
            "lrt_stat": float(LRT_stat),
            "df_diff": float(df_diff),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    def zadanie_4e_f_ewaluacja(self, formula, cut_off=0.5, test_size=0.3):
        """
        Zadanie 4e i 4f: Podział na zbiór treningowy/testowy i ewaluacja
        wielowymiarowego modelu za pomocą macierzy pomyłek.
        """
        logger.info("%s", "=" * 50)
        logger.info("ZADANIE 4e i 4f: Podział Train/Test i Ewaluacja")
        logger.info("Model: %s | Cut-off: %s", formula, cut_off)
        logger.info("%s", "=" * 50)
        
        train_data, test_data, _model, y_true, y_pred_prob = self._trenuj_i_predykcja(
            formula=formula,
            test_size=test_size,
        )
        y_pred_bin = (y_pred_prob >= cut_off).astype(int)
        
        # Obliczanie metryk (4e)
        cm = confusion_matrix(y_true, y_pred_bin)
        acc = accuracy_score(y_true, y_pred_bin)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        
        logger.info("Rozmiar zbioru treningowego: %d", len(train_data))
        logger.info("Rozmiar zbioru testowego: %d", len(test_data))
        logger.info("Macierz pomyłek (Confusion Matrix):")
        for row in cm.tolist():
            logger.info("%s", row)
        logger.info("Accuracy (Dokładność): %.4f", acc)
        logger.info("Precision (Precyzja):  %.4f", prec)
        logger.info("Recall (Czułość):      %.4f", rec)
        logger.info("F1-score:              %.4f", f1)

        return {
            "formula": formula,
            "cut_off": float(cut_off),
            "test_size": float(test_size),
            "n_train": int(len(train_data)),
            "n_test": int(len(test_data)),
            "confusion_matrix": cm.tolist(),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    def wyznacz_optymalny_cutoff(self, formula, test_size=0.3):
        """
        Zadanie 3d/4: Wyznaczenie optymalnego punktu odcięcia (cut-off) 
        maksymalizującego Balanced Accuracy dla niezbalansowanego zbioru danych.
        """
        logger.info("%s", "=" * 50)
        logger.info("Szukanie optymalnego cut-off dla modelu: %s", formula)
        logger.info("%s", "=" * 50)

        _train_data, _test_data, _model, y_true, y_pred_prob = self._trenuj_i_predykcja(
            formula=formula,
            test_size=test_size,
        )
        
        best_cutoff = 0.5
        best_score = 0.0
        
        # Testujemy cut-offy od 0.1 do 0.9 z krokiem 0.05
        thresholds = np.arange(0.1, 0.95, 0.05)
        for t in thresholds:
            y_pred_bin = (y_pred_prob >= t).astype(int)

            score = balanced_accuracy_score(y_true, y_pred_bin)

            if score > best_score:
                best_score = score
                best_cutoff = t

        best_cutoff = round(float(best_cutoff), 2)
        logger.info(
            "Znaleziono optymalny cut-off: %.2f (Balanced Accuracy: %.4f)",
            best_cutoff,
            best_score,
        )

        return {
            "formula": formula,
            "test_size": float(test_size),
            "best_cutoff": best_cutoff,
            "balanced_accuracy": float(best_score),
            "thresholds_tested": [round(float(t), 2) for t in thresholds.tolist()],
        }


    def rysuj_rozklad_chi2(self, lrt_stat, df, output_path=None):
        """Generuje wykres gęstości rozkładu chi-kwadrat z zaznaczoną statystyką LRT."""
        logger.info("%s", "=" * 50)
        logger.info("Generowanie wykresu rozkładu chi-kwadrat...")
        
        # Oś X od 0 do wartości nieco większej niż nasza statystyka LRT lub minimum 10
        x_max = max(10.0, float(lrt_stat) + 3.0)
        x = np.linspace(0, x_max, 500)
        y = chi2.pdf(x, df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label=f'Gęstość $\chi^2$ (df={int(df)})', color='tab:blue')
        
        # Pionowa linia przerywana dla statystyki LRT
        ax.axvline(x=lrt_stat, color='tab:blue', linestyle='--', label=f'LRT = {lrt_stat:.2f}')
        ax.plot(lrt_stat, 0, marker='o', color='tab:blue') # Kropka na osi X

        ax.set_title("Zadanie 4: Rozkład chi-kwadrat i statystyka LRT")
        ax.set_xlabel("Wartość statystyki (x)")
        ax.set_ylabel("Gęstość prawdopodobieństwa f(x)")
        ax.legend()
        
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            logger.info("Zapisano wykres LRT: %s", output)

        plt.close(fig)

    def rysuj_macierz_pomylek(self, cm_list, output_path=None):
        """Generuje graficzną reprezentację macierzy pomyłek (Heatmap)."""
        logger.info("%s", "=" * 50)
        logger.info("Generowanie graficznej macierzy pomyłek...")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_list, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False, 
            ax=ax,
            xticklabels=['Przewidziane: 0\n(Zgon)', 'Przewidziane: 1\n(Przeżycie)'],
            yticklabels=['Rzeczywiste: 0\n(Zgon)', 'Rzeczywiste: 1\n(Przeżycie)']
        )
        
        ax.set_title("Zadanie 4: Macierz Pomyłek modelu wielowymiarowego")
        
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            logger.info("Zapisano wykres macierzy pomyłek: %s", output)

        plt.close(fig)