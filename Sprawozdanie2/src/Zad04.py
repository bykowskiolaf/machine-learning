import logging

import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)

class Zadanie4:
    def __init__(self, df):
        self.df = df

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
        
        model1 = smf.logit(form_prostsza, data=self.df).fit(disp=False)
        model2 = smf.logit(form_zlozona, data=self.df).fit(disp=False)
        
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

    def zadanie_4e_f_ewaluacja(self, formula, cut_off=0.5, test_size=0.3):
        """
        Zadanie 4e i 4f: Podział na zbiór treningowy/testowy i ewaluacja
        wielowymiarowego modelu za pomocą macierzy pomyłek.
        """
        logger.info("%s", "=" * 50)
        logger.info("ZADANIE 4e i 4f: Podział Train/Test i Ewaluacja")
        logger.info("Model: %s | Cut-off: %s", formula, cut_off)
        logger.info("%s", "=" * 50)
        
        # Wyciągnięcie zmiennych z formuły, aby usunąć dla nich braki danych
        zmienne = formula.replace('~', '+').split('+')
        zmienne = [z.strip() for z in zmienne]
        df_clean = self.df.dropna(subset=zmienne)
        
        # Podział danych (4f)
        train_data, test_data = train_test_split(df_clean, test_size=test_size, random_state=42)
        
        # Trenowanie modelu
        model = smf.logit(formula, data=train_data).fit(disp=False)
        
        # Predykcja
        y_true = test_data['P90_bin']
        y_pred_prob = model.predict(test_data)
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
