import logging
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

class Zadanie3:
    def __init__(self, df):
        self.df = df

    # Wykres regresji logistycznej
    def krzywa_logistyczna(self, output_path=None):
        """Rysuje krzywe logistyczne P90_bin ~ zmienne niezależne ciągłe."""
        wymagane = ["WIEK", "CHOLESTEROL", "DBP", "SBP", "P90_bin"]
        dane = self.df.dropna(subset=wymagane)

        # Wybór predyktorów ciągłych - z binarnych nie da się sporządzić wykresu regresji logistycznej
        variables = ["WIEK", "CHOLESTEROL", "DBP", "SBP"]
        for variable in variables:
            data = dane.dropna(subset=[variable])
            model = smf.logit(formula="P90_bin ~ " + variable, data=data)
            results = model.fit()
            logger.info(results.summary())

            # Wykres regresji
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(
                x=variable,
                y="P90_bin",
                data=dane,
                logistic=True,
                ci=99,
                scatter_kws={"alpha": 0.5, "s": 30},
                line_kws={"color": "red", "linewidth": 2},
                ax=ax,
            )
            title = "Krzywa logistyczna P90_bin ~ " + variable
            ax.set_title(title)
            ax.set_xlabel(variable)
            ax.set_ylabel("P90_bin")
            ax.set_yticks([0, 1])

            if output_path is not None:
                filename = "zad03_krzywa_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)



    # Krzywa ROC
    def roc(self, output_path=None):
        data = self.df
        y = data['P90_bin']

        variables = ["WIEK", "CHOLESTEROL", "DBP", "SBP"]
        for variable in variables:
            # predyktory i odpowiedź
            X = data[[variable]]

            # rozdział na dane uczące (70%) i testowe
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

            # model regresji logistycznej
            log_regression = LogisticRegression()

            # fit the model using the training data
            log_regression.fit(X_train,y_train)

            y_pred_proba = log_regression.predict_proba(X_test)[::,1]
            fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            # krzywa ROC
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.plot(fpr,tpr,label="AUC = "+str(auc))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
            title = "Krzywa ROC P90_bin ~ " + variable
            ax.set_title(title)

            if output_path is not None:
                filename = "zad03_ROC_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)

    def odds_ratios(self, output_path=None):
        data = self.df
        y = data['P90_bin']

        variables = ["WIEK", "CHOLESTEROL", "DBP", "SBP"]
        for variable in variables:
            # predyktory i odpowiedź
            # ~ - odwrócić dla łatwiejszej interpretacji
            X = ~data[[variable]]
            clf = LogisticRegression()
            clf.fit(X,y)
            ratio = np.exp(clf.coef_)
            logger.info("%s: OR = %f", variable, ratio[0][0])

    def smokers(self, output_path=None):
        required = ["WIEK", "CHOLESTEROL", "DBP", "SBP", "P90_bin"]
        data = self.df.dropna(subset=required)
        smokers = data[data['PAPIEROSY_bin'] == 1]
        nonsmokers = data[data['PAPIEROSY_bin'] == 0]

        logger.info("====================================")
        logger.info("KRZYWA LOGISTYCZNA")
        logger.info("====================================")

        # Wybór predyktorów ciągłych - z binarnych nie da się sporządzić wykresu regresji logistycznej
        variables = ["WIEK", "CHOLESTEROL", "DBP", "SBP"]

        logger.info("Palacze")
        logger.info("====================================")
        for variable in variables:
            _data = smokers.dropna(subset=[variable])
            model = smf.logit(formula="P90_bin ~ " + variable, data=_data)
            results = model.fit()
            logger.info(results.summary())

            # Wykres regresji
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(x=variable, y="P90_bin", data=_data, logistic=True, ci=99, scatter_kws={"alpha": 0.5, "s": 30}, line_kws={"color": "red", "linewidth": 2}, ax=ax)
            title = "Krzywa logistyczna P90_bin ~ " + variable + " dla palaczy"
            ax.set_title(title)
            ax.set_xlabel(variable)
            ax.set_ylabel("P90_bin")
            ax.set_yticks([0, 1])

            if output_path is not None:
                filename = "pkt_b/zad03_krzywa_PALACZE_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)

        logger.info("Niepalacze")
        logger.info("====================================")
        for variable in variables:
            _data = nonsmokers.dropna(subset=[variable])
            model = smf.logit(formula="P90_bin ~ " + variable, data=_data)
            results = model.fit()
            logger.info(results.summary())

            # Wykres regresji
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(x=variable, y="P90_bin", data=_data, logistic=True, ci=99, scatter_kws={"alpha": 0.5, "s": 30}, line_kws={"color": "red", "linewidth": 2}, ax=ax)
            title = "Krzywa logistyczna P90_bin ~ " + variable + " dla niepalaczy"
            ax.set_title(title)
            ax.set_xlabel(variable)
            ax.set_ylabel("P90_bin")
            ax.set_yticks([0, 1])

            if output_path is not None:
                filename = "pkt_b/zad03_krzywa_NIEPALACZE_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)



        logger.info("====================================")
        logger.info("KRZYWA ROC I ILORAZY SZANS")
        logger.info("====================================")

        logger.info("Palacze")
        logger.info("====================================")
        for variable in variables:
            _data = smokers.dropna(subset=[variable])
            # predyktory i odpowiedź
            y = _data['P90_bin']
            X = _data[[variable]]

            # rozdział na dane uczące (70%) i testowe
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

            # model regresji logistycznej
            log_regression = LogisticRegression()

            # fit the model using the training data
            log_regression.fit(X_train,y_train)

            y_pred_proba = log_regression.predict_proba(X_test)[::,1]
            fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            # krzywa ROC
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.plot(fpr,tpr,label="AUC = "+str(auc))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
            title = "Krzywa ROC P90_bin ~ " + variable + " dla palaczy"
            ax.set_title(title)

            if output_path is not None:
                filename = "pkt_b/zad03_ROC_PALACZE_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)

            X = ~_data[[variable]]
            clf = LogisticRegression()
            clf.fit(X,y)
            ratio = np.exp(clf.coef_)
            logger.info("%s (palacze): OR = %f", variable, ratio[0][0])

        logger.info("Niepalacze")
        logger.info("====================================")
        for variable in variables:
            _data = nonsmokers.dropna(subset=[variable])
            # predyktory i odpowiedź
            y = _data['P90_bin']
            X = _data[[variable]]

            # rozdział na dane uczące (70%) i testowe
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

            # model regresji logistycznej
            log_regression = LogisticRegression()

            # fit the model using the training data
            log_regression.fit(X_train,y_train)

            y_pred_proba = log_regression.predict_proba(X_test)[::,1]
            fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            # krzywa ROC
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.plot(fpr,tpr,label="AUC = "+str(auc))
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
            title = "Krzywa ROC P90_bin ~ " + variable + " dla niepalaczy"
            ax.set_title(title)

            if output_path is not None:
                filename = "pkt_b/zad03_ROC_NIEPALACZE_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)

            X = ~_data[[variable]]
            clf = LogisticRegression()
            clf.fit(X,y)
            ratio = np.exp(clf.coef_)
            logger.info("%s (niepalacze): OR = %f", variable, ratio[0][0])

    def confusion(self, output_path=None):
        data = self.df
        y = data['P90_bin']

        variables = ["WIEK", "CHOLESTEROL", "DBP", "SBP"]
        for variable in variables:
            # predyktory i odpowiedź
            X = data[[variable]]

            # rozdział na dane uczące (50%) i testowe
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0) 

            # model regresji logistycznej
            lr = LogisticRegression()
            # fit the model using the training data
            lr.fit(X_train,y_train)

            # the probability of being P90_bin = 1 => patient alive
            y_prob = lr.predict_proba(X_test)[:,1]

            # using ROC for optimum cutoff value
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            # Method 1: Youden's Index (Maximizes TPR - FPR)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_cutoff = thresholds[optimal_idx]
            logger.info(optimal_cutoff)

            actual = y_test.values.tolist()
            predicted = [1 if i > optimal_cutoff else 0 for i in y_prob]

            matrix = confusion_matrix(actual, predicted)
            cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [0, 1])

            logger.info("====== " + variable + " ======")
            logger.info("Accuracy = " + str(accuracy_score(actual, predicted)))
            logger.info("Precision = " + str(precision_score(actual, predicted)))
            logger.info("Recall = " + str(recall_score(actual, predicted)))
            logger.info("F1_score = " + str(f1_score(actual, predicted)))

            # krzywa ROC
            cm_display.plot()
            title = "Macierz pomyłek P90_bin ~ " + variable
            plt.title(title)

            if output_path is not None:
                filename = "pkt_d/zad03_matrix_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close()

    def graph(self, output_path=None):
        required = ["WIEK", "CHOLESTEROL", "DBP", "SBP", "P90_bin"]
        data = self.df.dropna(subset=required)

        variables = ["PAPIEROSY_bin", "NADCISNIENIE_bin", "MIGRENA_bin", "CUKRZYCA_bin"]
        for variable in variables:
            group1 = data[data[variable] == 1]
            group2 = data[data[variable] == 0]

            _data = group1.dropna(subset=[variable])
            y = _data['P90_bin']
            X = _data['WIEK']

            # Wykres regresji - grupa #1
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(
                x=X,
                y=y,
                data=_data,
                ci=None,
                logistic=True,
                scatter_kws={"alpha": 0.5, "s": 30},
                line_kws={"color": "red", "linewidth": 2},
                ax=ax,
            )

            _data = group2.dropna(subset=[variable])
            y = _data['P90_bin']
            X = _data['WIEK']

            # Wykres regresji - grupa #2
            sns.regplot(
                x=X,
                y=y,
                data=_data,
                ci=None,
                logistic=True,
                scatter_kws={"alpha": 0.5, "s": 30},
                line_kws={"color": "blue", "linewidth": 2},
                ax=ax,
            )
            title = "Krzywa logistyczna P90_bin ~ WIEK: dychotomia zmiennej " + variable
            ax.set_title(title)
            ax.set_xlabel("WIEK")
            ax.set_ylabel("P(przeżycie)")
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            red_patch = mpatches.Patch(color='red', label='Tak')
            blue_patch = mpatches.Patch(color='blue', label='Nie')
            plt.legend(handles=[red_patch, blue_patch])

            if output_path is not None:
                filename = "pkt_c/zad03_dychotomia_" + variable + ".png"
                output = Path(output_path) / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=150, bbox_inches="tight")
                logger.info("Zapisano wykres: %s", output)

            plt.close(fig)
