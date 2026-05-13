import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class Zadanie3:
    def __init__(self, df):
        self.df = df

    def zadanie_3_krzywa_logistyczna(self, output_path=None, show_plot=False):
        """Rysuje krzywa logistyczna P90_bin ~ WIEK."""
        wymagane = ["WIEK", "P90_bin"]
        brakujace = [kol for kol in wymagane if kol not in self.df.columns]
        if brakujace:
            raise ValueError(f"Brakuje wymaganych kolumn dla Zad03: {', '.join(brakujace)}")

        dane = self.df.dropna(subset=wymagane)
        if dane.empty:
            raise ValueError("Brak danych po usunieciu brakow dla kolumn WIEK i P90_bin")

        logger.info("Liczba obserwacji dla Zad03: %d", len(dane))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(
            x="WIEK",
            y="P90_bin",
            data=dane,
            logistic=True,
            ci=None,
            scatter_kws={"alpha": 0.5, "s": 30},
            line_kws={"color": "red", "linewidth": 2},
            ax=ax,
        )
        ax.set_title("Zadanie 3: Krzywa logistyczna P90_bin ~ WIEK")
        ax.set_xlabel("WIEK")
        ax.set_ylabel("P90_bin")

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
            logger.info("Zapisano wykres: %s", output)

        if show_plot:
            plt.show()

        plt.close(fig)
