from Zadanie03 import Zad03
from Zadanie06 import Zad06
from Zadanie07 import Zad07

def _section(title):
    print("#" * 60)
    print(title)
    print("#" * 60)

_section("Zadanie 3.")
zad03 = Zad03()
zad03.run()

_section("Zadanie 6.")
zad06 = Zad06()
zad06.run()

_section("Zadanie 7.")
zad07 = Zad07()
zad07.run()
