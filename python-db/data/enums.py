"""
enums for string variables
starting from 1, higher is better if there is such a natural order
"""
from enum import Enum


class Objektunterart(Enum):
    """ What is the objects subspecies """
    einfamilienhaus_einliegerwohnung = 1
    einfamilienhaus = 2
    zweifamilienhaus = 3
    doppelhaushaelfte = 4
    reihenhaus = 5
    reihenmittelhaus = 6
    reihenendhaus = 7
    eigentumswohnung = 8

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        # pylint: disable=too-many-return-statements
        if label is None:
            return None
        if label == 'Eigentumswohnung':
            return Objektunterart.eigentumswohnung
        if label == 'Einfamilienhaus':
            return Objektunterart.einfamilienhaus
        if label == 'Doppelhaushälfte':
            return Objektunterart.doppelhaushaelfte
        if label == 'Zweifamilienhaus':
            return Objektunterart.zweifamilienhaus
        if label == 'Reihenmittelhaus':
            return Objektunterart.reihenmittelhaus
        if label == 'Reihenendhaus':
            return Objektunterart.reihenendhaus
        if label == 'Reihenhaus':
            return Objektunterart.reihenhaus
        if label == 'Einfamilienhaus mit Einliegerwohnung':
            return Objektunterart.einfamilienhaus_einliegerwohnung
        return None

    def __str__(self):
        return str(self.name)


class Objektausstattung(Enum):
    """ How is the quality of the furnishing """
    sehr_einfach = 1
    einfach = 2
    mittel = 3
    gehoben = 4
    sehr_gehoben = 5

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        # pylint: disable=too-many-return-statements
        if label is None:
            return None
        if '1' in label:
            return Objektausstattung.sehr_einfach
        if '2' in label:
            return Objektausstattung.einfach
        if '3' in label:
            return Objektausstattung.mittel
        if '4' in label:
            return Objektausstattung.gehoben
        if '5' in label:
            return Objektausstattung.sehr_gehoben
        return None


class Objektzustand(Enum):
    """ How is the objects condition """
    katastrophal = 1
    schlecht = 2
    maessig = 3
    mittel = 4
    gut = 5
    sehr_gut = 6

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        # pylint: disable=too-many-return-statements
        if label == 'katastrophal':
            return Objektzustand.katastrophal
        if label == 'schlecht':
            return Objektzustand.schlecht
        if label == 'mäßig':
            return Objektzustand.maessig
        if label == 'mittel':
            return Objektzustand.mittel
        if label == 'gut':
            return Objektzustand.gut
        if label == 'sehr gut':
            return Objektzustand.sehr_gut
        return None


class Lagequalitaet(Enum):
    """ How is the quality of the location """
    katastrophal = 1
    sehr_schlecht = 2
    schlecht = 3
    maessig = 4
    unterdurchschnittlich = 5
    durchschnittlich = 6
    ueberdurchschnittlich = 7
    gut = 8
    sehr_gut = 9
    exzellent = 10

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        # pylint: disable=too-many-return-statements
        if label is None:
            return None
        if label in ('beste', 'exzellent', '1 - exzellent'):
            return Lagequalitaet.exzellent
        if label in ('sehr gut', '2 - sehr gut'):
            return Lagequalitaet.sehr_gut
        if label in ('gut', '3 - gut', 'Gut'):
            return Lagequalitaet.gut
        if label in ('4 - überdurchschnittlich', 'überdurchschnittlich', 'gehoben'):
            return Lagequalitaet.ueberdurchschnittlich
        if label in ('mittel', 'befriedigend', 'durchschnittlich', '5 - durchschnittlich'):
            return Lagequalitaet.durchschnittlich
        if label in ('unterdurchschnittlich', '6 - unterdurchschnittlich'):
            return Lagequalitaet.unterdurchschnittlich
        if label in ('mäßig', 'ausreichend'):
            return Lagequalitaet.maessig
        if label in ('mangelhaft', 'schlecht', '8 - schlecht'):
            return Lagequalitaet.schlecht
        if label in ('sehr schlecht', '9 - sehr schlecht'):
            return Lagequalitaet.sehr_schlecht
        if label == '10 - katastrophal':
            return Lagequalitaet.katastrophal
        return None


class Bauweise(Enum):
    """ Kind of construction """
    massivbauweise = 1
    fertighaus = 2
    fachwerkhaus = 3
    holzhaus = 4

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        if label is None:
            return None
        if label == 'Massivbauweise':
            return Bauweise.massivbauweise
        if label == 'Fertighaus':
            return Bauweise.fertighaus
        if label == 'Fachwerkhaus':
            return Bauweise.fachwerkhaus
        if label == 'Holzhaus':
            return Bauweise.holzhaus
        return None


class Hochwassergefahrenzone(Enum):
    """ How big is the risk of flood? """
    hoch = 1
    mittel = 2
    gering = 3
    sehr_gering = 4
    keine = 5

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        if label is None:
            return Hochwassergefahrenzone.keine
        if 'GK 1' in label:
            return Hochwassergefahrenzone.sehr_gering
        if 'GK 2' in label:
            return Hochwassergefahrenzone.gering
        if 'GK 3' in label:
            return Hochwassergefahrenzone.mittel
        if 'GK 4' in label:
            return Hochwassergefahrenzone.hoch
        return Hochwassergefahrenzone.keine


class Vermietbarkeit(Enum):
    """ quality of the verwertbarkeit """
    eingeschraenkt = 1
    normal = 2
    gut = 3
    sehr_gut = 4

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        if label is None:
            return None
        if label in ('eingeschraenkt', 'eingeschränkt'):
            return Vermietbarkeit.eingeschraenkt
        if label == 'normal':
            return Vermietbarkeit.normal
        if label == 'gut':
            return Vermietbarkeit.gut
        if label == 'sehr gut':
            return Vermietbarkeit.sehr_gut
        return None


class Verwertbarkeit(Enum):
    """ quality of the verwertbarkeit """
    schwer = 1
    eingeschraenkt = 2
    normal = 3
    gut = 4
    sehr_gut = 5

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        # pylint: disable=too-many-return-statements
        if label is None:
            return None
        if label == 'schwer':
            return Verwertbarkeit.schwer
        if label in ('eingeschraenkt', 'eingeschränkt'):
            return Verwertbarkeit.eingeschraenkt
        if label == 'normal':
            return Verwertbarkeit.normal
        if label == 'gut':
            return Verwertbarkeit.gut
        if label == 'sehr gut':
            return Verwertbarkeit.sehr_gut
        return None


class Dachausbau(Enum):
    """ quality of the dachausbau """
    nicht_ausgebautes_dachgeschoss = 1
    ausgebautes_dachgeschoss = 2
    flachdach = 3

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        if label is None:
            return None
        if label == 'ausgebautes Dachgeschoss':
            return Dachausbau.ausgebautes_dachgeschoss
        if label == 'nicht ausgebautes Dachgeschoss':
            return Dachausbau.nicht_ausgebautes_dachgeschoss
        if label == 'Flachdach':
            return Dachausbau.flachdach
        return None


class Verwendung(Enum):
    """ How is the Immobilie currently used """
    eigennutzung = 1
    fremdnutzung = 2
    eigen_und_fremdnutzung = 3

    @staticmethod
    def from_str(label):
        """ find the value by loooking at the string from database """
        if label is None:
            return None
        if label == 'Eigennutzung':
            return Verwendung.eigennutzung
        if label == 'Fremdnutzung':
            return Verwendung.fremdnutzung
        if label == 'Eigen- und Fremdnutzung':
            return Verwendung.eigen_und_fremdnutzung
        return None
