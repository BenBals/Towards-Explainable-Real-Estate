""" custom types to remove nonsensical data on creation """
from datetime import date
from math import log


class Baujahr(int):
    """ Sets Baujahr to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (1800 <= value <= 2020 or japan):
            return int.__new__(cls, value)
        return None


# pylint: disable=invalid-name
# We want to use this function like the classes in this module
def Wertermittlungsstichtag(value, japan=True):
    """Sets Wertermittlungsstichtag to none if its value is unrealistically"""
    if value is not None and (date(2000, 1, 1) <= value <= date(2022, 1, 1) or japan):
        return value
    return None


class Wohnflaeche(float):
    """ Sets Wohnflaeche to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (20 < value < 2000 or japan):
            return float.__new__(cls, value)
        return None


class Quadratmeterpreis(float):
    """ Sets Quadratmeterpreis to none if its value is unrealistically """

    def __new__(cls, marktwert, wohnflaeche, japan=True):
        if marktwert is None or wohnflaeche is None or wohnflaeche <= 0:
            return None
        quadratmeterpreis = marktwert / wohnflaeche
        if japan or 100 < quadratmeterpreis < 20000:
            return float.__new__(cls, quadratmeterpreis)
        return None


class Grundstuecksgroesse(float):
    """ Sets Grundstuecksgroesse to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (0 <= value <= 20000 or japan):
            return float.__new__(cls, value)
        return None


class Marktwert(float):
    """ Sets Marktwert to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (20000 <= value <= 2000000 or japan):
            return float.__new__(cls, value)
        return None


class Kaufpreis(float):
    """ Sets Kaufpreis to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (20000 <= value <= 2000000 or japan):
            return float.__new__(cls, value)
        return None


class Ertragswert(float):
    """ NOT USED; would set Ertragswert to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (20000 <= value <= 2000000 or japan):
            return float.__new__(cls, value)
        return None


class AnzahlZimmer(float):
    """ Sets AnzahlZimmer to none if its value is unrealistically """

    def __new__(cls, value, japan=True):
        if value is not None and (1 <= value <= 10 or japan):
            return float.__new__(cls, value)
        return None
