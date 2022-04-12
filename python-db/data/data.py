""" We create a class representing real-estate from a data-tuple from database """

from dataclasses import dataclass
from datetime import date
from itertools import takewhile
from typing import List, Dict

from bson.objectid import ObjectId
from data.enums import Objektunterart, Objektausstattung, Objektzustand, Lagequalitaet, \
    Bauweise, Hochwassergefahrenzone, Vermietbarkeit, Verwertbarkeit, Dachausbau, Verwendung
from data.wrappers import Baujahr, Wohnflaeche, Grundstuecksgroesse, \
    Marktwert, Kaufpreis, AnzahlZimmer, Quadratmeterpreis, Wertermittlungsstichtag

attribute_name_dict = {
    'anzahl_carport': 'kurzgutachten.objektangabenAnzahlCarport',
    'anzahl_garagen': 'kurzgutachten.objektangabenAnzahlGaragen',
    'anzahl_stellplaetze_aussen': 'kurzgutachten.objektangabenAnzahlStellplaetzeAussen',
    'anzahl_stellplaetze_innen': 'kurzgutachten.objektangabenAnzahlStellplaetzeInnen',
    'anzahl_wohneinheiten': 'kurzgutachten.objektangabenAnzahlWohneinheiten',
    'anzahl_zimmer': 'kurzgutachten.objektangabenAnzahlZimmer',
    'ausgebauter_spitzboden': 'kurzgutachten.objektangabenAusgebauterSpitzboden',
    'ausstattung': 'kurzgutachten.objektangabenAusstattung',
    'ausstattungGuessed': 'guessedAusstattung',
    'balcony_area': 'balcony_area',
    'baujahr': 'kurzgutachten.objektangabenBaujahr',
    'bauweise': 'kurzgutachten.bauweise',
    'beleihungswert': 'beleihungswert',
    'bodenwert': 'kurzgutachten.bodenwertBodenWert',
    'bundesland': 'bundesland',
    'centrality': 'Acxiom.centrality',
    'dachausbau': 'kurzgutachten.objektangabenDachausbau',
    'dully_2020_split': 'predictions.dully-split2020-germany',
    'dully_japan': 'predictions.dully-split201703-cleanv3',
    'ea_japan': 'predictions.ea-split201703-cleanv3',
    'dully_japan_unclean': 'predictions.dully-split201703-uncleanv3',
    'dully_discrete_split': 'predictions.dully-discrete-split-germany',
    'einliegerwohnung': 'kurzgutachten.objektangabenEinliegerwohnung',
    'ertragswert': 'ertragswertGerundetMwt',
    'ertragswert_wohnflaeche': 'kurzgutachten.ertragswertWohnflaeche',
    'flaecheneinheitspreis': 'flaecheneinheitspreisKaufpreis',
    'grundstuecksgroesse': 'grundstuecksgroesseInQuadratmetern',
    'hausnummer': 'hausnummer',
    'hochwassergefahrenzone': 'kurzgutachten.hochwasserGefahrenzone',
    'house_kaisuu': 'house_kaisuu',
    'id': '_id',
    'kaufpreis': 'kaufpreis',
    'kreis': 'kreis_canonic',
    'lagequalitaet': 'kurzgutachten.objektangabenLagequalitaet',
    'land_toshi': 'land_toshi',
    'lat': 'location',
    'lng': 'location',
    'location': 'location',
    'lor': 'berlin_plr_id',
    'marktueblicher_kaufpreis': 'marktueblicherKaufpreis',
    'marktwert': 'marktwert',
    'objektunterart': 'objektunterart',
    'ort': 'ort',
    'ortsteil': 'ortsteil',
    'plane_x': 'plane_location',
    'plane_y': 'plane_location',
    'plz': 'plz',
    'pois': 'pois_micro_scores',
    'prefecture': 'prefecture',
    'regiotyp': 'Acxiom.regioTyp',
    'restnutzungsdauer': 'restnutzungsdauer',
    'school_el_distance': 'school_ele_distance',
    'school_jun_distance': 'school_jun_distance',
    'scores_all': 'scores.ALL',
    'vermietbarkeit': 'kurzgutachten.vermietbarkeit',
    'verwertbarkeit': 'kurzgutachten.verwertbarkeit',
    'verwertbarkeitGuessed': 'guessed_verwertbarkeit',
    'walk_distance1': 'walk_distance1',
    'wertermittlungsstichtag': 'wertermittlungsstichtag',
    'wohnflaeche': 'kurzgutachten.objektangabenWohnflaeche',
    'zustand': 'kurzgutachten.objektangabenZustand',
}


def attribute_name_to_database_column(attribute_name: str) -> str:
    """ maps attribute name to database column """
    return attribute_name_dict.get(attribute_name, attribute_name)


@dataclass
class Immobilie:
    """ actually this represents real-estate """
    # pylint: disable=too-many-instance-attributes

    # pylint: disable=invalid-name
    id: ObjectId = None

    # value
    marktwert: float = None
    quadratmeterpreis: float = None
    quadratmeterkaufpreis: float = None
    beleihungswert: float = None
    bodenwert: List[float] = None  # in euro
    ertragswert: float = None
    kaufpreis: float = None  # by kurzgutachten
    marktueblicher_kaufpreis: bool = None
    verwertbarkeit: Verwertbarkeit = None  # kurzgutachten.verwertbarkeit
    vermietbarkeit: Vermietbarkeit = None  # kurzgutachten.vermietbarkeit
    verwendung: Verwendung = None

    # location
    bundesland: str = None
    kreis: str = None
    ort: str = None
    ortsteil: str = None
    plz: str = None
    strasse: str = None
    hausnummer: str = None
    hausnummer_int: int = None
    lng: float = None
    lat: float = None
    lor: str = None
    plane_x: float = None
    plane_y: float = None
    regiotyp: int = None
    centrality: int = None
    scores_all: int = None

    # attributes
    objektunterart: Objektunterart = None
    baujahr: int = None
    ertragswert_wohnflaeche: float = None
    grundstuecksgroesse: float = None  # in qm
    # Onlinedaten, Sachwert, Vergleichswert, ergebnis, ertragswert
    hochwassergefahrenzone: Hochwassergefahrenzone = None  # kurzgutachten...
    bauweise: Bauweise = None  # kurzgutachten.bauweise
    flaecheneinheitspreis: float = None
    wertermittlungsstichtag: date = None
    restnutzungsdauer: float = None

    # japan
    balcony_area: float = None
    walk_distance1: float = None
    land_toshi: float = None
    house_kaisuu: int = None
    school_el_distance: float = None
    school_jun_distance: float = None
    prefecture: str = None

    # kurzgutachten.objektangaben...
    lagequalitaet: Lagequalitaet = None  # kurzgutachten.objektangabenLagequalitaet
    ausstattung: Objektausstattung = None  # kurzgutachten.objektangabenAusstattung
    ausstattungGuessed: Objektausstattung = None
    ausstattungGuessedOrActual: Objektausstattung = None
    anzahl_carport: int = None
    anzahl_garagen: int = None
    anzahl_stellplaetze_aussen: int = None
    anzahl_stellplaetze_innen: int = None
    anzahl_wohneinheiten: int = None
    anzahl_zimmer: int = None
    ausgebauter_spitzboden: bool = None
    dachausbau: Dachausbau = None
    einliegerwohnung: bool = None
    unterkellerungsgrad: int = None  # in prozent, siehe objektangabenKeller
    stockwerk: int = None  # kurzgutachten.objektangabenLageImGebaeude
    wohnflaeche: float = None
    zustand: Objektzustand = None  # kurzgutachten.objektangabenZustand

    # values written by us
    poi_arrays: Dict = None
    dully_2020_split = None
    dully_discrete_split = None
    ea_japan = None
    dully_japan = None
    dully_japan_unclean = None

    # set this for ensemble prediction
    cbr_prediction = None

    # will be used in every algorithm, defaults to U_Germany
    # if you want to use U_berlin or U_brandenburg, overwrite U
    U: float = None
    U_berlin: float = None
    U_brandenburg: float = None

    def __init__(self, data: dict, cbr_column):
        # pylint: disable=too-many-statements
        # This method reads a lot of data from the db.
        # There is no way to avoid having many statements
        kurzgutachten = data.get('kurzgutachten', {})
        self.id = data.get('_id')
        # money
        self.marktwert = Marktwert(data.get('marktwert'))
        self.beleihungswert = data.get('beleihungswert')
        self.ertragswert = data.get('ertragswertGerundetMwt')
        self.kaufpreis = Kaufpreis(data.get('kaufpreis'))
        self.quadratmeterpreis = Quadratmeterpreis(self.marktwert,
                                                   kurzgutachten.get('objektangabenWohnflaeche'))
        self.quadratmeterkaufpreis = Quadratmeterpreis(data.get('kaufpreis'), kurzgutachten.get(
            'objektangabenWohnflaeche'))
        self.verwertbarkeit = Verwertbarkeit.from_str(
            kurzgutachten.get('verwertbarkeit'))
        self.verwertbarkeitGuessed = Objektausstattung.from_str(
            data.get('guessed_verwertbarkeit'))
        self.verwertbarkeitGuessedOrActual = self.verwertbarkeit or self.verwertbarkeitGuessed
        self.vermietbarkeit = Vermietbarkeit.from_str(
            kurzgutachten.get('vermietbarkeit'))
        self.verwendung = Verwendung.from_str(
            kurzgutachten.get('objektangabenVerwendung'))
        self.ertragswert_wohnflaeche = kurzgutachten.get(
            'ertragswertWohnflaeche')

        # location
        self.bundesland = data.get('bundesland')
        self.kreis = data.get('kreis_canonic')
        self.ort = data.get('ort')
        self.ortsteil = data.get('ortsteil')
        if self.plz is None or not self.plz.isdigit():
            self.plz = data.get('plz')
        self.strasse = data.get('strasse')
        self.hausnummer = data.get('hausnummer')
        self.hausnummer_int = int(
            '0' + ''.join(takewhile(lambda x: x.isnumeric(), self.hausnummer))) \
            if self.hausnummer is not None else None
        if data.get('location') is not None:
            self.lng = data.get('location')[0]
            self.lat = data.get('location')[1]
        self.lor = data.get('berlin_plr_id')
        if data.get('plane_location') is not None:
            self.plane_x = data.get('plane_location')[0]
            self.plane_y = data.get('plane_location')[1]
        if data.get('Acxiom') is not None:
            self.regiotyp = data.get('Acxiom').get('regioTyp')
            self.centrality = data.get('Acxiom').get('centrality')
        if data.get('scores') is not None:
            self.scores_all = data.get('scores').get('ALL')

        # attributes
        self.objektunterart = Objektunterart.from_str(
            data.get('objektunterart'))
        self.bauweise = Bauweise.from_str(kurzgutachten.get('bauweise'))
        self.baujahr = Baujahr(kurzgutachten.get('objektangabenBaujahr'))
        self.wohnflaeche = Wohnflaeche(
            kurzgutachten.get('objektangabenWohnflaeche'))
        self.grundstuecksgroesse = Grundstuecksgroesse(
            data.get('grundstuecksgroesseInQuadratmetern'))
        self.flaecheneinheitspreis = data.get('flaecheneinheitspreisKaufpreis')
        if data.get('wertermittlungsstichtag') is not None:
            self.wertermittlungsstichtag = Wertermittlungsstichtag(data.get(
                'wertermittlungsstichtag').date())
        self.restnutzungsdauer = data.get('restnutzungsdauer')

        # japan
        self.balcony_area = float(data.get('balcony_area') or 0)
        self.walk_distance1 = float(data.get('walk_distance1') or 985)  # avg dist
        self.land_toshi = float(data.get('land_toshi') or 0)
        self.house_kaisuu = int(data.get('house_kaisuu') or 0)
        self.school_el_distance = float(data.get('school_ele_distance') or 777)  # avg dist
        self.school_jun_distance = float(data.get('school_jun_distance') or 1133)  # avg dist
        self.prefecture = data.get('prefecture')

        # objektangaben
        self.lagequalitaet = Lagequalitaet.from_str(
            kurzgutachten.get('objektangabenLagequalitaet'))
        self.ausstattung = Objektausstattung.from_str(
            kurzgutachten.get('objektangabenAusstattung'))
        self.ausstattungGuessed = Objektausstattung.from_str(
            data.get('guessed_ausstattung'))
        self.ausstattungGuessedOrActual = self.ausstattung or self.ausstattungGuessed
        self.anzahl_carport = kurzgutachten.get(
            'objektangabenAnzahlCarport', 0) or 0
        self.anzahl_garagen = kurzgutachten.get(
            'objektangabenAnzahlGaragen', 0) or 0
        self.anzahl_stellplaetze_aussen = kurzgutachten.get(
            'objektangabenAnzahlStellplaetzeAussen', 0) or 0
        self.anzahl_stellplaetze_innen = kurzgutachten.get(
            'objektangabenAnzahlStellplaetzeInnen', 0) or 0
        self.anzahl_wohneinheiten = kurzgutachten.get(
            'objektangabenAnzahlWohneinheiten', 1)
        self.anzahl_zimmer = AnzahlZimmer(
            kurzgutachten.get('objektangabenAnzahlZimmer'))
        self.ausgebauter_spitzboden = kurzgutachten.get(
            'objektangabenAusgebauterSpitzboden', False)
        self.dachausbau = Dachausbau.from_str(
            kurzgutachten.get('objektangabenDachausbau'))
        self.einliegerwohnung = kurzgutachten.get(
            'objektangabenEinliegerwohnung') == 'Ja'
        self.zustand = Objektzustand.from_str(
            kurzgutachten.get('objektangabenZustand'))
        self.hochwassergefahrenzone = Hochwassergefahrenzone.from_str(
            kurzgutachten.get('hochwasserGefahrenzone'))

        # self.unterkellerungsgrad = interpret boolean from \
        # objektangabenKeller and string from objektangabenUnterkellerungsgrad
        if kurzgutachten.get('objektangabenKeller') == 'nicht unterkellert':
            self.unterkellerungsgrad = 0
        elif kurzgutachten.get('objektangabenUnterkellerungsgrad') is not None and len(
                kurzgutachten.get('objektangabenUnterkellerungsgrad')) > 1:
            self.unterkellerungsgrad = int(kurzgutachten.get(
                'objektangabenUnterkellerungsgrad')[:-1])

        # self.stockwerk =
        self.bodenwert = kurzgutachten.get('bodenwertBodenWert')

        # pois
        self.poi_arrays = {}
        pois_padding_value = 10000
        for key, value in data.get('pois_micro_scores', {}).items():
            if key != "PlaceCityDriving-car":
                self.poi_arrays[key] = value + [pois_padding_value] * (10 - len(value))

        self.U = data.get('U_Germany', None)
        self.U_berlin = data.get('U_berlin', None)
        self.U_brandenburg = data.get('U_brandenburg', None)

        if data.get('predictions') is not None:
            self.dully_2020_split = data.get('predictions').get('dully-split2020-germany')
            self.dully_discrete_split = data.get('predictions').get('dully-discrete-split-germany')
            self.ea_japan = data.get('predictions').get('ea-split201703-cleanv3')
            self.dully_japan = data.get('predictions').get('dully-split201703-cleanv3')
            self.dully_japan_unclean = data.get('predictions').get('dully-split201703-uncleanv3')

        if cbr_column is not None and data.get('predictions') is not None:
            self.cbr_prediction = data.get('predictions').get(cbr_column)
