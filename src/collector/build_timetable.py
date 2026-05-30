"""
build_timetable.py
------------------
Extrage orarul CTP din PDF-uri și îl salvează ca JSON structurat.
Procesează toate liniile din ctp_routes.json, detectează automat sensul
dus/întors și combină stațiile care se întind pe mai multe pagini.
Owner:
Input:  data/raw/ctp_routes.json + data/ctp_timetable/*.pdf
Output: data/processed/orar_ctp_all.json
Funcționalitate implementată:
- Parsare tabel PDF cu pdfplumber și extragere ore/minute per stație
- Detecție automată sens dus/întors prin pattern-ul A→B→A pe 3 pagini consecutive
- Combinare stații care se întind pe mai multe pagini (merge_schedules)
- Gestionare erori per linie fără a opri procesarea celorlalte
- Suport pentru linii cu un singur sens (SINGLE_DIRECTION)
- Normalizare nume stații (strip + uppercase)
- Deduplicare minute duplicate per oră

"""

from pathlib import Path
import json
import pdfplumber
# ============================================================
# 1. CONFIGURARE
# ============================================================
# Aici definim căile către foldere/fișiere.
# Nu mai folosim un singur PDF_PATH hardcodat, pentru că vrem să procesăm toate liniile.

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TIMETABLE_DIR = BASE_DIR / "data" / "ctp_timetable"
ROUTES_METADATA_PATH = BASE_DIR / "data" / "raw" / "ctp_routes.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "orar_ctp_all.json"

DAYS = ["Luni-Vineri", "Sambata", "Duminica"]


# ============================================================
# 2. FUNCȚII MICI / UTILITARE
# ============================================================

def load_routes_metadata():
    """
    Citește fișierul data/raw/ctp_routes.json.

    Acest fișier trebuie să conțină lista liniilor CTP:
    [
      {
        "route_id": "3",
        "vehicle_type": "tramvai",
        "route_name": "...",
        "pdf_file": "track-3.pdf"
      }
    ]

    json.load(file) transformă JSON-ul într-o listă Python.
    """

    with open(ROUTES_METADATA_PATH, "r", encoding="utf-8") as file:
        routes = json.load(file)

    return routes


def normalize_station_name(station_name):
    """
    Normalizează numele stației.

    Exemplu:
    ' Gara ' -> 'GARA'

    Facem asta ca să avem chei consistente în JSON.
    """

    return station_name.strip().upper()


def create_empty_station_schedule():
    """
    Creează structura goală pentru o stație.

    Fiecare stație are 3 categorii:
    - Luni-Vineri
    - Sambata
    - Duminica

    Fiecare categorie va conține orele și minutele.
    """

    return {
        "Luni-Vineri": {},
        "Sambata": {},
        "Duminica": {}
    }


def merge_minutes(existing_minutes, new_minutes):
    """
    Unește două liste de minute.

    Exemplu:
    existing_minutes = ["05", "13"]
    new_minutes = ["13", "21"]

    Rezultat:
    ["05", "13", "21"]

    Folosim set() ca să eliminăm duplicatele.
    Apoi sorted(..., key=lambda minute: int(minute)) ca să sortăm numeric.
    """

    all_minutes = existing_minutes + new_minutes

    return sorted(
        set(all_minutes),
        key=lambda minute: int(minute)
    )


# ============================================================
# 3. PARSARE CELULĂ DIN TABEL
# ============================================================

def parse_schedule_cell(cell):
    """
    Primește o celulă din tabelul PDF.

    Exemplu de celulă extrasă din PDF:
    '8 05 13 21 29 39 49 59 06 14 22\\n30 40 50'

    Primul număr este ora:
    8

    Restul numerelor sunt minutele:
    05, 13, 21, 29, ...

    Returnează:
    ("08", ["05", "06", "13", "14", "21", "22", ...])
    """

    if cell is None:
        return None

    # În PDF, unele celule au rânduri noi. Le transformăm în spații.
    cell = cell.replace("\n", " ")

    # split() împarte textul după spații.
    # "8 05 13" -> ["8", "05", "13"]
    parts = cell.split()

    # Avem nevoie de măcar ora + un minut.
    if len(parts) < 2:
        return None

    # Primul element este ora.
    # zfill(2): "8" -> "08"
    hour = parts[0].zfill(2)

    # Verificăm că ora e validă.
    if not hour.isdigit() or not 0 <= int(hour) <= 23:
        return None

    # Restul sunt minutele.
    minutes = parts[1:]

    valid_minutes = []

    for minute in minutes:
        # Păstrăm doar minute între 00 și 59.
        if minute.isdigit() and 0 <= int(minute) <= 59:
            valid_minutes.append(minute.zfill(2))

    if len(valid_minutes) == 0:
        return None

    # Eliminăm duplicatele și sortăm numeric.
    valid_minutes = sorted(set(valid_minutes), key=lambda x: int(x))

    return hour, valid_minutes


# ============================================================
# 4. EXTRAGERE DIN PDF
# ============================================================

def extract_station_name(tables):
    """
    Caută numele stației în tabelele extrase de pdfplumber.

    De obicei, apare într-o celulă de forma:
    'STATIA GARA'

    Returnează:
    'GARA'

    Dacă nu găsește stație, returnează None.
    """

    for table in tables:
        if not table:
            continue

        for row in table:
            if not row:
                continue

            for cell in row:
                if cell is None:
                    continue

                if "STATIA" in cell:
                    station_name = cell.replace("STATIA ", "").strip()
                    return normalize_station_name(station_name)

    return None


def find_schedule_table(tables):
    """
    Găsește tabelul care conține orarul.

    În multe PDF-uri:
    - tables[0] este tabelul cu numele stației
    - tables[1] este tabelul cu orarul

    Dar nu ne bazăm strict pe index, pentru că unele pagini pot fi diferite.

    Căutăm tabelul care conține:
    Luni-Vineri, Sambata, Duminica
    """

    for table in tables:
        if not table:
            continue

        for row in table:
            if not row:
                continue

            row_text = " ".join(
                [str(cell) for cell in row if cell is not None]
            )

            if (
                "Luni-Vineri" in row_text
                and "Sambata" in row_text
                and "Duminica" in row_text
            ):
                return table

    # Fallback: dacă nu găsim explicit header-ul, luăm ultimul tabel.
    # De obicei ultimul tabel este cel cu orarul.
    if len(tables) > 0:
        return tables[-1]

    return None


# ============================================================
# 5. PARSARE TABEL DE ORAR
# ============================================================

def parse_schedule_table(table, route_id):
    """
    Primește tabelul de orar al unei pagini și îl transformă în dicționar.

    Exemplu rezultat:
    {
      "Luni-Vineri": {
        "08": ["05", "13", "21"]
      },
      "Sambata": {
        "08": ["09", "20"]
      },
      "Duminica": {
        "08": ["09", "20"]
      }
    }

    route_id este necesar pentru a ignora rândul de tip:
    ["3", None, None]
    sau
    ["30b", None, None]
    """

    schedule = create_empty_station_schedule()

    for row in table:
        if row is None:
            continue

        if len(row) == 0:
            continue

        first_cell = row[0]

        if first_cell is None:
            continue

        first_cell = str(first_cell).strip()

        # Ignorăm rândul cu numărul liniei.
        # lower() ajută pentru linii de tip 23b / 30b.
        if first_cell.lower() == route_id.lower():
            continue

        # Ignorăm header-ul.
        if first_cell == "Luni-Vineri":
            continue

        # zip(DAYS, row) face perechi:
        # ("Luni-Vineri", prima celulă)
        # ("Sambata", a doua celulă)
        # ("Duminica", a treia celulă)
        for day, cell in zip(DAYS, row):
            parsed = parse_schedule_cell(cell)

            if parsed is None:
                continue

            hour, minutes = parsed

            if hour not in schedule[day]:
                schedule[day][hour] = []

            schedule[day][hour] = merge_minutes(
                schedule[day][hour],
                minutes
            )

    return schedule


def merge_schedules(existing_schedule, new_schedule):
    """
    Unește două orare pentru aceeași stație.

    Este necesar pentru situații ca:
    - pagina 1: GARA, orele 03-16
    - pagina 2: GARA, orele 17-22

    Vrem ca în JSON să avem o singură stație GARA,
    cu toate orele combinate.
    """

    for day in DAYS:
        for hour, minutes in new_schedule[day].items():
            if hour not in existing_schedule[day]:
                existing_schedule[day][hour] = []

            existing_schedule[day][hour] = merge_minutes(
                existing_schedule[day][hour],
                minutes
            )

    return existing_schedule


# ============================================================
# 6. COLECTARE STAȚII PE PAGINI
# ============================================================

def collect_page_stations(pdf_path):
    """
    Parcurge toate paginile unui PDF și detectează stația curentă.

    Unele stații continuă pe două pagini.
    De exemplu:
    pagina 1: STATIA GARA
    pagina 2: continuare pentru GARA, dar fără titlu nou

    De aceea folosim current_station:
    - dacă pagina are stație nouă, o actualizăm
    - dacă nu are, păstrăm stația anterioară

    Returnează:
    [
      {"page": 1, "station": "GARA"},
      {"page": 2, "station": "GARA"},
      {"page": 3, "station": "PIATA UNIRII"}
    ]
    """

    page_stations = []
    current_station = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            station_name = extract_station_name(tables)

            if station_name is not None:
                current_station = station_name

            page_stations.append({
                "page": page_index,
                "station": current_station
            })

    return page_stations


def inspect_page_stations(pdf_path):
    """
    Funcție de debugging.

    O poți folosi dacă vrei să vezi:
    Pagina 1: GARA
    Pagina 2: GARA
    etc.
    """

    page_stations = collect_page_stations(pdf_path)

    for item in page_stations:
        print(f"Pagina {item['page']}: {item['station']}")


# ============================================================
# 7. DETECTARE SENS DUS / ÎNTORS
# ============================================================

def detect_reverse_start_page(page_stations):
    """
    Detectează automat unde începe sensul întors.

    Regula folosită:
    ne uităm la câte 3 stații consecutive.

    Exemplu:
    Pagina 15: BLOCURI DANCU
    Pagina 16: DANCU
    Pagina 17: BLOCURI DANCU

    Avem:
    prima stație == a treia stație
    BLOCURI DANCU == BLOCURI DANCU

    iar stația din mijloc este diferită:
    DANCU

    Asta înseamnă că pagina 16 este capătul,
    iar pagina 17 este începutul sensului întors.

    Returnează:
    17
    """

    for index in range(len(page_stations) - 2):
        first = page_stations[index]
        middle = page_stations[index + 1]
        third = page_stations[index + 2]

        first_station = first["station"]
        middle_station = middle["station"]
        third_station = third["station"]

        if first_station is None or middle_station is None or third_station is None:
            continue

        if first_station == third_station and first_station != middle_station:
            reverse_start_page = third["page"]

            print("  Switch detectat automat:")
            print(f"  Pagina {first['page']}: {first_station}")
            print(f"  Pagina {middle['page']}: {middle_station}")
            print(f"  Pagina {third['page']}: {third_station}")
            print(f"  Sensul întors începe la pagina {reverse_start_page}")

            return reverse_start_page

    # Dacă nu găsim switch, presupunem că traseul este circular sau într-un singur sens.
    # Pentru unele linii circulare, cum ar fi cele care se întorc în același punct,
    # regula cu 3 pagini poate să nu funcționeze.
    print("  Nu am detectat automat sens întors. Traseul va fi tratat ca un singur sens.")
    return None


def build_direction_names(page_stations, reverse_start_page):
    """
    Construiește numele sensurilor.

    Dacă avem reverse_start_page:
    - prima stație = începutul dusului
    - stația de pe pagina reverse_start_page - 1 = capătul dusului

    Exemplu:
    first_station = GARA
    terminus_station = DANCU

    Rezultat:
    GARA_TO_DANCU
    DANCU_TO_GARA

    Dacă nu avem reverse_start_page, întoarcem un singur sens:
    SINGLE_DIRECTION
    """

    first_station = page_stations[0]["station"]

    if reverse_start_page is None:
        return "SINGLE_DIRECTION", None

    terminus_station = None

    for item in page_stations:
        if item["page"] == reverse_start_page - 1:
            terminus_station = item["station"]
            break

    if first_station is None or terminus_station is None:
        return "DUS", "INTORS"

    outbound_direction = f"{first_station}_TO_{terminus_station}"
    return_direction = f"{terminus_station}_TO_{first_station}"

    return outbound_direction, return_direction


# ============================================================
# 8. PARSARE UN SINGUR PDF / O SINGURĂ LINIE
# ============================================================

def parse_single_route(route):
    """
    Parsează PDF-ul pentru o singură linie CTP.

    route este un dicționar din ctp_routes.json:
    {
      "route_id": "3",
      "vehicle_type": "tramvai",
      "route_name": "...",
      "pdf_file": "track-3.pdf"
    }
    """

    route_id = route["route_id"]
    pdf_file = route["pdf_file"]
    pdf_path = TIMETABLE_DIR / pdf_file

    if not pdf_path.exists():
        print(f"  PDF lipsă: {pdf_file}")
        return None

    page_stations = collect_page_stations(pdf_path)

    if "reverse_start_page" in route and route["reverse_start_page"] is not None:
        reverse_start_page = route["reverse_start_page"]
        print(f"  reverse_start_page hardcodat: pagina {reverse_start_page}")
    else:
        reverse_start_page = detect_reverse_start_page(page_stations)

    outbound_direction, return_direction = build_direction_names(
        page_stations,
        reverse_start_page
    )

    directions = {
        outbound_direction: {}
    }

    if return_direction is not None:
        directions[return_direction] = {}

    route_result = {
        "route_id": route_id,
        "vehicle_type": route["vehicle_type"],
        "route_name": route["route_name"],
        "source_pdf": pdf_file,
        "reverse_start_page": reverse_start_page,
        "directions": directions
    }

    current_station = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            if not tables:
                continue

            station_name = extract_station_name(tables)

            if station_name is not None:
                current_station = station_name

            if current_station is None:
                continue

            schedule_table = find_schedule_table(tables)

            if schedule_table is None:
                continue

            page_schedule = parse_schedule_table(schedule_table, route_id)

            if reverse_start_page is None:
                direction = outbound_direction
            elif page_index < reverse_start_page:
                direction = outbound_direction
            else:
                direction = return_direction

            direction_data = route_result["directions"][direction]

            if current_station not in direction_data:
                direction_data[current_station] = create_empty_station_schedule()

            direction_data[current_station] = merge_schedules(
                direction_data[current_station],
                page_schedule
            )

    return route_result


# ============================================================
# 9. PARSARE TOATE LINIILE
# ============================================================

def parse_all_routes():
    """
    Citește toate liniile din ctp_routes.json și parsează PDF-ul fiecăreia.

    Rezultatul final va fi un dicționar:
    {
      "3": {...orar linia 3...},
      "5": {...orar linia 5...}
    }
    """

    routes = load_routes_metadata()

    all_timetables = {}

    for route in routes:
        route_id = route["route_id"]

        print(f"Procesez linia {route_id} ({route['vehicle_type']})...")

        try:
            route_timetable = parse_single_route(route)

            if route_timetable is None:
                continue

            all_timetables[route_id] = route_timetable

            print(f"  Linia {route_id} procesată cu succes.")

        except Exception as error:
            # Dacă o linie are PDF diferit/stricat, nu vrem să pice tot programul.
            # Afișăm eroarea și continuăm cu următoarea linie.
            print(f"  Eroare la linia {route_id}: {error}")

    return all_timetables


# ============================================================
# 10. SALVARE JSON FINAL
# ============================================================

def save_timetable(timetable):
    """
    Salvează JSON-ul final în data/processed/orar_ctp_all.json.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(timetable, file, indent=2, ensure_ascii=False)

    print(f"\nOrarul complet a fost salvat în: {OUTPUT_PATH}")


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":
    timetable = parse_all_routes()
    save_timetable(timetable)