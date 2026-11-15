"""
Coffeelab CFO-dashboard
AI miniopdracht 3: AI data applicatie (dashboard, cards, filters)
AI miniopdracht 5: AI data met AskAI en AI advies

Doelgroep: de CFO van Coffeelab, die snel inzicht wil in omzet, openstaande
bedragen (cashflow) en spreiding over provincies/productcategorieen, om
strategische keuzes te onderbouwen (zie Coffeelab-casus).

De app heeft drie pagina's:
1. Dashboard  - cards, filters, grafieken en de knop "Genereer AI-inzicht".
2. AskAI      - stel een vraag in gewone taal over de data.
3. Validatie  - vaste testvragen: AskAI tegenover een traditionele
                pandas-analyse met bekende uitkomsten.

Hoe AskAI werkt (text-to-pandas, alles via directe API-calls naar Groq):
  stap 1  Groq krijgt de vraag + alleen de kolomnamen en een paar
          voorbeeldwaarden (geen ruwe data) en schrijft EEN pandas-expressie.
  stap 2  De app controleert die code (alleen lezen: geen imports, geen
          bestanden, data niet aanpassen) en rekent het antwoord zelf uit op
          de echte data. Geeft de code een fout, dan krijgt Groq die fout
          terug en mag hij het een keer verbeteren.
  stap 3  Groq krijgt het BEREKENDE resultaat en schrijft daarover een kort
          antwoord en advies voor de CFO.
Zo komen de cijfers uit pandas en niet uit het "geheugen" van het taalmodel,
en kan de gebruiker in het controlepaneel precies zien hoe het is berekend.
"""

import ast
import json
import math
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------
DATA_DIR = "data"
GROQ_MODEL = "openai/gpt-oss-120b"  # zelfde model als in opdracht 1 (EDA) en 3

# Koffiestijl kleurenpalet (gebruikt in grafieken)
COFFEE_COLORS = ["#6F4E37", "#C68B59", "#A9746E", "#8B5E3C", "#D9B48F", "#4B3621", "#E8C39E"]

MAAND_NAMEN = {
    1: "jan", 2: "feb", 3: "mrt", 4: "apr", 5: "mei", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "dec",
}

PAGINA_DASHBOARD = "📊 Dashboard"
PAGINA_ASKAI = "💬 AskAI"
PAGINA_VALIDATIE = "✅ Validatie"

st.set_page_config(
    page_title="Coffeelab CFO-dashboard",
    page_icon="☕",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Koffiestijl (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #FFF8F0; }
    [data-testid="stSidebar"] { background-color: #F0E0CC; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    h1, h2, h3 { color: #4B3621; }
    p, label, span, div { color: #3B2317; }

    div[data-testid="stMetric"] {
        background-color: #FFF3E0;
        border: 1px solid #D9B48F;
        padding: 14px 14px 8px 14px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] { color: #6F4E37; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #4B3621; }

    .stButton button, .stDownloadButton button {
        background-color: #6F4E37;
        color: #FFF8F0;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #4B3621; color: #FFF8F0;
    }
    .stButton button p, .stDownloadButton button p { color: #FFF8F0; }

    [data-testid="stExpander"] {
        background-color: #FBEFE0;
        border-radius: 8px;
        border: 1px solid #E3C9A3;
    }
    [data-testid="stChatMessage"] {
        background-color: #FBEFE0;
        border-radius: 12px;
        border: 1px solid #E3C9A3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data laden en samenvoegen
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    verkoop = pd.read_parquet(os.path.join(DATA_DIR, "f_Verkooptransacties.parquet"))
    artikelen = pd.read_parquet(os.path.join(DATA_DIR, "d_Artikelen.parquet"))
    klanten = pd.read_parquet(os.path.join(DATA_DIR, "d_Klanten.parquet"))

    verkoop["Orderdatum"] = pd.to_datetime(verkoop["Orderdatum"], errors="coerce")

    df = verkoop.merge(
        artikelen[["Artikelnummer", "Product L1", "Product L2"]],
        on="Artikelnummer",
        how="left",
    )
    df = df.merge(
        klanten[["Klantnummer", "Provincie", "Type"]],
        on="Klantnummer",
        how="left",
    )

    df["_jaar"] = df["Orderdatum"].dt.year
    df["_jaarmaand"] = df["Orderdatum"].dt.to_period("M").astype(str)
    return df


df = load_data()


# ---------------------------------------------------------------------------
# Kleine hulpfuncties voor weergave
# ---------------------------------------------------------------------------
def nl_getal(x, decimalen=None):
    """Getal in Nederlandse notatie: 1.234.567,89"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if decimalen is None and x.is_integer() and 1900 <= x <= 2100:
        return str(int(x))  # waarschijnlijk een jaartal: 2024, niet 2.024
    if decimalen is None:
        decimalen = 0 if x.is_integer() else 2
    tekst = f"{x:,.{decimalen}f}"
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")


def is_getal(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def voor_weergave(res):
    """Maakt een Series/DataFrame geschikt voor st.dataframe (Period/datum-index -> tekst)."""
    if isinstance(res, pd.Series):
        res = res.to_frame(name=res.name if res.name is not None else "waarde")
    res = res.copy()
    if isinstance(res.index, pd.MultiIndex):
        res = res.reset_index()
    if not pd.api.types.is_numeric_dtype(res.index):
        res.index = res.index.astype(str)
    for kol in res.columns:
        if isinstance(res[kol].dtype, pd.PeriodDtype):
            res[kol] = res[kol].astype(str)
    res.columns = [str(c) for c in res.columns]
    return res


def resultaat_als_tekst(res, max_rijen=25):
    """Zet het berekende resultaat om in tekst die naar het taalmodel kan."""
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        if isinstance(res, (pd.DataFrame, pd.Series)):
            tekst = res.head(max_rijen).to_string()
            if len(res) > max_rijen:
                tekst += f"\n... ({len(res)} rijen in totaal, eerste {max_rijen} getoond)"
            return tekst
    if is_getal(res):
        return f"{float(res):,.4f}".rstrip("0").rstrip(".")
    return str(res)


def kort_resultaat(res):
    """Korte samenvatting van een resultaat voor in een tabel."""
    if res is None:
        return "—"
    if isinstance(res, (pd.Series, pd.DataFrame)):
        if len(res) == 0:
            return f"lege {type(res).__name__}"
        eerste = res.iloc[0] if isinstance(res, pd.Series) else res.iloc[0, 0]
        return f"{type(res).__name__} ({len(res)} rijen), eerste: {res.index[0]} = {eerste}"
    if is_getal(res):
        return nl_getal(res)
    return str(res)


# ---------------------------------------------------------------------------
# Groq-verbinding (gedeeld door AI-inzicht, AskAI en Validatie)
# ---------------------------------------------------------------------------
GEEN_SLEUTEL_MELDING = (
    "Geen GROQ_API_KEY gevonden. Lokaal: zet 'set GROQ_API_KEY=jouw_sleutel' "
    "in je terminal voordat je de app start. Op Streamlit Cloud: zet hem bij "
    "Settings -> Secrets van de app."
)


def maak_groq_client():
    # Lokaal: via 'set GROQ_API_KEY=...' in de terminal (omgevingsvariabele).
    # Op Streamlit Community Cloud: via Settings -> Secrets (st.secrets).
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        st.error(GEEN_SLEUTEL_MELDING)
        return None
    # max_retries: bij "te veel verzoeken" (rate limit) wacht de client even en
    # probeert het opnieuw. Handig bij de validatietest met veel vragen achter elkaar.
    return Groq(api_key=api_key, max_retries=5)


def groq_chat(client, messages, json_modus=False, temperature=0.2):
    kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=temperature)
    if json_modus:
        try:
            r = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            return r.choices[0].message.content
        except Exception as e:
            # JSON-modus mislukt (model gaf geen geldige JSON): nog een keer zonder.
            if type(e).__name__ != "BadRequestError":
                raise
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content


# ---------------------------------------------------------------------------
# AskAI stap 1: schema (beschrijving van de kolommen) voor het taalmodel
# ---------------------------------------------------------------------------
@st.cache_data
def maak_schema():
    """Beschrijft de kolommen van df. Alleen dit (geen ruwe data) gaat naar Groq."""
    d = load_data()
    regels = [
        f"DataFrame `df` met {len(d):,} rijen. Eén rij = één artikelregel op een order "
        "(f_Verkooptransacties, gekoppeld aan d_Artikelen en d_Klanten).",
        "Kolommen:",
    ]
    for kol in d.columns:
        if kol.startswith("_"):
            continue
        s = d[kol]
        if pd.api.types.is_datetime64_any_dtype(s):
            info = f"datum, van {s.min():%Y-%m-%d} t/m {s.max():%Y-%m-%d}"
        elif pd.api.types.is_numeric_dtype(s):
            info = f"getal, min {s.min():,.2f}, max {s.max():,.2f}"
        else:
            waarden = s.dropna().astype(str).unique()
            if len(waarden) <= 25:
                info = f"tekst, {len(waarden)} waarden: " + ", ".join(repr(v) for v in sorted(waarden))
            else:
                info = f"tekst, {len(waarden):,} unieke waarden, bijv. " + ", ".join(
                    repr(v) for v in waarden[:3]
                )
        regels.append(f'- df["{kol}"] ({s.dtype}): {info}')
    return "\n".join(regels)


CODE_PROMPT = """Je zet vragen van de CFO van Coffeelab (B2B-verkoop van koffie en koffiemachines) om in pandas-code.

{schema}

Regels voor de code:
- Schrijf precies EEN Python-expressie (mag over meerdere regels binnen haakjes). Geen toewijzingen, geen imports, geen print, geen lambda, geen list comprehensions.
- Gebruik alleen `df` en `pd`. Schrijf kolommen altijd als df["Kolomnaam"] (sommige namen bevatten spaties).
- De data mag niet worden aangepast: geen inplace=True.
- Vraagt de vraag om één getal of één naam, geef dan precies één waarde terug (bijv. met .sum(), .nunique(), .idxmax()).
- Vraagt de vraag om een overzicht of vergelijking, geef dan een Series of DataFrame terug, logisch gesorteerd en maximaal 20 rijen.
- Is de vraag niet te beantwoorden met deze kolommen, geef dan als code precies "None" en leg in de uitleg uit waarom.
"""

DEFINITIES = """
Bedrijfsdefinities van Coffeelab (gebruik deze altijd):
- Omzet = som van df["Bedrag"].
- Aantal orders = aantal unieke df["Ordernummer"] (.nunique()), niet het aantal rijen.
- Openstaand = rijen met df["Status"] == "Openstaand". Openstaand bedrag = som van Bedrag van die rijen.
- Percentage openstaand = openstaand bedrag / totale omzet * 100 (geef het percentage, niet de fractie).
- Gemiddelde orderwaarde = totale omzet / aantal unieke orders.
- Jaar = df["Orderdatum"].dt.year, maand = df["Orderdatum"].dt.to_period("M").
"""

JSON_INSTRUCTIE = """
Antwoord alleen met JSON in dit formaat:
{"code": "<de pandas-expressie>", "uitleg": "<één of twee zinnen in het Nederlands: wat berekent de code en welke keuzes/aannames zijn gemaakt>"}
"""


def lees_json_antwoord(tekst):
    """Haalt code en uitleg uit het antwoord van het model (ook als het geen nette JSON is)."""
    tekst = (tekst or "").strip()
    data = None
    try:
        data = json.loads(tekst)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", tekst, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        m = re.search(r"```(?:python)?\s*(.*?)```", tekst, re.S)
        data = {"code": m.group(1) if m else tekst, "uitleg": ""}
    code = str(data.get("code", "")).strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
    return code, str(data.get("uitleg", "")).strip()


def genereer_code(client, vraag, schema, met_definities=True, vorige_code=None, vorige_fout=None):
    systeem = CODE_PROMPT.format(schema=schema) + (DEFINITIES if met_definities else "") + JSON_INSTRUCTIE
    messages = [
        {"role": "system", "content": systeem},
        {"role": "user", "content": vraag},
    ]
    if vorige_fout:
        messages.append({"role": "assistant", "content": json.dumps({"code": vorige_code})})
        messages.append({
            "role": "user",
            "content": f"Deze code gaf een fout: {vorige_fout}\nGeef verbeterde code in hetzelfde JSON-formaat.",
        })
    return lees_json_antwoord(groq_chat(client, messages, json_modus=True, temperature=0.1))


# ---------------------------------------------------------------------------
# AskAI stap 2: de code controleren en veilig uitvoeren
# ---------------------------------------------------------------------------
# Dit is een eenvoudige "zandbak": de code van het taalmodel mag alleen
# rekenen met df en pd. Het is geen volledige beveiliging, maar het blokkeert
# het belangrijkste: imports, bestanden lezen/schrijven en de data aanpassen.
VEILIGE_BUILTINS = {
    "len": len, "round": round, "sum": sum, "min": min, "max": max, "abs": abs,
    "int": int, "float": float, "str": str, "sorted": sorted, "list": list, "dict": dict,
}
TOEGESTANE_NAMEN = {"df", "pd"} | set(VEILIGE_BUILTINS)
VERBODEN_ATTRIBUTEN = {
    "eval", "query", "pipe", "merge", "join", "insert", "pop", "update",
    "plot", "hist", "boxplot", "options", "set_option", "reset_option", "get_option",
    "describe_option", "io", "api", "core", "util", "testing", "plotting",
    "ExcelWriter", "HDFStore", "attrs", "flags",
}
TOEGESTANE_TO = {
    "to_period", "to_timestamp", "to_list", "to_dict", "to_frame", "to_numpy",
    "to_string", "to_datetime", "to_numeric", "to_timedelta",
}
VERBODEN_KNOPEN = (
    ast.Lambda, ast.NamedExpr, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
)


class OnveiligeCode(Exception):
    pass


def is_verboden_naam(naam):
    return (
        naam.startswith("_")
        or naam in VERBODEN_ATTRIBUTEN
        or naam.startswith("read_")
        or (naam.startswith("to_") and naam not in TOEGESTANE_TO)
    )


def controleer_code(code):
    try:
        boom = ast.parse(code.strip(), mode="eval")
    except SyntaxError as e:
        raise OnveiligeCode(f"geen geldige enkele Python-expressie ({e.msg})")
    for knoop in ast.walk(boom):
        if isinstance(knoop, VERBODEN_KNOPEN):
            raise OnveiligeCode(f"'{type(knoop).__name__}' is niet toegestaan; gebruik pandas-methodes")
        if isinstance(knoop, ast.Name) and knoop.id not in TOEGESTANE_NAMEN:
            raise OnveiligeCode(f"naam '{knoop.id}' is niet toegestaan (alleen df, pd en basisfuncties)")
        if isinstance(knoop, ast.Attribute) and is_verboden_naam(knoop.attr):
            raise OnveiligeCode(f"'.{knoop.attr}' is niet toegestaan")
        # pandas kan methodes ook via een tekst aanroepen, bijv. df.apply("to_csv")
        if isinstance(knoop, ast.Constant) and isinstance(knoop.value, str) and is_verboden_naam(knoop.value):
            raise OnveiligeCode(f"de tekst '{knoop.value}' is niet toegestaan")
        if isinstance(knoop, ast.keyword) and knoop.arg == "inplace":
            raise OnveiligeCode("'inplace' is niet toegestaan: de data mag niet worden aangepast")
    return compile(boom, "<askai>", "eval")


def voer_code_uit(code, data):
    gecompileerd = controleer_code(code)
    omgeving = {"__builtins__": VEILIGE_BUILTINS, "df": data, "pd": pd}
    return eval(gecompileerd, omgeving, {})


# ---------------------------------------------------------------------------
# AskAI stap 3: antwoord en advies op basis van het berekende resultaat
# ---------------------------------------------------------------------------
ADVIES_PROMPT = """Je bent een data-analist die de CFO van Coffeelab adviseert (B2B-verkoop van koffie en koffiemachines).
Je krijgt een vraag van de CFO, de pandas-code die is uitgevoerd en het BEREKENDE resultaat.

Regels:
- Gebruik alleen getallen die in het resultaat staan. Je mag afronden of een eenvoudig percentage uitrekenen uit die getallen. Verzin geen cijfers.
- Beantwoordt het resultaat de vraag niet (helemaal), zeg dat dan eerlijk.
- Schrijf in eenvoudig Nederlands, zonder jargon. Bedragen als € 1.234.567. Gebruik geen $-tekens.

Gebruik precies dit format (maximaal 130 woorden):
**Antwoord:** één of twee zinnen met het antwoord op de vraag.
**Advies:** één tot drie zinnen met een concreet advies voor de CFO.
**Let op:** één zin over een aanname of beperking (bijvoorbeeld de definitie, de periode of het filter).
"""


def schrijf_advies(client, vraag, code, uitleg, resultaat_tekst, bron_label):
    inhoud = (
        f"Vraag van de CFO: {vraag}\n\n"
        f"Databron: {bron_label}\n\n"
        f"Uitgevoerde pandas-code:\n{code}\n\n"
        f"Uitleg bij de code: {uitleg}\n\n"
        f"Berekend resultaat:\n{resultaat_tekst}"
    )
    tekst = groq_chat(
        client,
        [{"role": "system", "content": ADVIES_PROMPT}, {"role": "user", "content": inhoud}],
        temperature=0.3,
    )
    tekst = (tekst or "").replace("$", "\\$")  # $ zou in Streamlit als formule worden gelezen
    return re.sub(r"\n+(?=\*\*)", "\n\n", tekst)  # elk kopje op een eigen regel


def beantwoord_vraag(client, vraag, data, schema, bron_label, met_definities=True,
                     met_advies=True, max_pogingen=2):
    """De volledige AskAI-pijplijn. Geeft een dict terug met alle tussenstappen."""
    start = time.perf_counter()
    item = {
        "vraag": vraag, "bron": bron_label, "code": "", "uitleg": "", "resultaat": None,
        "fout": None, "pogingen": [], "advies": None, "geen_antwoord": False,
        "tijdstip": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    code, fout = None, None
    for poging in range(1, max_pogingen + 1):
        try:
            code, uitleg = genereer_code(client, vraag, schema, met_definities, code, fout)
        except Exception as e:
            item["fout"] = f"Groq-fout: {type(e).__name__}: {e}"
            break
        item["code"], item["uitleg"] = code, uitleg
        if code in ("None", ""):
            item["geen_antwoord"] = True
            item["fout"] = None
            break
        try:
            item["resultaat"] = voer_code_uit(code, data)
            item["fout"] = None
            break
        except Exception as e:
            fout = f"{type(e).__name__}: {e}"
            item["fout"] = fout
            item["pogingen"].append({"poging": poging, "code": code, "fout": fout})

    if met_advies and item["fout"] is None:
        try:
            if item["geen_antwoord"]:
                item["advies"] = (
                    "**Antwoord:** deze vraag kan niet met de beschikbare data worden "
                    f"beantwoord. {item['uitleg']}"
                )
            else:
                item["advies"] = schrijf_advies(
                    client, vraag, item["code"], item["uitleg"],
                    resultaat_als_tekst(item["resultaat"]), bron_label,
                )
        except Exception as e:
            item["fout"] = f"Groq-fout bij het schrijven van het advies: {type(e).__name__}: {e}"

    # Grote tabellen niet helemaal bewaren in het geheugen van de sessie
    if isinstance(item["resultaat"], (pd.DataFrame, pd.Series)) and len(item["resultaat"]) > 200:
        item["resultaat_rijen"] = len(item["resultaat"])
        item["resultaat"] = item["resultaat"].head(200)
    item["duur"] = round(time.perf_counter() - start, 1)
    return item


def toon_resultaat(res):
    if isinstance(res, (pd.DataFrame, pd.Series)):
        try:
            st.dataframe(voor_weergave(res), width="stretch")
        except Exception:
            st.text(resultaat_als_tekst(res))
    elif is_getal(res):
        st.markdown(f"**{nl_getal(res)}**")
    else:
        st.markdown(f"**{res}**")


def toon_grafiek(res, sleutel):
    """Automatische staafgrafiek als het resultaat een lijstje getallen is."""
    if isinstance(res, pd.Series) and 2 <= len(res) <= 40 and pd.api.types.is_numeric_dtype(res):
        labels = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in res.index]
        plot_df = pd.DataFrame({"categorie": labels, "waarde": res.values})
        fig = px.bar(plot_df, x="categorie", y="waarde", color_discrete_sequence=[COFFEE_COLORS[0]])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=None, yaxis_title=res.name or None)
        st.plotly_chart(fig, width="stretch", key=sleutel)


def toon_askai_antwoord(item, index):
    if item["fout"]:
        st.error(f"Het lukte niet om deze vraag te beantwoorden. {item['fout']}")
    elif item["advies"]:
        st.markdown(item["advies"])
    elif item["resultaat"] is not None:
        toon_resultaat(item["resultaat"])

    if item["resultaat"] is not None and not item["fout"]:
        toon_grafiek(item["resultaat"], sleutel=f"askai_grafiek_{index}")

    with st.expander("🔍 Controle: hoe is dit berekend?"):
        st.caption(f"Databron: {item['bron']} · duur: {item.get('duur', '?')} s · {item['tijdstip']}")
        if item["uitleg"]:
            st.markdown(f"**Uitleg van de AI:** {item['uitleg']}")
        if item["code"]:
            st.markdown("**Door de AI geschreven pandas-code** (uitgevoerd door de app op de echte data):")
            st.code(item["code"], language="python")
        for p in item["pogingen"]:
            st.warning(f"Poging {p['poging']} gaf een fout en is door de AI verbeterd: {p['fout']}")
            st.code(p["code"], language="python")
        if item["resultaat"] is not None:
            extra = f" (eerste 200 van {item['resultaat_rijen']:,} rijen)" if item.get("resultaat_rijen") else ""
            st.markdown(f"**Berekend resultaat**{extra}:")
            toon_resultaat(item["resultaat"])


# ---------------------------------------------------------------------------
# Validatie: testvragen met bekende uitkomsten (traditionele pandas-analyse)
# ---------------------------------------------------------------------------
VALIDATIE_VRAGEN = [
    {"id": 1, "vraag": "Wat is de totale omzet?",
     "referentie": 'df["Bedrag"].sum()'},
    {"id": 2, "vraag": "Hoeveel orders zijn er in totaal?",
     "referentie": 'df["Ordernummer"].nunique()'},
    {"id": 3, "vraag": "Hoeveel euro staat er nog open?",
     "referentie": 'df.loc[df["Status"] == "Openstaand", "Bedrag"].sum()'},
    {"id": 4, "vraag": "Welk percentage van de omzet staat nog open?",
     "referentie": 'df.loc[df["Status"] == "Openstaand", "Bedrag"].sum() / df["Bedrag"].sum() * 100'},
    {"id": 5, "vraag": "Hoeveel orders staan nog open?",
     "referentie": 'df.loc[df["Status"] == "Openstaand", "Ordernummer"].nunique()'},
    {"id": 6, "vraag": "Wat is de gemiddelde orderwaarde?",
     "referentie": 'df["Bedrag"].sum() / df["Ordernummer"].nunique()'},
    {"id": 7, "vraag": "Welke provincie heeft de hoogste omzet?",
     "referentie": 'df.groupby("Provincie")["Bedrag"].sum().idxmax()'},
    {"id": 8, "vraag": "Welke provincie heeft de laagste omzet?",
     "referentie": 'df.groupby("Provincie")["Bedrag"].sum().idxmin()'},
    {"id": 9, "vraag": "In welk jaar was de omzet het hoogst?",
     "referentie": 'df.groupby(df["Orderdatum"].dt.year)["Bedrag"].sum().idxmax()'},
    {"id": 10, "vraag": "Hoeveel verschillende klanten hebben iets gekocht?",
     "referentie": 'df["Klantnummer"].nunique()'},
    {"id": 11, "vraag": "Welk percentage van de omzet komt uit koffiemachines?",
     "referentie": 'df.loc[df["Product L1"].str.contains("machine", case=False, na=False), "Bedrag"].sum() / df["Bedrag"].sum() * 100'},
    {"id": 12, "vraag": "Welke productgroep (Product L2) heeft de hoogste omzet?",
     "referentie": 'df.groupby("Product L2")["Bedrag"].sum().idxmax()'},
]

# Moeilijkere testvragen: nog steeds één controleerbaar antwoord, maar AskAI moet
# meerdere stappen combineren (groeperen, delen, filteren op jaar, drempels,
# kostprijs gebruiken) en soms zelf bepalen wat een begrip betekent.
VALIDATIE_MOEILIJK = [
    {"id": 101, "vraag": "Met hoeveel procent is de omzet gegroeid van 2022 naar 2023?",
     "referentie": '(df.loc[df["Orderdatum"].dt.year == 2023, "Bedrag"].sum() / df.loc[df["Orderdatum"].dt.year == 2022, "Bedrag"].sum() - 1) * 100'},
    {"id": 102, "vraag": "Welke provincie heeft het hoogste percentage openstaande omzet?",
     "referentie": '(df.loc[df["Status"] == "Openstaand"].groupby("Provincie")["Bedrag"].sum() / df.groupby("Provincie")["Bedrag"].sum()).idxmax()'},
    {"id": 103, "vraag": "Hoeveel klanten hebben in totaal meer dan 1 miljoen euro omzet?",
     "referentie": '(df.groupby("Klantnummer")["Bedrag"].sum() > 1000000).sum()'},
    {"id": 104, "vraag": "Wat is de gemiddelde orderwaarde van de orders die nog openstaan?",
     "referentie": 'df.loc[df["Status"] == "Openstaand", "Bedrag"].sum() / df.loc[df["Status"] == "Openstaand", "Ordernummer"].nunique()'},
    {"id": 105, "vraag": "Welk type klant heeft de hoogste gemiddelde orderwaarde?",
     "referentie": '(df.groupby("Type")["Bedrag"].sum() / df.groupby("Type")["Ordernummer"].nunique()).idxmax()'},
    {"id": 106, "vraag": "Hoeveel brutomarge in euro heeft Coffeelab in totaal gemaakt?",
     "referentie": 'df["Bedrag"].sum() - df["Kostprijs omzet"].sum()'},
    {"id": 107, "vraag": "Welke productgroep (Product L2) heeft de hoogste marge in procenten?",
     "referentie": '((df.groupby("Product L2")["Bedrag"].sum() - df.groupby("Product L2")["Kostprijs omzet"].sum()) / df.groupby("Product L2")["Bedrag"].sum()).idxmax()'},
    {"id": 108, "vraag": "Welk percentage van de klanten heeft nog een openstaand bedrag?",
     "referentie": 'df.loc[df["Status"] == "Openstaand", "Klantnummer"].nunique() / df["Klantnummer"].nunique() * 100'},
    {"id": 109, "vraag": "Hoeveel orders hebben een totale orderwaarde van meer dan 1.000 euro?",
     "referentie": '(df.groupby("Ordernummer")["Bedrag"].sum() > 1000).sum()'},
    {"id": 110, "vraag": "Hoeveel euro verschil in omzet zit er tussen de provincie met de hoogste en de laagste omzet?",
     "referentie": 'df.groupby("Provincie")["Bedrag"].sum().max() - df.groupby("Provincie")["Bedrag"].sum().min()'},
    {"id": 111, "vraag": "In welke provincie is de omzet van koffiemachines het hoogst?",
     "referentie": 'df.loc[df["Product L1"].str.contains("machine", case=False, na=False)].groupby("Provincie")["Bedrag"].sum().idxmax()'},
    {"id": 112, "vraag": "Welke klant had in 2023 de hoogste omzet?",
     "referentie": 'df.loc[df["Orderdatum"].dt.year == 2023].groupby("Klantnummer")["Bedrag"].sum().idxmax()'},
]

TESTSETS = {
    "Basis (12 vragen)": VALIDATIE_VRAGEN,
    "Moeilijk (12 vragen)": VALIDATIE_MOEILIJK,
}


@st.cache_data
def referentie_uitkomsten():
    d = load_data()
    uitkomsten = {}
    for t in VALIDATIE_VRAGEN + VALIDATIE_MOEILIJK:
        start = time.perf_counter()
        try:
            waarde = voer_code_uit(t["referentie"], d)
        except Exception as e:  # bijv. als een kolom ontbreekt
            waarde = f"(referentie mislukt: {type(e).__name__})"
        uitkomsten[t["id"]] = {"waarde": waarde, "tijd": time.perf_counter() - start}
    return uitkomsten


def naar_getal(x):
    if is_getal(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace("%", "").replace("€", "").strip())
        except ValueError:
            return None
    return None


def vergelijk(ai_res, verwacht, rel_tol=1e-3):
    """Vergelijkt het AskAI-resultaat met de bekende uitkomst. Geeft (status, toelichting)."""
    # Eén waarde uit een tabel van 1x1 of een Series van lengte 1 halen
    lijst_res = None
    if isinstance(ai_res, pd.DataFrame):
        if ai_res.shape == (1, 1):
            ai_res = ai_res.iat[0, 0]
        else:
            lijst_res = ai_res.iloc[:, 0] if ai_res.shape[1] else None
    elif isinstance(ai_res, pd.Series):
        if len(ai_res) == 1 and is_getal(verwacht):
            ai_res = ai_res.iloc[0]
        else:
            lijst_res = ai_res

    if isinstance(verwacht, str):
        if lijst_res is not None:
            if len(lijst_res) and str(lijst_res.index[0]).strip().lower() == verwacht.lower():
                return "⚠️ Deels juist", "juiste naam bovenaan, maar als lijst i.p.v. één naam"
            return "❌ Onjuist", "lijst teruggegeven, en de juiste naam staat niet bovenaan"
        if str(ai_res).strip().lower() == verwacht.strip().lower():
            return "✅ Juist", ""
        return "❌ Onjuist", f"verwacht '{verwacht}'"

    v = naar_getal(verwacht)
    if lijst_res is not None:
        return "❌ Onjuist", "lijst teruggegeven i.p.v. één getal"
    a = naar_getal(ai_res)
    if a is None:
        return "❌ Onjuist", f"geen getal teruggegeven ({type(ai_res).__name__})"
    if math.isclose(a, v, rel_tol=rel_tol, abs_tol=1e-6):
        return "✅ Juist", ""
    if math.isclose(a * 100, v, rel_tol=rel_tol) or math.isclose(a / 100, v, rel_tol=rel_tol):
        return "⚠️ Deels juist", "juiste uitkomst, maar in een andere eenheid (fractie i.p.v. procent)"
    afwijking = (a - v) / v * 100 if v else float("inf")
    return "❌ Onjuist", f"afwijking {nl_getal(afwijking, 1)}%"


# ---------------------------------------------------------------------------
# Hulpfunctie: aan-/afvinkbare filtergroep met "Alles" / "Geen" knoppen
# ---------------------------------------------------------------------------
def checkbox_filter(label, opties, key_prefix, expanded=True):
    with st.sidebar.expander(label, expanded=expanded):
        c1, c2 = st.columns(2)
        if c1.button("Alles", key=f"{key_prefix}_alles_btn", width="stretch"):
            for optie in opties:
                st.session_state[f"{key_prefix}_{optie}"] = True
        if c2.button("Geen", key=f"{key_prefix}_geen_btn", width="stretch"):
            for optie in opties:
                st.session_state[f"{key_prefix}_{optie}"] = False

        geselecteerd = []
        for optie in opties:
            k = f"{key_prefix}_{optie}"
            if k not in st.session_state:
                st.session_state[k] = True
            if st.checkbox(str(optie), key=k):
                geselecteerd.append(optie)
    return geselecteerd


# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("☕ Filters")

# --- Periode: duidelijk kiezen tussen Dag / Maand / Jaar ---
st.sidebar.subheader("Periode")
granulariteit = st.sidebar.radio(
    "Type periode", ["Dag", "Maand", "Jaar"], horizontal=True, key="periode_type"
)

min_datum = df["Orderdatum"].min()
max_datum = df["Orderdatum"].max()

if granulariteit == "Jaar":
    jaar_opties = sorted(df["_jaar"].dropna().unique().astype(int).tolist())
    gekozen_jaren = st.sidebar.multiselect(
        "Jaren", jaar_opties, default=jaar_opties, key="jaren_select"
    )
    periode_mask = df["_jaar"].isin(gekozen_jaren)

elif granulariteit == "Maand":
    maand_opties = sorted(df["_jaarmaand"].dropna().unique().tolist())
    maand_labels = {
        ym: f"{MAAND_NAMEN[int(ym.split('-')[1])]} {ym.split('-')[0]}" for ym in maand_opties
    }
    gekozen_maanden = st.sidebar.multiselect(
        "Maanden",
        maand_opties,
        default=maand_opties,
        format_func=lambda ym: maand_labels[ym],
        key="maanden_select",
    )
    periode_mask = df["_jaarmaand"].isin(gekozen_maanden)

else:  # Dag
    periode = st.sidebar.date_input(
        "Exacte periode (van - tot)",
        value=(min_datum, max_datum),
        min_value=min_datum,
        max_value=max_datum,
        key="dag_select",
    )
    if isinstance(periode, tuple) and len(periode) == 2:
        start, eind = periode
    else:
        start, eind = min_datum, max_datum
    periode_mask = (df["Orderdatum"] >= pd.Timestamp(start)) & (
        df["Orderdatum"] <= pd.Timestamp(eind)
    )

# --- Provincie / productcategorie / status: aan-/afvinkbaar met Alles/Geen ---
provincies = sorted(df["Provincie"].dropna().unique().tolist())
gekozen_provincies = checkbox_filter("Provincie (d_Klanten)", provincies, "prov")

categorieen = sorted(df["Product L1"].dropna().unique().tolist())
gekozen_categorieen = checkbox_filter("Productcategorie (Product L1)", categorieen, "cat")

statussen = sorted(df["Status"].dropna().unique().tolist())
gekozen_status = checkbox_filter("Status (f_Verkooptransacties)", statussen, "stat")

# ---------------------------------------------------------------------------
# Filters toepassen
# ---------------------------------------------------------------------------
# 'gefilterd' = alles wat je links hebt aangevinkt, incl. status
mask = (
    periode_mask
    & df["Provincie"].isin(gekozen_provincies)
    & df["Product L1"].isin(gekozen_categorieen)
    & df["Status"].isin(gekozen_status)
)
gefilterd = df[mask]

# 'basis' = periode/provincie/categorie, maar bewust ZONDER statusfilter.
# Nodig om een betrouwbaar "% nog openstaand" te kunnen tonen, ongeacht welke
# statussen je links hebt aan-/uitgevinkt.
basis_mask = (
    periode_mask
    & df["Provincie"].isin(gekozen_provincies)
    & df["Product L1"].isin(gekozen_categorieen)
)
basis = df[basis_mask]

# ---------------------------------------------------------------------------
# Header en navigatie
# ---------------------------------------------------------------------------
st.title("☕ Coffeelab CFO-dashboard")
pagina = st.radio(
    "Pagina",
    [PAGINA_DASHBOARD, PAGINA_ASKAI, PAGINA_VALIDATIE],
    horizontal=True,
    label_visibility="collapsed",
    key="pagina",
)

# ===========================================================================
# PAGINA 1: DASHBOARD (miniopdracht 3)
# ===========================================================================
if pagina == PAGINA_DASHBOARD:
    st.caption(
        "Gebaseerd op f_Verkooptransacties, d_Artikelen en d_Klanten "
        f"— {len(gefilterd):,} van {len(df):,} rijen zichtbaar na filtering."
    )

    # -----------------------------------------------------------------------
    # Cards (KPI's) — reageren op alle filters, incl. status
    # -----------------------------------------------------------------------
    totale_omzet = gefilterd["Bedrag"].sum()
    aantal_orders = gefilterd["Ordernummer"].nunique()
    openstaand_bedrag = gefilterd.loc[gefilterd["Status"] == "Openstaand", "Bedrag"].sum()
    gem_orderwaarde = totale_omzet / aantal_orders if aantal_orders else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Totale omzet", f"€ {totale_omzet:,.0f}")
    col2.metric("Aantal orders", f"{aantal_orders:,}")
    col3.metric("Openstaand bedrag", f"€ {openstaand_bedrag:,.0f}")
    col4.metric("Gem. orderwaarde", f"€ {gem_orderwaarde:,.0f}")

    # -----------------------------------------------------------------------
    # Percentage nog openstaand (los van de statusfilter)
    # -----------------------------------------------------------------------
    omzet_basis = basis["Bedrag"].sum()
    orders_basis = basis["Ordernummer"].nunique()
    omzet_open_basis = basis.loc[basis["Status"] == "Openstaand", "Bedrag"].sum()
    orders_open_basis = basis.loc[basis["Status"] == "Openstaand", "Ordernummer"].nunique()

    pct_omzet_open = (omzet_open_basis / omzet_basis * 100) if omzet_basis else 0
    pct_orders_open = (orders_open_basis / orders_basis * 100) if orders_basis else 0

    st.caption(
        "De twee percentages en taartdiagrammen hieronder tellen altijd over "
        "**alle statussen samen** (los van de statusfilter links), zodat ze "
        "blijven kloppen ongeacht welke status je hebt aan-/uitgevinkt."
    )
    col5, col6 = st.columns(2)
    col5.metric("% omzet nog openstaand", f"{pct_omzet_open:.1f}%")
    col6.metric("% orders nog openstaand", f"{pct_orders_open:.1f}%")

    st.divider()

    # -----------------------------------------------------------------------
    # Visualisaties
    # -----------------------------------------------------------------------
    links, rechts = st.columns(2)

    with links:
        per_maand = (
            gefilterd.dropna(subset=["Orderdatum"])
            .assign(Maand=gefilterd["Orderdatum"].dt.to_period("M").astype(str))
            .groupby("Maand", as_index=False)["Bedrag"].sum()
            .sort_values("Maand")
        )
        fig1 = px.line(per_maand, x="Maand", y="Bedrag", title="Omzet per maand")
        fig1.update_traces(line_color=COFFEE_COLORS[0])
        st.plotly_chart(fig1, width="stretch")

    with rechts:
        per_provincie = (
            gefilterd.groupby("Provincie", as_index=False)["Bedrag"].sum()
            .sort_values("Bedrag", ascending=False)
        )
        fig2 = px.bar(
            per_provincie, x="Provincie", y="Bedrag", title="Omzet per provincie",
            color_discrete_sequence=[COFFEE_COLORS[0]],
        )
        st.plotly_chart(fig2, width="stretch")

    per_categorie = (
        gefilterd.groupby("Product L1", as_index=False)["Bedrag"].sum()
        .sort_values("Bedrag", ascending=False)
    )
    fig3 = px.pie(
        per_categorie, names="Product L1", values="Bedrag",
        title="Omzet per productcategorie (Product L1)",
        color_discrete_sequence=COFFEE_COLORS,
    )
    st.plotly_chart(fig3, width="stretch")

    # --- Betaald vs. openstaand: omzet en orders (los van de statusfilter) ---
    st.subheader("Betaald vs. openstaand")
    p1, p2 = st.columns(2)

    omzet_split = basis.groupby("Status", as_index=False)["Bedrag"].sum()
    fig4 = px.pie(
        omzet_split, names="Status", values="Bedrag",
        title="Omzet: betaald vs. openstaand",
        color_discrete_sequence=COFFEE_COLORS,
    )
    p1.plotly_chart(fig4, width="stretch")

    orders_split = (
        basis.groupby("Status")["Ordernummer"].nunique().reset_index(name="Aantal")
    )
    fig5 = px.pie(
        orders_split, names="Status", values="Aantal",
        title="Orders: betaald vs. openstaand",
        color_discrete_sequence=COFFEE_COLORS,
    )
    p2.plotly_chart(fig5, width="stretch")

    st.divider()

    # -----------------------------------------------------------------------
    # AI-inzicht via API-call (Groq)
    # -----------------------------------------------------------------------
    st.subheader("🤖 AI-inzicht (via Groq API)")
    st.caption(
        "Deze knop stuurt een samenvatting van de huidige gefilterde cijfers "
        "rechtstreeks naar de Groq API en toont het antwoord hieronder — er wordt "
        "geen los chatvenster gebruikt. Wil je zelf een vraag stellen? Ga naar "
        "de pagina **💬 AskAI** bovenaan."
    )

    if st.button("Genereer AI-inzicht"):
        client = maak_groq_client()
        if client:
            samenvatting = {
                "aantal_rijen": int(len(gefilterd)),
                "totale_omzet": round(float(totale_omzet), 2),
                "aantal_orders": int(aantal_orders),
                "openstaand_bedrag": round(float(openstaand_bedrag), 2),
                "gem_orderwaarde": round(float(gem_orderwaarde), 2),
                "pct_omzet_nog_openstaand": round(float(pct_omzet_open), 1),
                "pct_orders_nog_openstaand": round(float(pct_orders_open), 1),
                "omzet_per_provincie": per_provincie.round(2).to_dict(orient="records"),
                "omzet_per_categorie": per_categorie.round(2).to_dict(orient="records"),
            }

            with st.spinner("AI analyseert de gefilterde cijfers..."):
                antwoord = groq_chat(
                    client,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Je bent een data-analist die de CFO van Coffeelab "
                                "ondersteunt. Je krijgt samengevatte cijfers van het "
                                "huidige, gefilterde deel van het verkoopdashboard. "
                                "Geef in het Nederlands: (1) de belangrijkste "
                                "observatie, (2) een risico of aandachtspunt "
                                "(bijvoorbeeld rond het openstaande bedrag), en "
                                "(3) een concrete aanbeveling. Maximaal 150 woorden, "
                                "duidelijke taal, geen jargon zonder uitleg."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Cijfers van het huidige dashboard-filter: {samenvatting}",
                        },
                    ],
                    temperature=0.7,
                )
                st.success(antwoord)

    st.divider()

    # -----------------------------------------------------------------------
    # Gefilterde data (raw preview)
    # -----------------------------------------------------------------------
    with st.expander("Bekijk de gefilterde data (eerste 100 rijen)"):
        st.caption(
            "Dit is een voorbeeld van de onderliggende, rij-voor-rij verkoopdata "
            "(uit f_Verkooptransacties) die na jouw filters overblijft — dus elke "
            "rij is één artikelregel op een order, met de gekoppelde "
            "productcategorie (uit d_Artikelen) en provincie (uit d_Klanten). "
            "De cards en grafieken hierboven zijn samenvattingen van precies "
            "deze rijen."
        )
        st.dataframe(gefilterd.head(100))


# ===========================================================================
# PAGINA 2: AskAI (miniopdracht 5)
# ===========================================================================
elif pagina == PAGINA_ASKAI:
    st.subheader("💬 AskAI: stel een vraag over de Coffeelab-data")
    st.caption(
        "Typ een vraag in gewone taal. De AI (Groq, model gpt-oss-120b) schrijft "
        "pandas-code, de app rekent het antwoord uit op de echte data, en de AI "
        "geeft daarna een kort antwoord en advies. Onder elk antwoord kun je bij "
        "**🔍 Controle** zien hoe het is berekend."
    )

    with st.expander("Hoe werkt AskAI? (3 stappen)"):
        st.markdown(
            "1. **Vraag → code.** Je vraag gaat samen met alleen de kolomnamen en een "
            "paar voorbeeldwaarden naar Groq. De ruwe verkoopdata gaat níet naar de AI. "
            "Groq schrijft één regel pandas-code.\n"
            "2. **Code → resultaat.** De app controleert de code (alleen lezen, geen "
            "bestanden, data niet aanpassen) en voert hem uit op de echte data. Geeft "
            "de code een fout, dan krijgt de AI de foutmelding terug en mag hij het "
            "één keer verbeteren.\n"
            "3. **Resultaat → advies.** Groq krijgt het berekende resultaat en schrijft "
            "een kort antwoord en advies. De cijfers komen dus uit de berekening, niet "
            "uit het geheugen van de AI."
        )

    bron_keuze = st.radio(
        "Vragen stellen over",
        ["Hele dataset", "Huidige filterselectie (zijbalk)"],
        horizontal=True,
        key="askai_bron",
    )
    if bron_keuze == "Hele dataset":
        bron_data = df
        bron_label = f"hele dataset ({len(df):,} rijen)"
    else:
        bron_data = gefilterd
        bron_label = f"filterselectie uit de zijbalk ({len(gefilterd):,} van {len(df):,} rijen)"
    st.caption(f"Databron: {bron_label}")

    st.markdown("**Voorbeeldvragen** (klik om te stellen):")
    voorbeeldvragen = [
        "Welke 5 klanten hebben het hoogste openstaande bedrag?",
        "Hoe ontwikkelt de omzet zich per jaar?",
        "Welke provincie heeft het hoogste percentage openstaande omzet?",
        "Wat is de omzet per productgroep (Product L2)?",
        "In welke maand was de omzet het hoogst?",
        "Waar zit het grootste cashflowrisico?",
    ]
    gekozen_voorbeeld = None
    vb_kolommen = st.columns(2)
    for i, vb in enumerate(voorbeeldvragen):
        if vb_kolommen[i % 2].button(vb, key=f"voorbeeld_{i}", width="stretch"):
            gekozen_voorbeeld = vb

    if "askai_historie" not in st.session_state:
        st.session_state["askai_historie"] = []
    historie = st.session_state["askai_historie"]

    # Eerdere vragen en antwoorden tonen
    for i, item in enumerate(historie):
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(item["vraag"])
        with st.chat_message("assistant", avatar="☕"):
            toon_askai_antwoord(item, i)

    vraag = st.chat_input("Stel een vraag over omzet, orders, klanten, provincies...")
    vraag = vraag or gekozen_voorbeeld

    if vraag:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(vraag)
        with st.chat_message("assistant", avatar="☕"):
            client = maak_groq_client()
            if client:
                with st.spinner("AskAI schrijft code, rekent en schrijft een advies..."):
                    item = beantwoord_vraag(client, vraag, bron_data, maak_schema(), bron_label)
                historie.append(item)
                toon_askai_antwoord(item, len(historie) - 1)

    if historie:
        st.divider()
        c1, c2 = st.columns([1, 1])
        export = pd.DataFrame([
            {
                "tijdstip": h["tijdstip"],
                "vraag": h["vraag"],
                "databron": h["bron"],
                "code": h["code"],
                "uitleg_ai": h["uitleg"],
                "resultaat_kort": kort_resultaat(h["resultaat"]),
                "antwoord_en_advies": (h["advies"] or "").replace("\\$", "$"),
                "fout": h["fout"] or "",
                "aantal_verbeterpogingen": len(h["pogingen"]),
                "duur_sec": h.get("duur"),
            }
            for h in historie
        ])
        c1.download_button(
            "⬇️ Download vragen en antwoorden (CSV)",
            export.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="askai_vragen_en_antwoorden.csv",
            mime="text/csv",
            width="stretch",
        )
        if c2.button("🗑️ Gesprek wissen", width="stretch"):
            st.session_state["askai_historie"] = []
            st.rerun()


# ===========================================================================
# PAGINA 3: VALIDATIE (miniopdracht 5)
# ===========================================================================
elif pagina == PAGINA_VALIDATIE:
    st.subheader("✅ Validatie: AskAI vs. traditionele analyse")
    st.markdown(
        "Hoe betrouwbaar is AskAI? Hier krijgt AskAI een vaste set testvragen "
        "waarvan het juiste antwoord al bekend is. Dat juiste antwoord is berekend "
        "met **handgeschreven pandas-code** (de traditionele methode, zoals een "
        "analist het zelf zou doen). De app vergelijkt beide uitkomsten automatisch."
    )
    st.caption(
        "De validatie gebruikt altijd de hele dataset (de filters in de zijbalk "
        "tellen hier niet mee). Er wordt geen advies geschreven, alleen de code en "
        "het getal worden getest."
    )

    referenties = referentie_uitkomsten()

    testset = st.radio(
        "Testset",
        list(TESTSETS),
        horizontal=True,
        key="testset",
        help=(
            "Basis: eenvoudige kerncijfers. Moeilijk: vragen waarbij AskAI meerdere "
            "stappen moet combineren, zoals groeien tussen jaren, drempels, marges "
            "met de kostprijs en percentages per groep."
        ),
    )
    VALIDATIE_VRAGEN_GEKOZEN = TESTSETS[testset]

    with st.expander("Bekijk de testvragen en de traditionele pandas-analyse (bekende uitkomsten)"):
        st.dataframe(
            pd.DataFrame([
                {
                    "nr": t["id"],
                    "testvraag": t["vraag"],
                    "bekende uitkomst": kort_resultaat(referenties[t["id"]]["waarde"]),
                    "handgeschreven pandas-code": t["referentie"],
                }
                for t in VALIDATIE_VRAGEN_GEKOZEN
            ]),
            hide_index=True,
            width="stretch",
        )

    o1, o2 = st.columns(2)
    met_definities = o1.toggle(
        "Bedrijfsdefinities meegeven aan de AI",
        value=True,
        help=(
            "Aan: de AI krijgt uitleg over wat Coffeelab bedoelt met omzet, orders, "
            "openstaand en gemiddelde orderwaarde. Uit: de AI krijgt alleen de "
            "kolomnamen en moet dit zelf raden. Zo kun je testen hoeveel context uitmaakt."
        ),
    )
    herhalingen = o2.number_input(
        "Aantal keer elke vraag stellen",
        min_value=1, max_value=3, value=1, step=1,
        help="Meer dan 1 laat zien of AskAI steeds hetzelfde antwoord geeft (consistentie).",
    )
    st.caption(
        f"Dit kost {len(VALIDATIE_VRAGEN_GEKOZEN) * herhalingen} tot "
        f"{len(VALIDATIE_VRAGEN_GEKOZEN) * herhalingen * 2} API-calls en duurt een paar minuten."
    )

    if st.button("▶️ Start validatietest"):
        client = maak_groq_client()
        if client:
            schema = maak_schema()
            resultaten = []
            totaal = len(VALIDATIE_VRAGEN_GEKOZEN) * herhalingen
            voortgang = st.progress(0.0, text="Validatie gestart...")
            stap = 0
            for ronde in range(1, herhalingen + 1):
                for t in VALIDATIE_VRAGEN_GEKOZEN:
                    stap += 1
                    voortgang.progress(
                        (stap - 1) / totaal,
                        text=f"Ronde {ronde}, vraag {t['id']}: {t['vraag']}",
                    )
                    ref = referenties[t["id"]]
                    ai = beantwoord_vraag(
                        client, t["vraag"], df, schema, "hele dataset",
                        met_definities=met_definities, met_advies=False,
                    )
                    if ai["fout"]:
                        status, toelichting = "💥 Fout", ai["fout"]
                    elif ai["geen_antwoord"]:
                        status, toelichting = "💥 Fout", "AI gaf aan de vraag niet te kunnen beantwoorden"
                    else:
                        status, toelichting = vergelijk(ai["resultaat"], ref["waarde"])
                    resultaten.append({
                        "ronde": ronde,
                        "nr": t["id"],
                        "testvraag": t["vraag"],
                        "status": status,
                        "toelichting": toelichting,
                        "AskAI-uitkomst": kort_resultaat(ai["resultaat"]),
                        "bekende uitkomst": kort_resultaat(ref["waarde"]),
                        "AskAI-code": ai["code"],
                        "pandas-code (traditioneel)": t["referentie"],
                        "verbeterpogingen": len(ai["pogingen"]),
                        "tijd AskAI (s)": ai["duur"],
                        "tijd pandas (s)": round(ref["tijd"], 3),
                    })
            voortgang.progress(1.0, text="Validatie klaar.")
            st.session_state["validatie"] = {
                "resultaten": resultaten,
                "met_definities": met_definities,
                "herhalingen": herhalingen,
                "testset": testset,
                "tijdstip": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

    if "validatie" in st.session_state:
        v = st.session_state["validatie"]
        res_df = pd.DataFrame(v["resultaten"])
        n = len(res_df)
        juist = (res_df["status"] == "✅ Juist").sum()
        deels = (res_df["status"] == "⚠️ Deels juist").sum()
        fout = n - juist - deels

        st.divider()
        st.markdown(
            f"**Uitslag van {v['tijdstip']}** · testset {v.get('testset', 'Basis')} · bedrijfsdefinities "
            f"{'aan' if v['met_definities'] else 'uit'} · {v['herhalingen']}× per vraag"
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Juist", f"{juist} / {n}", f"{juist / n * 100:.0f}%", delta_color="off")
        m2.metric("Deels juist", f"{deels}")
        m3.metric("Onjuist / fout", f"{fout}")
        m4.metric("Gem. tijd AskAI", f"{res_df['tijd AskAI (s)'].mean():.1f} s")
        m5.metric("Gem. tijd pandas", f"{res_df['tijd pandas (s)'].mean():.3f} s")

        if v["herhalingen"] > 1:
            consistent = (
                res_df.groupby("nr")
                .agg(testvraag=("testvraag", "first"),
                     keer_juist=("status", lambda s: (s == "✅ Juist").sum()),
                     verschillende_uitkomsten=("AskAI-uitkomst", "nunique"))
                .reset_index()
            )
            consistent["consistent"] = consistent["verschillende_uitkomsten"].eq(1).map(
                {True: "ja", False: "nee"}
            )
            st.markdown("**Consistentie** (geeft AskAI bij herhaling hetzelfde antwoord?)")
            st.dataframe(consistent, hide_index=True, width="stretch")

        st.markdown("**Resultaat per testvraag**")
        st.dataframe(
            res_df[["ronde", "nr", "testvraag", "status", "AskAI-uitkomst",
                    "bekende uitkomst", "toelichting", "verbeterpogingen", "tijd AskAI (s)"]],
            hide_index=True,
            width="stretch",
        )

        with st.expander("Vergelijk de code: AskAI tegenover handgeschreven pandas"):
            for _, r in res_df.iterrows():
                st.markdown(f"**{r['nr']}. {r['testvraag']}** — {r['status']} (ronde {r['ronde']})")
                k1, k2 = st.columns(2)
                k1.caption("AskAI (door de AI geschreven)")
                k1.code(r["AskAI-code"] or "(geen code)", language="python")
                k2.caption("Traditioneel (handgeschreven)")
                k2.code(r["pandas-code (traditioneel)"], language="python")

        st.download_button(
            "⬇️ Download validatieresultaten (CSV)",
            res_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"askai_validatie_{'moeilijk' if 'Moeilijk' in v.get('testset', '') else 'basis'}_{'met' if v['met_definities'] else 'zonder'}_definities.csv",
            mime="text/csv",
        )
