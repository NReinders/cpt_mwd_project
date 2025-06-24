Deze repository bevat alle scripts en data die zijn gebruikt voor het project waarin CPT-waarden worden voorspeld op basis van MWD-data.

Gebruik van uv (aanbevolen)
Voor het eenvoudig en betrouwbaar installeren van de juiste Python-pakketten en versies adviseren wij het gebruik van uv (package manager) die automatisch een virtuele omgeving aanmaakt in de projectmap. Hiermee weet je zeker dat alle dependencies correct zijn geïnstalleerd.

Stappen:

Clone deze repository naar een aparte map op je apparaat.
Open de terminal in deze map.
Voer het volgende commando uit:
uv sync

Dit maakt automatisch een virtuele omgeving aan en installeert alle benodigde pakketten zoals gedefinieerd in pyproject.toml

Indien uv nog niet is geïnstalleerd, kan dat via:
https://docs.astral.sh/uv/getting-started/installation/



Structuur van de repository

De mapstructuur is als volgt opgebouwd:
data/
├── 1_raw/          → de oorspronkelijke bestanden zoals aangeleverd, met handmatig verwijderde metadata
├── 2_cleaned/      → opgeschoonde data
├── 3_processed/    → samengestelde en bewerkte data die klaar is voor modellering
└── 4_predictions/  → outputbestanden met modelvoorspellingen

model/
└── *.py            → 14 Python-scripts voor datavoorbewerking, modellering, evaluatie en interpretatie


Belang van het rapport

Deze repository is gebruikt ter ondersteuning van het individuele onderzoeksrapport. In het rapport worden alle keuzes, stappen en resultaten uitgebreid toegelicht. We raden aan het rapport naast de code te gebruiken, zodat helder is waarom bepaalde keuzes zijn gemaakt en hoe de scripts geïnterpreteerd moeten worden.

Overige informatie

- Alle datasets vanaf 2_cleaned zijn volledig gegenereerd vanuit de Python-scripts in de model/ map.
- Modeloutput wordt bewaard in data/4_predictions/.
