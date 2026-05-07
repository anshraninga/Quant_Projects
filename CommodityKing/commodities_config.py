"""
COMMODITIES configuration — the single source of truth for all instruments.

Every field is consumed by at least one agent or model. Do not abbreviate.
"""

COMMODITIES: dict = {

    # ══════════════════════════════════════════════════════
    # PRECIOUS METALS
    # ══════════════════════════════════════════════════════

    "gold": {
        "ticker":           "GC=F",
        "name":             "Gold",
        "unit":             "USD/troy oz",
        "category":         "precious_metals",
        "weather_relevant": False,
        "usda_relevant":    False,
        "eia_relevant":     False,
        "weather_regions":  [],
        "key_drivers": [
            "US Federal Reserve policy and real interest rates",
            "USD strength (inverse relationship)",
            "Middle East and Eastern Europe geopolitical tension",
            "Central bank gold buying (China, India, Russia)",
            "Inflation expectations",
            "Safe haven demand during equity market stress",
            "Mining supply from South Africa, Australia, Russia",
            "ETF flow data (GLD, IAU)",
        ],
        "cointegrated_with": ["silver", "platinum"],
        "shock_weights": {
            "weather":       0.0,
            "geopolitical":  0.5,
            "inventory":     0.5,
        },
    },

    "silver": {
        "ticker":           "SI=F",
        "name":             "Silver",
        "unit":             "USD/troy oz",
        "category":         "precious_metals",
        "weather_relevant": False,
        "usda_relevant":    False,
        "eia_relevant":     False,
        "weather_regions":  [],
        "key_drivers": [
            "Industrial demand (solar panels, electronics, EVs)",
            "Gold/silver ratio mean reversion",
            "Fed policy (more volatile than gold)",
            "Solar energy sector growth",
            "Mining supply from Mexico, Peru, China",
            "Investment demand and ETF flows",
        ],
        "cointegrated_with": ["gold", "platinum"],
        "shock_weights": {
            "weather":       0.0,
            "geopolitical":  0.3,
            "inventory":     0.7,
        },
    },

    "platinum": {
        "ticker":           "PL=F",
        "name":             "Platinum",
        "unit":             "USD/troy oz",
        "category":         "precious_metals",
        "weather_relevant": False,
        "usda_relevant":    False,
        "eia_relevant":     False,
        "weather_regions":  [],
        "key_drivers": [
            "Automotive catalytic converter demand",
            "South Africa mining supply (dominant producer ~70%)",
            "South Africa energy crisis (load shedding) impact",
            "Hydrogen fuel cell technology demand",
            "Palladium substitution dynamics",
            "South African rand exchange rate",
            "Mining strikes and labour disputes",
        ],
        "cointegrated_with": ["gold", "silver"],
        "shock_weights": {
            "weather":       0.1,
            "geopolitical":  0.4,
            "inventory":     0.5,
        },
    },

    # ══════════════════════════════════════════════════════
    # INDUSTRIAL METALS
    # ══════════════════════════════════════════════════════

    "copper": {
        "ticker":           "HG=F",
        "name":             "Copper",
        "unit":             "USD/lb",
        "category":         "industrial_metals",
        "weather_relevant": True,
        "usda_relevant":    False,
        "eia_relevant":     False,
        "weather_regions": [
            {
                "name":       "Atacama Desert Chile",
                "lat":        -23.5,
                "lon":        -69.2,
                "importance": "critical",
                "reason":     "Chile produces 27% of global copper",
            },
            {
                "name":       "Collahuasi Mine Chile",
                "lat":        -20.9,
                "lon":        -68.7,
                "importance": "high",
                "reason":     "One of the world's largest copper mines",
            },
            {
                "name":       "Cerro Verde Peru",
                "lat":        -16.5,
                "lon":        -71.6,
                "importance": "high",
                "reason":     "Peru is second largest copper producer",
            },
            {
                "name":       "Zambia Copperbelt",
                "lat":        -13.0,
                "lon":         28.0,
                "importance": "medium",
                "reason":     "African copper supply",
            },
        ],
        "key_drivers": [
            "China manufacturing PMI and construction activity",
            "Global EV adoption rate (copper intensive)",
            "Chile and Peru mining output and strikes",
            "Flooding or drought disrupting Chilean mines",
            "LME warehouse inventory levels",
            "Global infrastructure spending",
            "China property sector health",
            "Green energy transition demand growth",
        ],
        "cointegrated_with": [],
        "shock_weights": {
            "weather":       0.3,
            "geopolitical":  0.3,
            "inventory":     0.4,
        },
    },

    # ══════════════════════════════════════════════════════
    # AGRICULTURAL
    # ══════════════════════════════════════════════════════

    "wheat": {
        "ticker":           "ZW=F",
        "name":             "Wheat",
        "unit":             "cents/bushel",
        "category":         "agricultural",
        "weather_relevant": True,
        "usda_relevant":    True,
        "eia_relevant":     False,
        "weather_regions": [
            {
                "name":       "Kansas USA",
                "lat":         38.5,
                "lon":        -98.5,
                "importance": "critical",
                "reason":     "Top US winter wheat state",
            },
            {
                "name":       "Oklahoma USA",
                "lat":         35.5,
                "lon":        -97.5,
                "importance": "high",
                "reason":     "Major US winter wheat producer",
            },
            {
                "name":       "Odessa Ukraine",
                "lat":         46.5,
                "lon":         30.7,
                "importance": "critical",
                "reason":     "Ukraine exports ~10% of global wheat",
            },
            {
                "name":       "Krasnodar Russia",
                "lat":         45.0,
                "lon":         39.0,
                "importance": "critical",
                "reason":     "Russia is the largest wheat exporter (~20%)",
            },
            {
                "name":       "Punjab India",
                "lat":         30.9,
                "lon":         75.8,
                "importance": "high",
                "reason":     "India major producer, export policy volatile",
            },
            {
                "name":       "New South Wales Australia",
                "lat":        -32.0,
                "lon":        147.0,
                "importance": "medium",
                "reason":     "Australia major Southern Hemisphere exporter",
            },
        ],
        "key_drivers": [
            "Russia-Ukraine Black Sea grain corridor",
            "US Great Plains drought monitor (NOAA)",
            "USDA WASDE monthly supply and demand report",
            "India wheat export ban or restriction policy",
            "Global stocks-to-use ratio",
            "La Nina and El Nino impact on Southern Hemisphere",
            "EU Common Agricultural Policy",
            "Pakistan and Egypt import demand",
        ],
        "cointegrated_with": ["corn"],
        "shock_weights": {
            "weather":       0.45,
            "geopolitical":  0.35,
            "inventory":     0.20,
        },
    },

    "corn": {
        "ticker":           "ZC=F",
        "name":             "Corn",
        "unit":             "cents/bushel",
        "category":         "agricultural",
        "weather_relevant": True,
        "usda_relevant":    True,
        "eia_relevant":     False,
        "weather_regions": [
            {
                "name":       "Iowa USA",
                "lat":         42.0,
                "lon":        -93.5,
                "importance": "critical",
                "reason":     "Top US corn producing state",
            },
            {
                "name":       "Illinois USA",
                "lat":         40.0,
                "lon":        -89.0,
                "importance": "critical",
                "reason":     "Second largest US corn state",
            },
            {
                "name":       "Mato Grosso Brazil",
                "lat":        -12.5,
                "lon":        -55.5,
                "importance": "critical",
                "reason":     "Brazil is the largest corn exporter",
            },
            {
                "name":       "Buenos Aires Argentina",
                "lat":        -34.6,
                "lon":        -60.0,
                "importance": "high",
                "reason":     "Argentina is a major corn exporter",
            },
            {
                "name":       "Ukraine Dnipro",
                "lat":         48.5,
                "lon":         35.0,
                "importance": "medium",
                "reason":     "Ukraine is a significant corn exporter",
            },
        ],
        "key_drivers": [
            "US corn belt weather during June-August pollination",
            "USDA planting progress weekly reports",
            "Brazil safrinha second crop conditions",
            "US ethanol demand (40% of US corn)",
            "China corn imports and stockpile policy",
            "Feed demand from livestock sector",
            "Wheat-corn substitution in feed markets",
        ],
        "cointegrated_with": ["wheat", "soybeans"],
        "shock_weights": {
            "weather":       0.50,
            "geopolitical":  0.20,
            "inventory":     0.30,
        },
    },

    "soybeans": {
        "ticker":           "ZS=F",
        "name":             "Soybeans",
        "unit":             "cents/bushel",
        "category":         "agricultural",
        "weather_relevant": True,
        "usda_relevant":    True,
        "eia_relevant":     False,
        "weather_regions": [
            {
                "name":       "Mato Grosso Brazil",
                "lat":        -12.5,
                "lon":        -55.5,
                "importance": "critical",
                "reason":     "Brazil produces ~35% of global soybeans",
            },
            {
                "name":       "Parana Brazil",
                "lat":        -24.0,
                "lon":        -51.5,
                "importance": "high",
                "reason":     "Second largest Brazilian soy state",
            },
            {
                "name":       "Iowa USA",
                "lat":         42.0,
                "lon":        -93.5,
                "importance": "high",
                "reason":     "Top US soybean state",
            },
            {
                "name":       "Cordoba Argentina",
                "lat":        -31.4,
                "lon":        -64.2,
                "importance": "high",
                "reason":     "Argentina is the third largest global producer",
            },
            {
                "name":       "Heilongjiang China",
                "lat":         47.0,
                "lon":        128.0,
                "importance": "medium",
                "reason":     "Chinese domestic soybean production",
            },
        ],
        "key_drivers": [
            "Brazil harvest progress and weather",
            "China soybean import demand (largest importer)",
            "US-China trade relations and tariffs",
            "Argentine peso devaluation impact on exports",
            "Palm oil substitution dynamics",
            "Crush margins for soybean oil and meal",
            "African Swine Fever impact on Chinese feed demand",
        ],
        "cointegrated_with": ["corn"],
        "shock_weights": {
            "weather":       0.50,
            "geopolitical":  0.25,
            "inventory":     0.25,
        },
    },

    "sugar": {
        "ticker":           "SB=F",
        "name":             "Sugar (Raw)",
        "unit":             "cents/lb",
        "category":         "agricultural",
        "weather_relevant": True,
        "usda_relevant":    True,
        "eia_relevant":     False,
        "weather_regions": [
            {
                "name":       "Sao Paulo Brazil",
                "lat":        -22.9,
                "lon":        -47.5,
                "importance": "critical",
                "reason":     "Brazil produces ~20% of global sugar",
            },
            {
                "name":       "Uttar Pradesh India",
                "lat":         26.8,
                "lon":         80.9,
                "importance": "critical",
                "reason":     "India is the largest sugar producer",
            },
            {
                "name":       "Queensland Australia",
                "lat":        -20.0,
                "lon":        146.0,
                "importance": "medium",
                "reason":     "Australia is a major sugar exporter",
            },
            {
                "name":       "Guangdong China",
                "lat":         23.1,
                "lon":        113.3,
                "importance": "medium",
                "reason":     "Chinese domestic sugar production",
            },
        ],
        "key_drivers": [
            "Brazilian sugarcane harvest and crush rate",
            "India sugar production and export policy",
            "Brazil ethanol vs sugar allocation decisions",
            "El Nino impact on Asian and Australian production",
            "Thailand export volumes",
            "Global consumption growth in developing markets",
        ],
        "cointegrated_with": [],
        "shock_weights": {
            "weather":       0.55,
            "geopolitical":  0.20,
            "inventory":     0.25,
        },
    },

    # ══════════════════════════════════════════════════════
    # ENERGY
    # ══════════════════════════════════════════════════════

    "oil_brent": {
        "ticker":           "BZ=F",
        "name":             "Brent Crude Oil",
        "unit":             "USD/barrel",
        "category":         "energy",
        "weather_relevant": True,
        "usda_relevant":    False,
        "eia_relevant":     True,
        "weather_regions": [
            {
                "name":       "Gulf of Mexico",
                "lat":         25.0,
                "lon":        -90.0,
                "importance": "high",
                "reason":     "Hurricane season disrupts US production",
            },
            {
                "name":       "North Sea",
                "lat":         57.0,
                "lon":          2.0,
                "importance": "medium",
                "reason":     "North Sea Brent production base",
            },
        ],
        "key_drivers": [
            "OPEC+ production quota decisions",
            "US strategic petroleum reserve releases",
            "Middle East conflict and Strait of Hormuz risk",
            "EIA weekly US crude inventory report",
            "China oil demand recovery",
            "US shale production response to price",
            "Russia oil export volumes and price cap",
            "Libya and Nigeria production outages",
            "Hurricane season impact on Gulf of Mexico",
        ],
        "cointegrated_with": ["natural_gas"],
        "shock_weights": {
            "weather":       0.15,
            "geopolitical":  0.55,
            "inventory":     0.30,
        },
    },

    "natural_gas": {
        "ticker":           "NG=F",
        "name":             "Natural Gas (Henry Hub)",
        "unit":             "USD/MMBtu",
        "category":         "energy",
        "weather_relevant": True,
        "usda_relevant":    False,
        "eia_relevant":     True,
        "weather_regions": [
            {
                "name":       "Boston USA",
                "lat":         42.4,
                "lon":        -71.1,
                "importance": "critical",
                "reason":     "Northeast US heating demand hub",
            },
            {
                "name":       "Chicago USA",
                "lat":         41.9,
                "lon":        -87.6,
                "importance": "high",
                "reason":     "Midwest heating and industrial demand",
            },
            {
                "name":       "Houston USA",
                "lat":         29.8,
                "lon":        -95.4,
                "importance": "high",
                "reason":     "Gulf Coast production and LNG exports",
            },
            {
                "name":       "Rotterdam Netherlands",
                "lat":         51.9,
                "lon":          4.5,
                "importance": "high",
                "reason":     "European LNG import hub",
            },
            {
                "name":       "Tokyo Japan",
                "lat":         35.7,
                "lon":        139.7,
                "importance": "medium",
                "reason":     "Asian LNG demand centre",
            },
        ],
        "key_drivers": [
            "Winter temperature forecasts for US Northeast",
            "EIA weekly natural gas storage report",
            "LNG export terminal capacity and utilisation",
            "Europe gas storage levels post-Russia cutoff",
            "US production from Permian and Marcellus basins",
            "Power generation demand in summer heat waves",
            "Australia LNG export disruptions",
        ],
        "cointegrated_with": ["oil_brent"],
        "shock_weights": {
            "weather":       0.45,
            "geopolitical":  0.25,
            "inventory":     0.30,
        },
    },
}

# ── Derived helpers ────────────────────────────────────────────────────────────

VALID_SYMBOLS: frozenset = frozenset(COMMODITIES.keys())

SYMBOLS_BY_CATEGORY: dict[str, list[str]] = {}
for _sym, _cfg in COMMODITIES.items():
    SYMBOLS_BY_CATEGORY.setdefault(_cfg["category"], []).append(_sym)

WEATHER_RELEVANT:  list[str] = [s for s, c in COMMODITIES.items() if c["weather_relevant"]]
USDA_RELEVANT:     list[str] = [s for s, c in COMMODITIES.items() if c["usda_relevant"]]
EIA_RELEVANT:      list[str] = [s for s, c in COMMODITIES.items() if c["eia_relevant"]]
