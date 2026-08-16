"""
Generate player_aliases.json — league-wise structure.
  { "global": {...},  "ipl": {...}, "t20blast": {...}, ... }
Run once: python generate_aliases.py
Re-run anytime you add a new CSV dataset.
"""
import pandas as pd, os, json

LEAGUE_CSVS = {
    "ipl":         "data/all_matches_ipl.csv",
    "t20blast":    "data/all_matches_t20blast.csv",
    "bbl":         "data/all_matches_bbl.csv",
    "cpl":         "data/all_matches_cpl.csv",
    "bpl":         "data/all_matches_bpl.csv",
    "lpl":         "data/all_matches_LPL.csv",
    "npl":         "data/all_matches_NPL.csv",
    "ilt":         "data/all_matches_ilt.csv",
    "mlc":         "data/all_matches_MLC.csv",
    "psl":         "data/all_matches_psl.csv",
    "sat20":       "data/all_matches_sat20.csv",
    "the100":      "data/all_matches_the100.csv",
    "the100women": "data/all_matches_the100women.csv",
    "wbb":         "data/all_matches_wbb.csv",
    "wpl":         "data/all_matches_wpl.csv",
}

league_players = {}
all_names = set()
for league, fpath in LEAGUE_CSVS.items():
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue
    try:
        df = pd.read_csv(fpath, low_memory=False)
        names = set()
        for col in ("striker", "bowler"):
            if col in df.columns:
                names.update(df[col].dropna().astype(str).unique())
        league_players[league] = names
        all_names.update(names)
        print(f"  {league:15} {len(names):4} players")
    except Exception as e:
        print(f"  skip {fpath}: {e}")

print(f"\nTotal unique players: {len(all_names)}")

MASTER = {
    "virat": "V Kohli", "virat kohli": "V Kohli",
    "rohit": "RG Sharma", "rohit sharma": "RG Sharma",
    "ms dhoni": "MS Dhoni", "mahi": "MS Dhoni", "mahendra dhoni": "MS Dhoni",
    "hardik": "HH Pandya", "hardik pandya": "HH Pandya",
    "krunal": "KH Pandya", "krunal pandya": "KH Pandya",
    "jasprit": "JJ Bumrah", "jasprit bumrah": "JJ Bumrah",
    "suresh raina": "SK Raina", "suresh": "SK Raina",
    "shikhar": "S Dhawan", "shikhar dhawan": "S Dhawan",
    "ravindra jadeja": "RA Jadeja", "ravindra": "RA Jadeja",
    "ravichandran ashwin": "R Ashwin", "ravichandran": "R Ashwin",
    "sanju": "SV Samson", "sanju samson": "SV Samson",
    "rishabh": "RR Pant", "rishabh pant": "RR Pant",
    "shreyas": "SS Iyer", "shreyas iyer": "SS Iyer",
    "dinesh karthik": "KD Karthik",
    "ambati": "AT Rayudu", "ambati rayudu": "AT Rayudu",
    "yuzvendra chahal": "YS Chahal", "yuzvendra": "YS Chahal", "yuzi": "YS Chahal",
    "zaheer khan": "Z Khan", "zaheer": "Z Khan",
    "sachin": "SR Tendulkar", "sachin tendulkar": "SR Tendulkar",
    "sourav": "SC Ganguly", "sourav ganguly": "SC Ganguly", "dada": "SC Ganguly",
    "rahul dravid": "R Dravid", "the wall": "R Dravid",
    "gautam gambhir": "G Gambhir", "gautam": "G Gambhir",
    "yuvraj": "Yuvraj Singh", "yuvraj singh": "Yuvraj Singh", "yuvi": "Yuvraj Singh",
    "harbhajan": "Harbhajan Singh", "harbhajan singh": "Harbhajan Singh", "bhajji": "Harbhajan Singh",
    "irfan pathan": "IK Pathan", "irfan": "IK Pathan",
    "yusuf pathan": "YK Pathan", "yusuf": "YK Pathan",
    "virender sehwag": "V Sehwag", "virender": "V Sehwag",
    "ishant sharma": "I Sharma", "ishant": "I Sharma",
    "ajinkya rahane": "AM Rahane", "ajinkya": "AM Rahane",
    "bhuvneshwar kumar": "B Kumar", "bhuvneshwar": "B Kumar",
    "umesh yadav": "UT Yadav", "umesh": "UT Yadav",
    "manish pandey": "MK Pandey", "manish": "MK Pandey",
    "robin uthappa": "RV Uthappa", "robin": "RV Uthappa",
    "kl rahul": "KL Rahul", "lokesh rahul": "KL Rahul",
    "axar patel": "AR Patel", "axar": "AR Patel",
    "washington sundar": "W Sundar", "washington": "W Sundar",
    "deepak chahar": "DL Chahar",
    "shardul thakur": "ST Thakur", "shardul": "ST Thakur",
    "prithvi shaw": "PP Shaw", "prithvi": "PP Shaw",
    "mayank agarwal": "MA Agarwal", "mayank": "MA Agarwal",
    "sai sudharsan": "B Sai Sudharsan",
    "rinku singh": "Rinku Singh",
    "devdutt padikkal": "D Padikkal", "devdutt": "D Padikkal",
    "arshdeep singh": "Arshdeep Singh",
    "kuldeep yadav": "Kuldeep Yadav", "kuldeep": "Kuldeep Yadav",
    "ishan kishan": "Ishan Kishan", "ishan": "Ishan Kishan",
    "shubman gill": "Shubman Gill", "shubman": "Shubman Gill",
    "suryakumar yadav": "SA Yadav", "suryakumar": "SA Yadav", "surya": "SA Yadav",
    "abhishek sharma": "Abhishek Sharma",
    "nitish rana": "N Rana", "nitish": "N Rana",
    "wriddhiman saha": "WP Saha",
    "joe root": "JE Root", "joe": "JE Root",
    "ben stokes": "BA Stokes", "ben": "BA Stokes",
    "jonny bairstow": "JM Bairstow", "jonny": "JM Bairstow",
    "jos buttler": "JC Buttler", "jos": "JC Buttler",
    "eoin morgan": "EJG Morgan", "eoin": "EJG Morgan",
    "jason roy": "JJ Roy",
    "alex hales": "AD Hales",
    "dawid malan": "DJ Malan", "dawid": "DJ Malan",
    "liam livingstone": "LS Livingstone", "liam": "LS Livingstone",
    "sam curran": "SM Curran",
    "tom curran": "TK Curran",
    "mark wood": "MA Wood",
    "chris woakes": "CR Woakes",
    "jofra archer": "JC Archer", "jofra": "JC Archer",
    "adil rashid": "AU Rashid", "adil": "AU Rashid",
    "moeen ali": "MM Ali", "moeen": "MM Ali",
    "phil salt": "PD Salt", "phil": "PD Salt",
    "will jacks": "WG Jacks",
    "harry brook": "HD Brook", "harry": "HD Brook",
    "kevin pietersen": "KP Pietersen", "kp": "KP Pietersen",
    "andrew flintoff": "A Flintoff", "freddie": "A Flintoff",
    "david warner": "DA Warner", "davey": "DA Warner",
    "aaron finch": "AJ Finch", "aaron": "AJ Finch",
    "steven smith": "SPD Smith", "steve smith": "SPD Smith",
    "glenn maxwell": "GJ Maxwell", "glenn": "GJ Maxwell", "maxi": "GJ Maxwell",
    "adam gilchrist": "AC Gilchrist", "gilly": "AC Gilchrist",
    "ricky ponting": "RT Ponting", "punter": "RT Ponting",
    "matthew hayden": "ML Hayden", "haydos": "ML Hayden",
    "brett lee": "B Lee",
    "mitchell starc": "MA Starc", "mitch starc": "MA Starc",
    "pat cummins": "PJ Cummins", "pat": "PJ Cummins",
    "josh hazlewood": "JR Hazlewood",
    "marcus stoinis": "MP Stoinis", "marcus": "MP Stoinis",
    "tim david": "Tim David",
    "travis head": "TM Head", "travis": "TM Head",
    "cameron green": "C Green", "cam green": "C Green",
    "mitchell marsh": "MR Marsh", "mitch marsh": "MR Marsh",
    "michael hussey": "MEK Hussey", "mr cricket": "MEK Hussey",
    "andrew symonds": "A Symonds", "roy": "A Symonds",
    "shane watson": "SR Watson",
    "chris lynn": "CA Lynn",
    "marnus labuschagne": "M Labuschagne", "marnus": "M Labuschagne",
    "kane williamson": "KS Williamson", "kane": "KS Williamson",
    "brendon mccullum": "BB McCullum", "baz": "BB McCullum",
    "ross taylor": "LRPL Taylor", "ross": "LRPL Taylor",
    "martin guptill": "MJ Guptill", "martin": "MJ Guptill",
    "trent boult": "TA Boult", "trent": "TA Boult",
    "devon conway": "DP Conway", "devon": "DP Conway",
    "finn allen": "F Allen", "finn": "F Allen",
    "james neesham": "JDS Neesham", "jimmy neesham": "JDS Neesham",
    "chris gayle": "CH Gayle", "universe boss": "CH Gayle",
    "kieron pollard": "KA Pollard", "kieron": "KA Pollard",
    "andre russell": "AD Russell", "dre russ": "AD Russell",
    "dwayne bravo": "DJ Bravo", "dj bravo": "DJ Bravo",
    "sunil narine": "SP Narine", "sunil": "SP Narine",
    "nicholas pooran": "N Pooran", "nicholas": "N Pooran",
    "shimron hetmyer": "SO Hetmyer", "shimron": "SO Hetmyer",
    "rovman powell": "R Powell", "rovman": "R Powell",
    "shai hope": "SD Hope", "shai": "SD Hope",
    "alzarri joseph": "AS Joseph", "alzarri": "AS Joseph",
    "ab de villiers": "AB de Villiers", "mr 360": "AB de Villiers", "ab": "AB de Villiers",
    "faf du plessis": "F du Plessis", "faf": "F du Plessis",
    "quinton de kock": "Q de Kock", "quinton": "Q de Kock",
    "david miller": "DA Miller", "killer miller": "DA Miller",
    "dale steyn": "DW Steyn", "dale": "DW Steyn",
    "kagiso rabada": "K Rabada", "kagiso": "K Rabada",
    "aiden markram": "AK Markram", "aiden": "AK Markram",
    "heinrich klaasen": "HE Klaasen", "heinrich": "HE Klaasen",
    "anrich nortje": "A Nortje", "anrich": "A Nortje",
    "hashim amla": "HM Amla", "hashim": "HM Amla",
    "imran tahir": "Imran Tahir",
    "shahid afridi": "Shahid Afridi", "boom boom": "Shahid Afridi",
    "shoaib akhtar": "Shoaib Akhtar", "rawalpindi express": "Shoaib Akhtar",
    "mohammad hafeez": "Mohammad Hafeez", "the professor": "Mohammad Hafeez",
    "babar azam": "Babar Azam", "babar": "Babar Azam",
    "mohammad rizwan": "Mohammad Rizwan", "rizwan": "Mohammad Rizwan",
    "fakhar zaman": "Fakhar Zaman", "fakhar": "Fakhar Zaman",
    "shadab khan": "Shadab Khan", "shadab": "Shadab Khan",
    "shaheen afridi": "Shaheen Shah Afridi", "shaheen": "Shaheen Shah Afridi",
    "naseem shah": "Naseem Shah", "naseem": "Naseem Shah",
    "haris rauf": "Haris Rauf", "haris": "Haris Rauf",
    "shoaib malik": "Shoaib Malik",
    "lasith malinga": "SL Malinga", "malinga": "SL Malinga", "slinga": "SL Malinga",
    "kumar sangakkara": "KC Sangakkara", "sangakkara": "KC Sangakkara",
    "mahela jayawardena": "DPMD Jayawardena", "mahela": "DPMD Jayawardena",
    "tillakaratne dilshan": "TM Dilshan", "dilshan": "TM Dilshan",
    "angelo mathews": "AD Mathews", "angelo": "AD Mathews",
    "kusal mendis": "BKG Mendis", "kusal": "BKG Mendis",
    "sanath jayasuriya": "ST Jayasuriya", "jayasuriya": "ST Jayasuriya",
    "muttiah muralitharan": "M Muralitharan", "murali": "M Muralitharan",
    "rashid khan": "Rashid Khan", "rashid": "Rashid Khan",
    "mohammad nabi": "Mohammad Nabi", "nabi": "Mohammad Nabi",
    "mujeeb ur rahman": "Mujeeb Ur Rahman", "mujeeb": "Mujeeb Ur Rahman",
    "shakib al hasan": "Shakib Al Hasan", "shakib": "Shakib Al Hasan",
    "mushfiqur rahim": "Mushfiqur Rahim", "mushfiqur": "Mushfiqur Rahim",
    "tamim iqbal": "Tamim Iqbal", "tamim": "Tamim Iqbal",
    "mustafizur rahman": "Mustafizur Rahman", "fizz": "Mustafizur Rahman",
    "sandeep lamichhane": "Sandeep Lamichhane", "sandeep": "Sandeep Lamichhane",
    "heather knight": "HC Knight", "heather": "HC Knight",
    "nat sciver-brunt": "NR Sciver-Brunt", "nat sciver": "NR Sciver-Brunt",
    "smriti mandhana": "S Mandhana", "smriti": "S Mandhana",
    "harmanpreet kaur": "H Kaur", "harmanpreet": "H Kaur",
    "deepti sharma": "Deepti Sharma",
    "shafali verma": "Shafali Verma", "shafali": "Shafali Verma",
    "alyssa healy": "AJ Healy", "alyssa": "AJ Healy",
    "meg lanning": "MM Lanning", "meg": "MM Lanning",
    "ellyse perry": "EA Perry", "ellyse": "EA Perry",

    # ── IPL 2026 new players ──────────────────────────────────────────────
    "mayank yadav": "MP Yadav",                                # Mayank Prabhu Yadav (LSG)
    "venkatesh iyer": "VR Iyer", "venkatesh": "VR Iyer",      # Venkatesh Rajasekaran Iyer (RCB)
    "vaibhav arora": "VG Arora",                               # Vaibhav Gopal Arora (KKR) – "vaibhav" alone is ambiguous
    "rahul chahar": "RD Chahar",                               # Rahul Desraj Chahar (MI)
    "shahrukh khan": "M Shahrukh Khan", "shahrukh": "M Shahrukh Khan",  # Masood Shahrukh Khan (PBKS)
    "prabhsimran singh": "P Simran Singh", "prabhsimran": "P Simran Singh",  # PBKS WK
    "anukul roy": "AS Roy", "anukul": "AS Roy",                # Anukul Sudhakar Roy (KKR)
    "arshin kulkarni": "AA Kulkarni", "arshin": "AA Kulkarni", # Arshin Kulkarni (LSG)
    "tripurana vijay": "T Vijay", "tripurana": "T Vijay",      # DC player from Andhra
    "salil arora": "S Arora",                                  # S Arora (Punjab/SRH)
    "rishab ghosh": "RS Ghosh", "rishab": "RS Ghosh",          # Rishab Suresh Ghosh (SRH WK)
    "eshan malinga": "E Malinga",                              # Sri Lankan (not Lasith)

    # ── MLC / international 2025 ─────────────────────────────────────────
    "rachin ravindra": "R Ravindra", "rachin": "R Ravindra",   # Rachin Ravindra (NZ – MLC)

    # ── CPL 2025 ─────────────────────────────────────────────────────────
    "joshua da silva": "J Da Silva", "da silva": "J Da Silva", # WI wicketkeeper
    "odean smith": "OF Smith",                                  # OF Smith (Jamaica / CPL)
    "dario bravo": "DM Bravo",                                  # DM Bravo (T&T)

    # ── BBL 2025/26 ───────────────────────────────────────────────────────
    "chris green": "CJ Green",                                  # Queensland spinner
}

global_section = {}
for k, v in MASTER.items():
    if v in all_names:
        global_section[k] = v
    else:
        match = next((n for n in all_names if n.lower() == v.lower()), None)
        if match:
            global_section[k] = match

league_sections = {lg: {} for lg in LEAGUE_CSVS}
for league, names in league_players.items():
    last_to_player = {}
    for n in names:
        parts = n.split()
        if len(parts) >= 2:
            last = parts[-1].lower()
            last_to_player[last] = None if last in last_to_player else n
    for last, player in last_to_player.items():
        if player is None: continue
        if last in global_section: continue
        if last == player.lower(): continue
        league_sections[league][last] = player

output = {
    "_note": "League-wise aliases. 'global' works for ALL leagues. Each league section covers that league only. Keys are lowercase. Re-run generate_aliases.py after adding new CSVs.",
    "global": dict(sorted(global_section.items())),
}
for lg in LEAGUE_CSVS:
    if lg in league_sections and league_sections[lg]:
        output[lg] = dict(sorted(league_sections[lg].items()))

with open("data/player_aliases.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\nWritten: data/player_aliases.json")
print(f"  global   : {len(global_section)} entries")
for lg, sec in league_sections.items():
    if sec:
        print(f"  {lg:15}: {len(sec)} entries")
total = len(global_section) + sum(len(v) for v in league_sections.values())
print(f"  TOTAL    : {total} aliases")
