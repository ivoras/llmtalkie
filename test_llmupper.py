#!/usr/bin/env python3

from llmtalkie import LLMConfig, LLMMap

# This test uppercases a list of words using the LLM - arguably the most inefficient
# way to do it - but it's a test :)

input_words = ["emanation's", "hilltop", "analgesia", "Carpentaria", "preshrunken", "Sumgait", "luridly", "horseplayer", "Strathclyde", "unjustifiable", "blastocoel", "sheering", "cheated", "smartened", "Ghanian's", "relaunch's", "Seaside", "overtaxation", "advertiser's", "mixology", "datasets", "cestode", "reenforces", "Kurtis's", "encoding", "Swaziland's", "replenisher", "magnetospheric", "gigabytes", "cosecants", "transonic", "gauffers", "eights", "circuit's", "repellingly", "Annetta", "ionospheres", "oozy", "ligature's", "dike's", "hotspots", "passionflower's", "highflier", "agio", "deposit's", "eolipile", "electric", "veinlet", "mummy's", "Bonner", "storyline", "Tanya", "cupcake", "Weber's", "legman", "trainable", "spectrographs", "schlepps", "uncharismatic", "pedophiles", "OAS", "reliabilities", "portage", "farthings", "sidesteps", "arriving", "liberalized", "bypass's", "disgracefulness's", "Robbins's", "dale", "despicablenesses", "hopped", "broadly", "northeastern", "formlessly", "Leighton", "beauties", "Essex's", "abuttals", "ascents", "statuses", "horselaughs", "geneticists", "haunt", "ablaut", "postmodernism", "melilots", "trochee's", "flopped", "whiling", "Sandinista", "swipe", "cutlers", "regretfulness", "temperateness's", "premieres", "waggle", "oenophiles", "societies", "parley", "homophobia", "csc", "crunchy", "thriver", "hussy", "braggart", "cocainized", "highlight's", "distressed", "cybercaf\u00e9s", "mudcat", "clarify", "becks", "barbell's", "Schulz's", "divesting", "graphology", "unchristened", "Tokharian", "interlanguage", "Falklands", "posture", "misanthropist", "tricrotic", "Balaton's", "Ariosto's", "Cr", "xor", "takeaway", "picklock", "doubts", "Scotchman's", "Algeria", "striae", "royal's", "speck's", "dingdong", "rhombohedron", "spiffily", "protestor", "hydrangeas", "integrates", "airtime", "Romagna", "snatch", "Brahmaputra's", "jinrikisha", "dolphins", "crosscurrent", "Marathas", "m\u00e9salliances", "yeshivoth", "universities", "MacArthur's", "gourd's", "pseudohermaphroditism", "petered", "amplifying", "FUD", "prizefighter", "launces", "scanter", "trademarks", "Nader's", "irresoluble", "hellgrammite", "Armageddon", "lethargy", "arrowroot", "Milagros", "converge", "archaeopteryxes", "husker", "Oder's", "delaying", "Quarter", "mysteriousness's", "resected", "satellited", "synthesize", "crosspieces", "uniformity's", "Madagascan", "stiffest", "Timbuktu's", "waisted", "abnegation's", "palliation", "Gail", "obstructing", "placidness", "gagged", "Callicrates", "acquiescent", "chattered", "tropeolin", "hinter's", "malamute's", "amalgamation's", "calumniator", "scenarists", "Roman's", "sprue", "repelled", "grizzling", "scrutineer", "autointoxication", "encouragements", "mandolin's", "affirmations", "taunters", "studying", "splatterpunk", "fatback", "moonlights", "calamitousness", "whoopla", "deportee's", "Levine's", "numerations", "conferrable", "Galvani's", "Criseyde", "contraindicated", "cobnut", "ogle's", "spill", "fickly", "Minimalist", "ritualistically", "constructing", "smokehouse", "helve's", "sightless", "Hangchou", "guessing", "supervision's", "oriel", "callow", "heaume", "Rabia", "paratyphoid's", "unusually", "alternated", "mockingbird's", "forefronts", "doodlebug's", "Avior's", "Messeigneurs", "Toronto", "ford", "acre", "Sucre", "verandas", "tattletale's", "HI", "voyage", "Lamont's", "pursuer's", "rendering's", "Sudoku's", "Studebaker", "yacking", "headreach", "Caesarea", "sedum", "chloride", "gentleness", "grenadiers", "Xamarin", "groundnut", "Horus", "Pontchartrain's", "Lindsey's", "anatomize", "begun", "brigs", "proffered", "sinister", "loggerhead's", "mirthful", "strafes", "money's", "inflammations", "protestation's", "damascenes", "expensively", "demoralizer", "doodle", "actress", "alpenstock", "exhibit", "shank", "Brahma", "Mandan", "nubilous", "basketwork", "theriomorphic", "delighting", "KKtP", "buggered", "Durrell", "entity", "Anselm's", "Faustian", "Raj", "bucketing", "scarabaeuses", "musing's", "disrupter", "nondelivery", "slacked", "leopardesses", "tower's", "aborigine's", "disdaining", "preregistering", "villeinage's", "unturned", "hypnotist", "spans", "Hertha", "indefatigabilities", "Matlock's", "mullet's", "recon", "Markov", "curator", "circularize", "staved", "hadji", "phenix", "Amman", "renegaded", "McNaughton's", "Lorre's", "monarchic", "mousiness's", "diode's", "Albania", "Highlander", "NBA's", "queued", "Walgreens", "Guinean", "endamage", "antipoverty", "pedigrees", "nominator's", "Cnidus", "eucalyptuses", "heteromorphic", "outlanders", "rheostat", "Christendom", "womenfolk's", "youthfully", "proscriptions", "spavin's", "Stanford's", "convict's", "backbench", "stodgily", "redintegrative", "sabotage's", "pawing", "dextralities", "excision's", "fireplaces", "Brasil's", "pennoned", "Angara", "anchorman's", "Rowling", "nonbasic", "quelling", "Goode's", "cortisone", "Marquez", "reproduction's", "disciple's", "titian's", "megaphoned", "barium's", "brunette's", "captious", "Valera's", "glutinosity", "meander", "superglue", "ashlars", "rah", "Lordship", "broiling", "yards", "Lysander", "disfavored", "tog's", "weaseled", "repast's", "armatures", "albacore's", "Trekkie", "regalement's", "supportive", "cachexia", "Sverdlovsk", "schmaltzier", "frith", "blurriness's", "allelic", "Dexamyl", "Hieronymus", "kilts", "coke", "buckler", "shortfall's", "metalworking", "simulating", "hideous", "draconian", "loused", "astuter", "rhapsody", "bearberries", "slackenings", "chignon's", "efficaciousnesses", "notable's", "phalanxes", "astronaut", "gtd", "greaser", "simoniacal", "scends", "weathervane", "agribusinesses", "bechanced", "garlic", "Sukkoth", "Haman's", "heats", "fuze", "careening", "birthstones", "ejects", "Vedda", "zedoary", "muscly", "cupule", "Popocatepetl's", "scratchily", "tanker's", "meshugga", "silva", "pentameters", "calla's", "automaton", "brave's", "hammock's", "Baidu", "Vicki", "hydrops", "warehouseman", "apoplectically", "dined", "nah", "cringe", "Rubik", "trough's", "luridness's", "melanic", "bridal", "hotheaded", "gestation", "palatial", "assertiveness", "coelacanth", "Louisianan's", "scrawniness's", "Willemstad", "Nathaniel", "maladminister", "surfperch", "centurial", "hundredth", "bipartisanship's", "centimes", "stammered", "keno's", "glasswort", "girasol", "murderer's", "underwrites", "hyperactivity's", "profusion's", "checked", "unteaching", "eclogue's", "Balboa", "tacked", "dendritic", "evasively", "speedup's", "snooping", "shriek's", "letterer's", "Cwmbran", "endarch", "Boer's", "aristocrat", "pustules", "tadpole's", "inunction", "chatter's", "tatterdemalions", "shrievalty", "solferino", "occasional", "prizefighters", "conglutinates", "yodeled", "Antoine", "Farouk", "teachers", "trains", "duplicitous", "alforja", "endlong", "midlines", "renovators", "estimate", "tenrecs", "formidabilities", "equivalent", "execrable", "trendy's", "cystectomy", "gossipping", "newsgroup's", "bavardage", "Wimsey's", "retitling", "geekier", "moonstone's", "warmups", "zestfully", "Brzezinski", "Lulu's", "Dorothea's", "Sedgemoor", "depict", "milliard", "besmearing", "beguiler's", "borderland", "restructuring's", "psychopathologic", "lizards", "pear", "containers", "dame's", "inciter", "Fonda", "podiatrists", "Nasik", "geologically", "paining", "floodgate's", "apparelled", "buckboard's", "interchangeable", "conglomeration", "asthmatic's", "photochemically", "overexpansion", "heliacal", "Wisdom's", "publican's", "overwintering", "flackery", "Orpingtons", "voiceful", "Volstead's", "microprogram", "grubber", "flanges", "jinrickshas", "dumbest", "Issac", "necroscopy", "uncooperatively", "stumper", "creation", "hydrating", "prentices", "syncopates", "harridans", "deliveryman's", "aridity", "ferule", "shiftiest", "goosy", "Gothically", "Karttikeya", "epigeal", "sebum", "conscious", "snowblowers", "hygiene's", "clonus", "urbanely", "calabooses", "Neptune", "Glenna's", "carnotite", "slip", "pangenesis", "archipelago", "blockier", "Waldo's", "bkcy", "Rydberg's", "exorcise", "diminutive", "earaches", "salaciousness's", "hops", "smuttiness's", "widowing", "regularize", "girlie", "Barnes's", "zirconias", "corduroys", "poison's", "constructors", "thinners", "sowens", "meritoriousness", "celestites", "ammoniacal", "methanol", "hosiery", "subtending", "subvocal", "refineries", "edging's", "clouded", "munching", "undersecretary", "acatalectic", "Fahrenheit's", "baronetage", "Wentworth", "act's", "stonewalled", "Phaedra's", "stroboscopes", "lithopone", "Ashmolean's", "McGee's", "trampers", "equalitarians", "disafforesting", "bibliog", "integral's", "dazzling", "Kay's", "protesters", "auctions", "amortization", "puttee", "analytics", "predestining", "pilgrimage", "walkabout", "ungotten", "sickbed", "actuarial", "Lab", "unimproved", "assignors", "triparted", "hullabaloo", "schoolboys", "uncluttered", "return", "woodcraft", "aggrades", "Apia", "spoiling", "flagrancy", "receptionist", "inception's", "overdecorates", "jeep's", "kayak", "fashionably", "splashes", "Amalia's", "tambours", "appose", "microparasite", "mementoes", "Nips", "Aboriginals", "microaggression", "conceives", "orthopaedists", "Decker's", "runway's", "osseous", "prestidigitation's", "crosshatched", "outsourced", "unsurfaced", "somniferous", "nape's", "handiwork's", "Heidi's", "overture", "escorted", "jimmied", "ambiguously", "darlings", "calculation's", "occidents", "chinquapins", "MacBride", "tenner", "confidant's", "zeros", "resiniferous", "tutorship's", "apex's", "chiseler", "infectiously", "festoon's", "recovered", "gryphons", "octennial", "novelizations", "arabesque", "sallied", "ingraining", "zingers", "stereography", "Jidda", "ensconcing", "densities", "interposing", "paramilitaries", "resettling", "molestations", "Adm", "lepidopterists", "agings", "mazy", "chains", "graving", "penetrates", "retirement's", "untunes", "gallfly", "Hank's", "selectiveness", "vacantly", "gigabits", "Porter", "graticule", "nonparticipants", "unimpressive", "jaywalk", "bordure", "aqueduct", "Gurkhas", "enlarger", "chattered", "garner", "agential", "equability", "proles", "statemented", "cruddier", "scraperboard", "fiat", "nitwit", "churner", "uninviting", "galvanism", "forecasted", "Hurd's", "leashes", "flecks", "lacing", "carrack", "shinnied", "requite", "hulas", "flora", "Copenhagen", "stupendous", "optics's", "chill's", "subdivision", "trinal", "suicides", "Home's", "backrooms", "pediatrics", "valuates", "motorcade", "postscript's", "rosiness's", "Farrow's", "unvoice", "seismography's", "retaliation's", "unequally", "sociol", "heteronym", "surmounter", "codpiece's", "immunize", "singing's", "paperer's", "hammerer", "Agnes's", "fallowing", "lumberer", "ecumenical", "agrimony", "saigas", "Daubigny", "jetliner's", "ventricular", "eastwards", "kino", "geochronology", "Boreal", "responsibly", "amphoteric", "penalties", "Wikileaks", "footrope", "Timon's", "Tai's", "prisage", "Yank's", "penetralia", "womb's", "narrates", "diagnostically", "plumpness", "diking", "inexorably", "aardvark's", "pulleys", "dialysis", "headbutts", "constitutions", "married's", "Doyle", "constellated", "Kimberly", "sweepstakes", "excoriation's", "aftertastes", "inaccurately", "subbranches", "effervesces", "Scheldt", "noncompetitive", "goober's", "meiotically", "cacophonies", "basing", "volcanologist", "suburbanization", "disallowing", "Newtonian", "hashish's", "vacuuming", "Berenice", "quantification", "Antwan", "fulminous", "Moore's", "reprising", "pacers", "Limousin", "mooch", "byelaw's", "venom", "stalkings", "Cambridgeshire", "washy", "biologically", "Mirach's", "singularities", "impurities", "southward's", "poperies", "attention's", "gallstones", "affenpinscher", "Weimaraner", "fomenting", "cesta", "addax", "Kaaba", "Hasidim's", "refueled", "beechnut's", "Duse", "shovelful's", "dreariest", "Shechem", "undershirt", "photons", "flittered", "syllabicities", "germinates", "piddled", "porkier", "gaslights", "Macintosh", "superimpose", "dredges", "Christology", "inconspicuousness", "lemmings", "hater's", "foyers", "forfeiter", "junta", "barnacled", "Korea", "HOV", "houseboat", "bos'n's", "dentists", "sphagnums", "lyric", "reminders", "literarily", "jointer", "indispensables", "dishcloth's", "alfalfa", "abodes", "fadeouts", "tortured", "yogurt's", "spermatophyte", "anamneses", "overfeed", "annalist's", "easel's", "opaquing", "dexterity's", "bereaving", "slapstick", "kicking", "nonparticipating", "thirteenths", "jonquil", "amplitude", "Soudan", "tungsten", "diddlysquat", "Winston's", "Ozenfant", "barricade", "patroness", "septs", "batterers", "outsiders", "wobbliness", "extinguishable", "vocalist's", "Hanukkah", "Elena", "draughtboards", "Tiberias", "shareware", "charge's", "desegregationist", "Ingram's", "meetly", "Kurd's", "lipoid", "Casper", "criminal's", "Snyder", "tailboard", "colorfastness's", "illegal's", "headland", "Reyna's", "mutative", "hypocrite", "Argentinean", "devise", "doddered", "honey's", "reanimation", "corrugates", "redressal", "mandarinism", "hemolysin", "frustum", "juvenile's", "licorices", "kilometer", "detainees", "tireder", "Brummell", "spagyric", "Shintoism's", "posy", "inflammably", "ruggers", "Podgorica's", "antirachitic", "obligations", "him", "Marla", "Cocytus", "Izanagi", "bullock", "Boccaccio's", "forebodingly", "sensuously", "towelings", "underactivity", "emotionally", "rationalist", "weekending", "xanthic", "cracked", "prettying", "shoulder", "mousier", "ruck", "hysterically", "douse", "stative", "masseuse's", "netbook's", "Cerberus", "autonomously", "angioplasties", "jaconet", "Qiqihar", "reboots", "sociologic", "Sisters", "bewrayed", "gadded", "proc", "interfluve", "mantelet", "reauthorizes", "unreformed", "mutt's", "blessings", "trivets", "Brabant", "Lutheran's", "remediated", "sloths", "rebus's", "intactness", "advisor's", "kilt's", "airfields", "gingerliness", "cropper", "mankind's", "resolve's", "allurement's", "peasantry's", "photocathodes", "vizir", "\u00e9migr\u00e9's", "diaphoretic", "Barlow", "braggart's", "bach", "ballot's", "sorry", "cadaverous", "johnnycake's", "liquidations", "punctilio's", "withdraws", "Muhammadanisms", "reis", "meliorable", "Pemba", "Eli's", "appraisees", "snowplowing", "damask's", "arousal's", "autobahn", "seabed's", "cornetcy", "suburban", "Shankar", "equable", "dona's", "Edson", "cert", "Gunther", "gear", "hatchels", "pullback", "halftone's", "Istria", "ithyphallic", "saccule", "recension", "jamb", "gluttonize", "disunion's", "cabana", "criticized", "beetroots", "storehouse", "rapports", "creationisms", "plated", "Sheryl", "cookie's", "packages", "illegal's", "responding", "reverter", "postdoctorate", "droop", "verdigrises", "bombsite", "oxbows", "phosphorous", "lodgements", "shilled", "eide", "MRI's", "stanza", "bloodstock", "mass's", "raid", "locknuts", "encircle", "Breslau's", "stereobate", "perchance", "butterball's", "Franklyn's", "penfriend", "Onsager's", "bend", "Isabela", "penne", "should've", "foin", "bantings", "primatology", "unquestioned", "sourer", "romantically", "mandrill", "dicky's", "customhouse", "hermaphrodite", "mobile's", "Bond's", "attainable", "arsenide", "leatherworker", "Chasity's", "fanlight's", "Territory", "overstimulating", "Solis", "philanthropical", "Nukualofa", "politicoes", "pennyweight's", "Golden", "Mamore's", "garnishment's", "byzantine", "Danville's", "rupture", "munchkin", "navigates", "rainbows", "catamounts", "antifeminisms", "laminitis", "revolter", "baron's", "premises", "courgette", "photothermic", "grout", "engined", "Scotswoman", "whaps", "layovers", "strikes", "rag", "Finney", "subordinating", "psychoactive", "feudist", "Field", "cornices", "topologists", "mezzanine", "stockiness", "flycatcher", "Gloria's", "Bisitun", "despoliation", "quatrain's", "tuneable", "anchorage", "mistakes", "inelegance's", "megastar", "Caslon", "sensitometer", "constructiveness's", "saleswoman", "goalkeeping", "baritone", "doubtfulness", "reminds", "ectomorphs", "Flatt's", "defensed", "nulls", "decimals", "hijacker", "Race's", "westerly's", "conifer's", "reclothe", "acerbate", "Danube's", "subtotals", "solderer's", "midfielder", "cagiest", "retroversion", "Glencoe", "defective's", "chickpeas", "stingrays", "mayn't", "artel", "satinwoods", "dinges", "flab", "abed", "insatiety", "Sharon's", "instants", "Ostyak's", "fudges", "Lynch's", "hegemony", "paperbark", "interlocutors", "suits", "sandwiched", "Massachuset", "epigeous", "illimitably", "spunk's", "sadists", "dedicator", "conspirators", "Sorb", "delightedly", "denaturalized", "vow's", "central", "garpike", "pissoirs", "items", "dachshund's", "refit", "collection", "misspend", "tithing", "lithometeor", "halleluiah", "Sextans", "lightweights", "Handy's", "Valerian's", "agminate", "Oxycontin's", "capt", "immateriality", "Ea", "accident", "psychotically", "foresightedly", "Tegucigalpa's", "entailing", "kinkier", "skoal's", "ensiling", "pygmean", "tropisms", "alpine", "blastogenesis", "incr", "dragonroot", "goldener", "toff", "UHF's", "bloodfin", "entraps", "I'll", "eyelash", "regolith", "superstates", "snug's", "crupper", "abl", "vindictively", "Dmitri's", "Akiva", "dozen's", "Instamatic", "Katie's", "fatherliness", "sterilization", "imparted", "ICU", "excels", "springlike", "corv\u00e9e", "mantelets", "nonorganic", "psid", "outlives", "moralization", "Bridalveil's", "allaying", "aciniform", "ageism", "officialdom", "philharmonics", "erupted", "glutting", "trivially", "gnu's", "dampeners", "shelters", "Cassidy's", "tanked", "solidus", "Ijssel", "guzzlers", "praise's", "interloped", "guffaws", "crematoria", "maid's", "incest's", "unesthetic", "unidentifiable", "shakeups", "tetrachord", "tooth", "remake's", "puerperal", "baldachino", "antivenin's", "amoebae", "raindrop's", "trillium's", "valuable's", "possibles", "roundelay", "collators", "doubler", "hits", "supporting", "meadowsweet", "Borneo's", "tabs", "hallucinator", "stupefies", "factotum", "hobnailing", "insensitive", "malfunction", "drowses", "Montpellier", "kitchenet", "shouters", "exorable", "footage's", "indisciplines", "buffers", "crayon's", "roadster", "phonics", "psychopathy's", "fanaticism's", "kingbolt", "Miguel's", "Cowell", "schoolman", "NIMBY", "winch's", "unsolder", "recuse", "Wartburg", "steeplechasers", "raggedness's", "toreador", "crabby", "reasons", "paracasein", "peatier", "salt", "sponging", "snub's", "diminutions", "grannies", "Nebo's", "blabbed", "doubt", "LXX's", "Exchequer", "butterball", "rugrat's", "moorfowl", "Magsaysay's", "casino", "paint", "puparium", "Chinaman", "steeliness", "orthoscope", "embus", "cottager's", "rummy", "anthracosilicosis", "ossification", "legionnaire's", "homosporous", "festiveness's", "discrete", "sphinx", "equivocators", "excitations", "lappet", "thornily", "binocular", "banking's", "watcher's", "anglicize", "Packard", "cored", "Gnosticisms", "wittol", "Salvadoreans", "scend", "pyxie", "meshworks", "guvs", "mulligrubs", "sourer", "confiscable", "Voldemort's", "jet's", "giber", "collusively", "drawees", "schoolchildren", "priorship", "uninstall", "unseen's", "configurative", "coequal's", "pilseners", "hemolysis", "premenstrually", "anything's", "arcading", "manducated", "cooker's", "thrush's", "domaine", "unclasp", "heterotrophic", "placekickers", "omentums", "Tyson's", "fairytale", "ticals", "overinfluential", "mockeries", "carbonates", "overturns", "shithead's", "arum", "showboating", "Father's", "gastrula", "Falange", "frisking", "ingrowing", "grottos", "orchestrate", "bioclimatologies", "striving", "cavicorn", "labellum", "snappishly", "simonizes", "snouts", "ceremonialism", "tasting's", "reenforce", "khoum", "robalo", "ideogrammatic", "dele", "nominee", "precalculation", "conservancy", "Janelle's", "fundamentally", "uncommonness", "Zubenelgenubi", "pities", "romancing", "chemiluminescent", "vulgarities", "foiled", "summerhouses", "littered", "backhanding", "Englishmen", "subvisible", "DynamoDB's", "coverings", "damply", "Sylvester", "Seward", "atonalist", "tenacious", "Ptah", "gad", "Canaan", "lolcats", "woollies", "gasped", "ecstasies", "Kirsten", "fustily", "slatings", "suckers", "gerunds", "Ryukyuan", "overdecorates", "evaded", "thermodynamically", "Bingham's", "envelopers", "astrogation", "newer", "Isolde", "twin's", "sambar", "Frenchwoman's", "seagirt", "lording", "terebene", "spangly", "fecundation", "crooners", "vendetta", "refection's", "curtly", "Franklin's", "lambkins", "mildness", "Mycenaean's", "sacker's", "undreamt", "Dunant's", "bunghole's", "ultraconservative's", "yen", "pleonastic", "revenging", "hyoids", "mustang's", "chipolatas", "sanctum's", "maturely", "unconventionality", "Barnett's", "Basilicata's", "frogmarch", "chances", "grisaille", "undersells", "Paleolithic's", "historic", "Amazon's", "premier's", "thievish", "greeted", "alterable", "witchery's", "qualifies", "Finn", "dorty", "crepitating", "midpoint", "certitude's", "Harlan", "repairing", "Gabriel's", "bitchier", "dictatorship", "Sandy", "unary", "requisitioning", "Pillsbury's", "Amtrak", "exams", "vibraculum", "apelike", "BlackBerry's", "psychological", "contract's", "Brenda's", "awes", "Chilin", "accord", "causeuse", "sweetbriar", "toxin", "dyne", "croissant", "schizogony", "threat", "untying", "wire", "uncomfortably", "Zanuck's", "brougham", "kelly", "pastimes", "Bolivia's", "svelter", "vinosity", "Urals's", "Baghdad", "sensationalistic", "overlook", "romanticizes", "scandalmonger's", "relabels", "subjugation", "antiphrases", "morrows", "Riel's", "Ashcroft", "Winnebago's", "mudpacks", "basketry's", "Reynold's", "today", "bawdier", "AstroTurf's", "hazel", "lacunose", "hagiographer", "norepinephrine", "unleaded's", "suet", "Patty's", "burglars", "lapful", "searcher's", "Leeuwarden", "ford's", "titrated", "cudgelling", "receipts", "accustom", "outsize", "deuces", "adermin", "unloosed", "intended", "riot's", "actinic", "Minuteman's", "pink's", "Sheraton", "reave", "escapees", "lightheadednesses", "waxier", "rescues", "biology's", "Catalina's", "temporariness", "sloganeer", "smell's", "Nero", "staging", "trouveur", "decongestant's", "demonstrably", "deprave", "Scrooge's", "harpoon", "gravesides", "footballer's", "featherbed", "carrageens", "Shaka", "Arturo", "seismometer", "backhoe's", "taxpayer", "arcuation", "shoemakers", "cheekiest", "mausoleum's", "sly", "Limavady's", "bailed", "venturous", "disappear", "uninitialized", "argyles", "surveillance's", "midnights", "disembody", "corrugation's", "transplantation's", "cardiovascular", "sickie's", "convenable", "jugular's", "jaywalked", "Bandung's", "malcontented", "Russianize", "harking", "muddied", "telecommuter's", "crutches", "postcoital", "dipstick", "dynamometry", "butte", "psychrometers", "offed", "exalts", "alerted", "gorgeously", "garb", "virtuous", "Selden's", "supersecret", "JPEG", "Bobbie's", "Pristina", "gibberish", "donating", "Patsy's", "Corregidor", "exostosis", "scissions", "grog's", "wheedler", "gum's", "hemotherapy", "Ruanda's", "adverseness", "backwardation", "tenth", "Oise", "vastitude", "clubbier", "forth", "signalment", "pub", "interstratify", "Gilroy's", "depleting", "modulations", "uncorroborated", "scrapyard's", "released", "heartstrings", "dived", "hepatitis", "Kagoshima", "allseed", "voraciousness", "deskill", "integumentary", "Vedantic", "modernizers", "mousetraps", "anaphylaxis", "hippocampus", "Mariehamn", "starworts", "Marylou's", "notched", "polymerism", "disinherited", "ovule's", "mirthfulness", "phantasmic", "disagrees", "hierarchs", "stamina's", "manganese's", "gunflint", "blandness", "lumbered", "ameliorates", "breastfeeds", "tortoiseshell", "vitaceous", "midi", "insignia", "hair", "skean", "scandalized", "bargain", "shirtwaist", "finds", "fungi", "underestimation's", "slanderously", "again", "brawniest", "flatfoot's", "apothegmatic", "nitrating", "Tammie", "fragrances", "revel's", "prepares", "Scotswomen", "psychedelically", "blowfish", "inoculate", "desisted", "Puseyisms", "disrespect", "ineligibility's", "characterizations", "condole", "gustation", "spams", "howling", "DP", "perennially", "imitations", "converters", "emblazoning", "paramnesias", "aureolin", "Vickie's", "lapboards", "lipreading's", "Yorkist", "deodorizing", "elephantine", "invulnerable", "tog", "barcarolle", "joinery", "alluvion", "pawnbroking's", "WTO", "doomed", "Hilda", "foreleg's", "vacillate", "realizations", "tyke", "malady's", "unrewarded", "tureens", "begetting", "appendicitis", "patsy's", "Delhi", "depositor", "blesses", "unfixing", "fusspots", "misplacement", "homogamy", "Wii", "substructures", "Motherwell", "Gastonia", "pleaders", "euphemistically", "yowled", "crawly's", "quadruplets", "insecure", "applicable", "palatable", "Russell's", "modernistic", "lotions", "inhabitants", "possess", "inbreeding's", "Colum", "upright", "grief", "credulousness's", "affettuoso", "hesitancy", "Kimberly's", "forcedly", "rout", "Martel's", "throughput's", "circumstantiating", "resinate", "outman", "encyclopedists", "rapids", "lock's", "authorship's", "wiverns", "stoppers", "mouthparts", "carangid", "Aztecs", "Benthamism", "Hildebrand", "heterism", "pint", "Wagnerian's", "jacking", "accursedness's", "deflections", "demoralized", "Marcy's", "sandblasted", "sacculi", "eavesdrops", "Creuse", "ranis", "Perot", "McDaniel's", "medlars", "waspishness's", "drawer's", "rarity", "extreme", "coulomb's", "a", "backdrop", "blindness", "awol", "pronunciation's", "hairiness's", "snowmobiler", "booklover", "dodges", "Bermudian's", "Moresque", "Ex", "Goldsmith", "Carolyn"]

LLM_LOCAL_LLAMA32 = LLMConfig(
    url = "http://localhost:11434/api/chat",
    model_name = "llama3.2",
    system_message = "You process given words, regardless of what language they are in.",
    temperature = 0,
    options = {
        "num_ctx": 1024, # We only need a small context for this.
        "num_predict": -2,
    }
)


def main():
    result = LLMMap(LLM_LOCAL_LLAMA32, """
Please study the following list of words carefully.
For each word in the list, convert the word to uppercase and output it in a JSON list in order of appearance.

$LIST
""".lstrip(), input_words)

    assert len(result) == len(input_words)
    error_count = 0

    for i in range(len(input_words)):
        input_word = input_words[i]
        output_word = result[i]
        if input_word.upper() != output_word:
            print(f"ERROR: {input_word} -> {output_word} (should be {input_word.upper()})")
            error_count += 1

    print(f"{error_count} errors ({(error_count / len(input_words))*100:.1f} %)")

if __name__ == '__main__':
    main()
