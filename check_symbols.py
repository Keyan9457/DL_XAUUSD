import MetaTrader5 as mt5

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print("initialize() failed")
        return False
    return True

def search_symbols(keywords):
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        print("No symbols found.")
        return

    print(f"Total symbols: {len(all_symbols)}")
    found = []
    for s in all_symbols:
        for k in keywords:
            if k.lower() in s.name.lower():
                found.append(s.name)
                break
    
    return sorted(list(set(found)))

if initialize_mt5():
    keywords = ["USD", "DXY", "DX", "US10Y", "TNX", "Bond", "Index"]
    matches = search_symbols(keywords)
    print("\n--- MATCHING SYMBOLS ---")
    for m in matches:
        print(m)
    mt5.shutdown()
