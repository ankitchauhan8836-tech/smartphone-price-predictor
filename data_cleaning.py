"""Parse the raw scraped smartphone CSV (text-heavy, partly misaligned) into
a clean numeric feature table.

Raw columns hold strings like "12 GB RAM, 256 GB inbuilt" or
"5000 mAh Battery with 100W Fast Charging". Some rows (foldables, feature
phones) have extra/missing cells that shift columns, so instead of trusting
column positions we locate each field by pattern across the row's cells.
"""
import re
import numpy as np
import pandas as pd

BRAND_MAP = {"oppo": "Oppo", "poco": "Poco", "itel": "Itel", "iqoo": "iQOO", "redmi": "Xiaomi",
             "lg": "LG", "zte": "ZTE", "oneplus": "OnePlus", "iphone": "Apple"}
CORES = {"single": 1, "dual": 2, "quad": 4, "hexa": 6, "octa": 8, "deca": 10}


def _find(cells, pattern):
    for c in cells:
        if isinstance(c, str) and re.search(pattern, c, re.I):
            return c
    return None


def _num(pattern, text, group=1, cast=float):
    if not isinstance(text, str):
        return np.nan
    m = re.search(pattern, text, re.I)
    return cast(m.group(group).replace(",", "")) if m else np.nan


def detect_variant(model):
    m = model.lower()
    if re.search(r"\b(ultra|max|pro\s*max)\b", m): return "ultra"
    if re.search(r"\bpro\b|\bpro\+|\bplus\b|\+", m): return "pro_plus"
    if re.search(r"\b(lite|go|neo|mini|play|spark|pop|a\d?\s*core)\b", m): return "lite"
    return "standard"


def parse_row(r):
    cells = [str(c).replace("\u2009", " ").replace("\xa0", " ") if pd.notna(c) else None
             for c in r.iloc[3:]]
    model = str(r["model"]).strip()
    first = model.split()[0].lower()
    brand = BRAND_MAP.get(first, model.split()[0].title() if first.islower() else model.split()[0])
    if model.lower().startswith("apple"):
        brand = "Apple"

    sim = _find(cells, r"\bsim\b|wi-?fi|4g|5g") or ""
    proc = _find(cells, r"processor") or ""
    ram = _find(cells, r"\bRAM\b") or ""
    bat = _find(cells, r"mAh") or ""
    disp = _find(cells, r"inches") or ""
    cam = _find(cells, r"\bMP\b.*(rear|front|camera)|camera") or ""
    card = _find(cells, r"memory card") or ""
    os_ = _find(cells, r"android|ios|harmony|emui|rtos|kai|miui") or ""
    all_text = " ".join(c for c in cells if c)

    # RAM / storage (handle MB, GB, TB)
    ram_val = _num(r"([\d.]+)\s*(?:GB|MB) RAM", ram)
    if re.search(r"MB RAM", ram):
        ram_val /= 1024
    st = re.search(r"([\d.]+)\s*(GB|MB|TB)\s*inbuilt", ram)
    storage = np.nan
    if st:
        storage = float(st.group(1)) * {"MB": 1 / 1024, "GB": 1, "TB": 1024}[st.group(2)]

    # processor
    pname = proc.split(",")[0] if proc else ""
    pname = re.sub(r"\s+", " ", pname).strip()
    cores = next((v for k, v in CORES.items() if k in proc.lower()), np.nan)
    family = pname.split()[0] if pname else "Unknown"

    # battery / charging
    mah = _num(r"([\d.]+)\s*mAh", bat)
    watts = _num(r"([\d.]+)\s*W\b", bat)
    has_fast = int("fast charging" in bat.lower())
    if not has_fast:
        watts = 0.0

    # display
    inches = _num(r"([\d.]+)\s*inches", disp)
    res = re.search(r"(\d+)\s*x\s*(\d+)\s*px", disp)
    rw, rh = (float(res.group(1)), float(res.group(2))) if res else (np.nan, np.nan)
    hz = _num(r"(\d+)\s*Hz", disp)
    notch = ("punch hole" if "punch" in disp.lower() else "water drop" if "water drop" in disp.lower()
             else "notch" if "notch" in disp.lower() else "none")

    # camera
    rear_part = cam.split("&")[0] if cam else ""
    front_part = cam.split("&")[1] if "&" in cam else ""
    rear_mps = [float(x) for x in re.findall(r"([\d.]+)\s*MP", rear_part)]
    front_mps = [float(x) for x in re.findall(r"([\d.]+)\s*MP", front_part)]

    # OS
    os_name = ("iOS" if "ios" in os_.lower() else "Android" if "android" in os_.lower()
               else "Other" if os_ else "Other")
    os_ver = _num(r"v\s*([\d]+)", os_)

    return dict(
        model=model, brand=brand, variant=detect_variant(model),
        price_inr=float(re.sub(r"[^\d.]", "", str(r["price"])) or np.nan),
        ram_gb=ram_val, storage_gb=storage,
        processor=pname, processor_family=family, cpu_cores=cores,
        clock_ghz=_num(r"([\d.]+)\s*GHz", proc),
        battery_mah=mah, fast_charge_w=watts, has_fast_charge=has_fast,
        screen_inches=inches, res_width=rw, res_height=rh, refresh_rate_hz=hz,
        notch_type=notch,
        rear_camera_mp=max(rear_mps) if rear_mps else np.nan,
        n_rear_cameras=len(rear_mps) if rear_mps else np.nan,
        front_camera_mp=max(front_mps) if front_mps else np.nan,
        has_5g=int("5g" in sim.lower() or "5g" in pname.lower()),
        has_nfc=int("nfc" in sim.lower()),
        has_ir_blaster=int("ir blaster" in sim.lower()),
        has_volte=int("volte" in sim.lower()),
        expandable_storage=int("supported" in card.lower() and "not supported" not in card.lower()
                               or "hybrid" in card.lower()),
        is_foldable=int("foldable" in all_text.lower()),
        os_name=os_name, os_version=os_ver,
    )


def load_and_clean(path="data/smartphones_raw.csv", min_price=2000, max_price=200000):
    raw = pd.read_csv(path)
    df = pd.DataFrame([parse_row(r) for _, r in raw.iterrows()])
    df["pixels_mp"] = df.res_width * df.res_height / 1e6
    df["refresh_rate_hz"] = df.refresh_rate_hz.fillna(60)        # no Hz listed => standard 60 Hz
    df = df.drop_duplicates(subset="model")
    df = df[df.price_inr >= min_price]                            # drop junk/placeholder prices
    df = df[df.price_inr <= max_price]                            # luxury/signature editions (e.g. Vertu) priced by brand, not specs
    df = df[(df.screen_inches >= 3) & (df.battery_mah >= 1000)]   # drop mis-scraped rows (watches/accessories)
    df = df[df.ram_gb >= 0.5]
    df = df.dropna(subset=["ram_gb", "storage_gb", "battery_mah", "screen_inches"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = load_and_clean()
    d.to_csv("data/smartphones_clean.csv", index=False)
    print(d.shape); print(d.isna().sum()[d.isna().sum() > 0]); print(d.describe().T.round(2).to_string())
    print(d.brand.value_counts().head(25).to_dict())
