import json
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="Smartphone Price Predictor", page_icon="📱", layout="wide")


@st.cache_resource
def load():
    return joblib.load("models/price_model.joblib"), json.load(open("models/metrics.json"))


@st.cache_data
def reference():
    return pd.read_csv("data/smartphones_clean.csv")


bundle, metrics = load()
model, df = bundle["model"], reference()

RES = {"HD+ (720p)": 1.2, "FHD+ (1080p)": 2.6, "QHD+ (1440p)": 4.6}
chip_price = df.groupby("processor").price_inr.median().sort_values()
chip_info = df.groupby("processor")[["cpu_cores", "clock_ghz"]].median()

st.title("📱 Smartphone Price Predictor")
st.caption(f"Estimates a phone's Indian market price (₹) from its specs. "
           f"Trained on {metrics['n_phones']} real phones · {metrics['best_model']}.")

with st.sidebar:
    st.header("Specifications")
    brand = st.selectbox("Brand", sorted(df.brand.unique()), index=sorted(df.brand.unique()).index("Samsung"))
    variant = st.selectbox("Variant", ["standard", "lite", "pro_plus", "ultra"],
                           format_func=lambda v: {"standard": "Standard", "lite": "Lite / Go / Neo",
                                                  "pro_plus": "Pro / Plus", "ultra": "Ultra / Max"}[v])
    chip = st.selectbox("Chipset", list(chip_price.index[::-1]), index=len(chip_price) // 2)
    ram = st.select_slider("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16, 18], value=8)
    storage = st.select_slider("Storage (GB)", [16, 32, 64, 128, 256, 512, 1024], value=128)
    battery = st.slider("Battery (mAh)", 2000, 7000, 5000, 50)
    charge = st.select_slider("Fast charging (W)", [0, 10, 15, 18, 25, 33, 45, 65, 80, 100, 120, 150], value=33)
    screen = st.slider("Screen size (inches)", 4.5, 8.0, 6.6, 0.05)
    res = st.selectbox("Resolution", list(RES), index=1)
    refresh = st.select_slider("Refresh rate (Hz)", [60, 90, 120, 144, 165], value=120)
    notch = st.selectbox("Display cut-out", ["punch hole", "water drop", "notch", "none"])
    rear = st.selectbox("Main rear camera (MP)", [8, 13, 16, 32, 48, 50, 64, 108, 200], index=5)
    n_rear = st.selectbox("Rear cameras", [1, 2, 3, 4], index=2)
    front = st.selectbox("Front camera (MP)", [5, 8, 13, 16, 32, 50], index=3)
    os_name = "iOS" if brand == "Apple" else "Android"
    os_ver = st.slider("OS version at launch", 8, 17, 13 if os_name == "Android" else 16)
    c1, c2 = st.columns(2)
    g5 = c1.checkbox("5G", True)
    nfc = c2.checkbox("NFC", False)
    ir = c1.checkbox("IR blaster", False)
    sd = c2.checkbox("microSD slot", False)
    fold = st.checkbox("Foldable", False)

row = pd.DataFrame([dict(
    brand=brand, os_name=os_name, notch_type=notch, variant=variant, processor=chip,
    ram_gb=ram, storage_gb=storage, cpu_cores=chip_info.loc[chip, "cpu_cores"],
    clock_ghz=chip_info.loc[chip, "clock_ghz"], battery_mah=battery, fast_charge_w=charge,
    screen_inches=screen, pixels_mp=RES[res], refresh_rate_hz=refresh, rear_camera_mp=rear,
    n_rear_cameras=n_rear, front_camera_mp=front, has_5g=int(g5), has_nfc=int(nfc),
    has_ir_blaster=int(ir), expandable_storage=int(sd), is_foldable=int(fold), os_version=os_ver,
)])
price = float(model.predict(row[bundle["features"]])[0])
lo, hi = price * (1 - metrics["test_mape_pct"] / 100), price * (1 + metrics["test_mape_pct"] / 100)

m1, m2, m3 = st.columns(3)
m1.metric("Predicted price", f"₹{price:,.0f}")
m2.metric("Typical range (±MAPE)", f"₹{lo:,.0f} – ₹{hi:,.0f}")
m3.metric("Within ±15% on test set", f"{metrics['within_15pct']}%")
st.caption("Brand-new or unusual spec combinations (e.g. luxury editions) fall outside the training data and "
           "may be predicted less accurately.")

st.subheader("Closest-priced phones in the dataset")
near = df.assign(gap=(df.price_inr - price).abs()).nsmallest(6, "gap")
st.dataframe(near[["model", "price_inr", "processor", "ram_gb", "storage_gb", "battery_mah",
                   "rear_camera_mp", "refresh_rate_hz", "has_5g"]],
             hide_index=True)

t1, t2, t3 = st.tabs(["Model performance", "What drives price", "Price explorer"])
with t1:
    a, b = st.columns(2)
    a.json(metrics)
    b.image("assets/actual_vs_pred.png")
with t2:
    st.image("assets/feature_importance.png", width=560)
with t3:
    sel = st.multiselect("Brands", sorted(df.brand.unique()), default=["Samsung", "Xiaomi", "Apple", "OnePlus"])
    if sel:
        st.bar_chart(df[df.brand.isin(sel)].groupby("brand").price_inr.median())
    st.scatter_chart(df, x="ram_gb", y="price_inr", color="brand" if sel else None)
