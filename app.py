from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from datetime import datetime, date

# Path to your Excel file (change if needed)
EXCEL_PATH = "E:\\Aspire Tech Academy Bangalore\\Data Science Tools\\Machine Learning\\Machine Learning Projects\\Xpress Bees Project\\XpressBees.xlsx"

app = Flask(__name__)

def load_options_from_excel(path=EXCEL_PATH):
    """
    Read the Excel and return a dict mapping column -> sorted unique values (as strings).
    If file or column missing, returns empty lists for that column.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        print("Error reading Excel:", e)
        df = pd.DataFrame()

    # List of attributes requested by user (keeps order)
    cols = [
        'ShipPinCode',
        'ProcessLocation1_InscanDate',
        'ProcessLocation1_BagOutScanDate',
        'ProcessLocation2_BagInScanDate',
        'ProcessLocation2_BagOutScanDate',
        'ProcessLocation3_BagInScanDate',
        'ProcessLocation4_BagInScanDate',
        'ProcessLocation4_BagOutScanDate',
        'Destination_BagInScanDate',
        'ProcessLocation3_BagOutScanDate_replaced',
        'Destination_BagOutScanDate_replaced',
        'OriginHubName',
        'OriginHubCity',
        'OriginHubZoneName',
        'ShipmentStatus',
        'ProcessLocation2_HubName_HubCity',
        'ProcessLocation3_HubName_HubCity',
        'ProcessLocation4_HubName_HubCity',
        'Destination_HubName_HubCity',
        'DestinationHubName',
        'DestinationHubCity',
        'DestinationHubZoneName',
        'Delivery Date',
        'Physical Weight',
        'Volumetric Weight',
        'lane'
    ]

    options = {}
    for c in cols:
        if c in df.columns:
            # Convert to strings for dropdown display, dropna, unique and sort
            vals = df[c].dropna().astype(str).unique().tolist()
            try:
                vals = sorted(vals)
            except:
                pass
            options[c] = vals
        else:
            options[c] = []
    return options

# Load options at startup (you can refresh by restarting server)
OPTIONS = load_options_from_excel()

def predict_route_mode(delivery_date_str, physical_weight_str, volumetric_weight_str, lane_str):
    """
    Simple rule-based predictor for demo:
    - If expected delivery date within 2 days -> Air
    - If physical weight >= 20 kg -> Surface (assume heavy shipments often surface) *
      (change as per your business logic)
    - If volumetric_weight > physical_weight * 3 -> Air (bulky but light)
    - Certain lanes can be prioritized for Air (e.g., if lane contains 'Metro' or 'ROI' we might prefer Air)
    - Default -> Surface

    NOTE: These are demo heuristics. Replace with a trained model for production.
    """
    # Parse weights
    try:
        w = float(physical_weight_str) if physical_weight_str not in (None, '', 'nan') else 0.0
    except:
        w = 0.0
    try:
        vw = float(volumetric_weight_str) if volumetric_weight_str not in (None, '', 'nan') else 0.0
    except:
        vw = 0.0

    # Parse delivery date
    today = date.today()
    days_to_delivery = None
    if delivery_date_str:
        try:
            # Attempt common formats
            dt = None
            try:
                dt = datetime.strptime(delivery_date_str, "%Y-%m-%d").date()
            except:
                try:
                    dt = datetime.strptime(delivery_date_str, "%d-%m-%Y").date()
                except:
                    dt = pd.to_datetime(delivery_date_str, errors='coerce').date()
            if isinstance(dt, date):
                days_to_delivery = (dt - today).days
        except Exception:
            days_to_delivery = None

    lane = (lane_str or "").lower()

    # Heuristics
    # urgent deliveries -> Air
    if days_to_delivery is not None and days_to_delivery <= 2:
        return "Air", f"Delivery in {days_to_delivery} day(s) → prioritized as Air (urgent)."

    # bulky but light -> Air
    if vw > 0 and vw >= w * 3:
        return "Air", "High volumetric weight relative to physical weight → route recommended: Air."

    # very heavy -> Surface
    if w >= 50:
        return "Surface", "Very heavy shipment (>= 50 kg) → Surface recommended."

    # medium weight but metro lanes → Air for speed
    if (w >= 10 and w < 50) and ("metro" in lane or "roi" in lane or "north" in lane or "east" in lane):
        return "Air", "Medium weight & metro/ROI/North East lane → Air preferred for speed."

    # light shipments on metro → Air
    if w < 10 and ("metro" in lane or "roi" in lane):
        return "Air", "Light shipment & Metro/ROI lane → Air recommended."

    # default
    return "Surface", "Default routing → Surface."

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", options=OPTIONS, company_name="XpressBees Logistics Services Provider")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect all fields (we will pass result back and also show what user selected)
    form_data = {}
    for k in OPTIONS.keys():
        # HTML form fields will use safe names (replace spaces with __)
        name = k.replace(" ", "__")
        form_data[k] = request.form.get(name, "")

    # additional numeric fields might be in separate numeric inputs
    physical_weight = request.form.get("Physical Weight".replace(" ", "__"), form_data.get("Physical Weight", ""))
    volumetric_weight = request.form.get("Volumetric Weight".replace(" ", "__"), form_data.get("Volumetric Weight", ""))
    delivery_date = request.form.get("Delivery Date".replace(" ", "__"), form_data.get("Delivery Date", ""))

    lane_val = request.form.get("lane".replace(" ", "__"), form_data.get("lane", ""))

    mode, reason = predict_route_mode(delivery_date, physical_weight, volumetric_weight, lane_val)

    return render_template("index.html",
                           options=OPTIONS,
                           company_name="XpressBees Logistics Services Provider",
                           prediction=mode,
                           reason=reason,
                           submitted=form_data)

if __name__ == "__main__":
    app.run(debug=True)
