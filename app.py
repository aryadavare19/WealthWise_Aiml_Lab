# app.py — final integrated version (ML yearly predictions, robust fallbacks)
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import joblib
import numpy as np
import sqlite3
import pandas as pd
import traceback
from risk_engine import compute_risk_score, get_risk_label


app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"

# ===============================
# MODEL PATHS (all in /models/)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPENSE_NLP_MODEL_PATH = os.path.join(MODELS_DIR, "expense_nlp_model.pkl")
# Combined inference pipeline exported by your training script
SELECTED_PIPELINE_PATH = os.path.join(MODELS_DIR, "selected_model_pipeline.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")

# other models (unchanged)
INVESTMENT_MODEL_PATH = os.path.join(MODELS_DIR, "investment.pkl")
EXPENSE_MODEL_PATH = os.path.join(MODELS_DIR, "feature3_expense_dataset (1).pkl")
RETIREMENT_MODEL_PATH = os.path.join(MODELS_DIR, "retirement_linear_model_indian.pkl")

# ===============================
# LOAD MODELS
# ===============================
def load_model(path):
    try:
        obj = joblib.load(path)
        print(f"✅ Model loaded successfully from: {path}")
        return obj
    except Exception as e:
        print(f"⚠️ Error loading model {path}: {e}")
        return None

asset_pipeline = load_model(SELECTED_PIPELINE_PATH)          # pipeline (preprocessor + model)
asset_label_encoders = load_model(LABEL_ENCODERS_PATH)      # optional dict of LabelEncoders
investment_model = load_model(INVESTMENT_MODEL_PATH)
expense_model = load_model(EXPENSE_MODEL_PATH)
retirement_linear_model_indian = load_model(RETIREMENT_MODEL_PATH)
expense_nlp_model=load_model(EXPENSE_NLP_MODEL_PATH)
# ===============================
# DATABASE SETUP
# ===============================
DB_NAME = os.path.join(BASE_DIR, "wealthwise.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # existing expenses table
    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        income REAL,
        total_expenses REAL,
        savings REAL
    )''')

    # assets table for portfolio & projections
    c.execute('''CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_name TEXT,
        asset_type TEXT,
        quantity REAL,
        price_per_unit REAL,
        expected_return REAL
    )''')

    conn.commit()
    conn.close()

init_db()

# ===============================
# Helper functions
# ===============================
def get_assets_list():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, asset_name, asset_type, quantity, price_per_unit, expected_return FROM assets ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    assets = []
    for r in rows:
        assets.append({
            "id": int(r[0]),
            "asset_name": r[1] or "",
            "asset_type": r[2] or "",
            "quantity": float(r[3]),
            "price_per_unit": float(r[4]),
            "expected_return": float(r[5])
        })
    return assets

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def predict_price_with_pipeline(row_df):
    """
    row_df: pandas.DataFrame with a single row. Returns predicted price (float) or None on failure.
    Uses label encoders if provided to transform categorical columns expected by pipeline.
    """
    try:
        # If label encoders provided (from training), apply them to row_df columns if needed.
        if asset_label_encoders:
            for col, le in asset_label_encoders.items():
                if col in row_df.columns:
                    val = str(row_df.at[0, col])
                    try:
                        encoded = le.transform([val])[0]
                    except Exception:
                        # unseen label -> fallback to first or 'missing' if present
                        try:
                            if 'missing' in le.classes_:
                                encoded = int(np.where(le.classes_ == 'missing')[0][0])
                            else:
                                encoded = 0
                        except Exception:
                            encoded = 0
                    row_df[col] = encoded

        # pipeline (preprocessor + model) expects DataFrame or array depending on how saved
        pred = asset_pipeline.predict(row_df)
        # convert to scalar float
        pred_val = float(np.array(pred).reshape(-1)[0])
        # sanity: must be finite
        if np.isfinite(pred_val):
            return pred_val
        else:
            return None
    except Exception as e:
        print("⚠️ Prediction with pipeline failed:", str(e))
        traceback.print_exc()
        return None

# ===============================
# ROUTES
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ----------------------------------------------------
## 💰 EXPENSES & SAVINGS (Updated with Categories)
# ----------------------------------------------------
@app.route('/expenses_savings', methods=['GET', 'POST'])
def expenses_savings():
    savings, income, total_expenses, prediction_text = None, 0, 0, None
    
    # Categories for dropdown
    expense_categories = ['entertainment', 'food', 'health', 'others', 'rent', 'transport', 'utilities']
    
    # Default empty row for GET
    expenses_list = [{"category": "", "amount": ""}]

    if request.method == 'POST':
        try:
            income = float(request.form.get('income', '0'))
            
            raw_categories = request.form.getlist('category[]')
            raw_amounts = request.form.getlist('amount[]')
            
            valid_expenses = []
            total_calc = 0
            
            # Combine the lists
            for cat, amt in zip(raw_categories, raw_amounts):
                if amt.strip(): # Only keep rows with amounts
                    val = float(amt)
                    total_calc += val
                    valid_expenses.append({"category": cat, "amount": amt})
            
            total_expenses = total_calc
            savings = round(income - total_expenses, 2)
            
            # Send this list back to HTML to retain values
            if valid_expenses:
                expenses_list = valid_expenses
            
            # DB and AI Logic...
            conn = sqlite3.connect(DB_NAME)
            conn.cursor().execute("INSERT INTO expenses (income, total_expenses, savings) VALUES (?, ?, ?)",
                           (income, total_expenses, savings))
            conn.commit()
            conn.close()

            if expense_model and income > 0:
                pred = expense_model.predict(np.array([[income]]))[0]
                prediction_text = f"AI Forecast: ₹{pred:,.2f}"

        except Exception as e:
            savings = f"Error: {e}"
            # On error, force retain raw inputs so user doesn't lose work
            expenses_list = [{"category": c, "amount": a} for c, a in zip(request.form.getlist('category[]'), request.form.getlist('amount[]'))]
    
    return render_template(
        'expenses_savings.html',
        savings=savings,
        income=income,
        total_expenses=total_expenses,
        prediction_text=prediction_text,
        expenses_list=expenses_list,        # Key variable for retention
        expense_categories=expense_categories # Key variable for dropdown
    )
    
    
# -------------------------
# Expense Category (NLP)
# -------------------------
@app.route('/expense_category')
def expense_category_page():
    return render_template('expense_category.html', description=None, prediction=None)

@app.route('/predict_expense_category', methods=['POST'])
def predict_expense_category():
    desc = request.form.get("description", "").strip()

    if desc == "":
        flash("Enter a description!", "error")
        return redirect(url_for('expense_category_page'))

    try:
        # Step 1: Model prediction
        pred = expense_nlp_model.predict([desc])[0]

        # Step 2: Keyword-based refinement
        desc_lower = desc.lower()

        # --- Keywords for each category ---
        keywords_map = {
            "Food & Dining": [
                "restaurant", "cafe", "fast food", "swiggy", "zomato", "starbucks", "mcdonald", "burger", "pizza", "dining"
            ],
            "Travel & Transport": [
                "uber", "ola", "taxi", "bus", "metro", "train", "flights", "airline", "fuel", "petrol", "diesel", "cab", "taxi fare"
            ],
            "Groceries & Essentials": [
                "supermarket", "d-mart", "big bazaar", "reliance fresh", "groceries", "vegetables", "milk", "eggs", "bread", "rice", "shopping essentials"
            ],
            "Entertainment & Leisure": [
                "netflix", "prime", "hulu", "movies", "gaming", "concert", "amusement", "theater", "subscription", "streaming", "game"
            ],
            "Health & Medical": [
                "pharmacy", "hospital", "doctor", "medicine", "clinic", "health", "dental", "eyecare", "checkup", "lab test"
            ],
            "Bills & Utilities": [
                "electricity", "water", "gas", "wifi", "internet", "mobile recharge", "phone bill", "rent", "utility"
            ],
            "Education Expenses": [
                "education", "tuition", "school", "college", "course", "exam", "certification", "training", "coaching", "fees", "admission"
            ],
            "Others / Miscellaneous": [
                "unknown", "random", "miscellaneous", "one-off", "misc", "donation", "gift", "charity", "other"
            ]
        }

        # --- Check keywords and override prediction if matched ---
        for category, keywords in keywords_map.items():
            for word in keywords:
                if word in desc_lower:
                    pred = category
                    break
            if pred == category:
                break

        return render_template("expense_category.html", description=desc, prediction=pred)

    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for('expense_category_page'))
# -------------------------
# Asset Growth UI (renders the page)
# -------------------------
@app.route('/asset_growth', methods=['GET'])
def asset_growth():
    default_years = 10
    assets = get_assets_list()
    return render_template('asset_growth.html', assets=assets, years=default_years)

# -------------------------
# Add asset (form POST)
# -------------------------
@app.route('/asset_growth/add', methods=['POST'])
def add_asset():
    try:
        # The HTML uses only asset_type dropdown; keep compatibility:
        asset_type = request.form.get('asset_type', '').strip()
        asset_name = request.form.get('asset_name', '').strip() or asset_type  # if asset_name missing, use asset_type
        quantity = safe_float(request.form.get('quantity', 0), 0)
        price_per_unit = safe_float(request.form.get('price_per_unit', 0), 0)
        expected_return = safe_float(request.form.get('expected_return', 0), 0)

        if asset_type == "" and asset_name == "":
            flash("Please choose an asset type or provide asset name.", "error")
            return redirect(url_for('asset_growth'))

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO assets (asset_name, asset_type, quantity, price_per_unit, expected_return)
            VALUES (?, ?, ?, ?, ?)
        """, (asset_name, asset_type, quantity, price_per_unit, expected_return))
        conn.commit()
        conn.close()

        flash("Asset added successfully!", "success")
        return redirect(url_for('asset_growth'))

    except Exception as e:
        traceback.print_exc()
        flash(f"Error adding asset: {e}", "error")
        return redirect(url_for('asset_growth'))

# -------------------------
# Delete asset (POST)
# -------------------------
@app.route('/delete_asset/<int:asset_id>', methods=['POST'])
def delete_asset(asset_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM assets WHERE id=?", (asset_id,))
        conn.commit()
        conn.close()
        flash("Asset deleted.", "success")
    except Exception as e:
        traceback.print_exc()
        flash(f"Error deleting asset: {e}", "error")
    return redirect(url_for('asset_growth'))

# -------------------------
# Update timeframe/projection (AJAX POST)
# -------------------------
@app.route('/update_timeframe', methods=['POST'])
def update_timeframe():
    """
    For each asset:
      - use the ML pipeline to predict Year 1 price (per unit) given current features
      - for each subsequent year:
          * advance baseline using asset's expected_return
          * create a feature row using the new baseline and call the ML pipeline again
      - final series values are quantity * predicted_price_per_unit_for_that_year
    Response JSON:
      {
        "years": [1,2,...],
        "projection": { "<asset_id>": [v1, v2, ...], ... },
        "labels": { "<asset_id>": "Gold (id#3)", ... },
        "total": [sum1, sum2, ...]
      }
    """
    try:
        years = int(request.form.get('years', 10) or 10)
        if years < 1:
            return jsonify({"error": "years must be >= 1"}), 400

        assets = get_assets_list()
        projection = {}
        labels = {}
        total_series = [0.0] * years

        for a in assets:
            aid = str(a['id'])
            qty = a['quantity']
            price = a['price_per_unit']
            exp_ret = a['expected_return']  # percent

            # Build base feature row (adjust these columns to match features your pipeline expects)
            # Use price for open/high/low/close/adj_close and conservative defaults for technicals
            def build_feature_row(base_price):
                return {
                    "open": base_price,
                    "high": base_price,
                    "low": base_price,
                    "close": base_price,
                    "adj_close": base_price,
                    "volume": 0.0,
                    "daily_return": 0.0,
                    "ma_5": base_price,
                    "ma_20": base_price,
                    "ma_50": base_price,
                    "vol_20": 0.0,
                    "vol_norm": 0.0,
                    "rsi_14": 50.0,
                    "macd_line": 0.0,
                    "macd_signal": 0.0,
                    "macd_hist": 0.0,
                    # categorials that might be used by model
                    "asset": a['asset_name'],
                    "asset_type": a['asset_type'] or "unknown",
                    "date": 0
                }

            series = []
            # YEAR 1: attempt ML prediction seeded by current price
            seed_row = pd.DataFrame([build_feature_row(price)])
            pred_price = None
            if asset_pipeline:
                pred_price = predict_price_with_pipeline(seed_row)

            # sanity/fallbacks:
            # if model returns None or absurdly small/negative result, fallback to current price
            if (pred_price is None) or (not np.isfinite(pred_price)) or (pred_price <= 0):
                pred_price = price

            # if model predicts drastically different (e.g., <50% of current price) we still allow it,
            # but to avoid unrealistic negative starts, if it's < 30% of current price, fallback to current price
            if price > 0 and pred_price < 0.3 * price:
                pred_price = price

            prev_price = pred_price

            # Year-by-year predictions:
            for y in range(1, years + 1):
                if y == 1:
                    unit_price_for_year = prev_price
                else:
                    # Advance baseline by expected_return to get new baseline for next-year prediction
                    baseline = prev_price * (1 + (exp_ret / 100.0))
                    # build new feature row for this baseline and call ML again
                    row_y = pd.DataFrame([build_feature_row(baseline)])
                    pred_y = None
                    if asset_pipeline:
                        pred_y = predict_price_with_pipeline(row_y)

                    # If model fails or gives non-sane result, fallback to baseline (compounded)
                    if (pred_y is None) or (not np.isfinite(pred_y)) or (pred_y <= 0):
                        pred_y = baseline

                    # again protect from absurd drops: if pred_y < 0.5 * baseline, use baseline
                    if pred_y < 0.5 * baseline:
                        pred_y = baseline

                    unit_price_for_year = pred_y
                    prev_price = pred_y  # use predicted as prev for next iteration

                # compute total value (quantity * unit_price)
                total_val = round(float(qty * unit_price_for_year), 2)
                series.append(total_val)

                # accumulate to totals
                total_series[y - 1] += total_val

            # label for chart legend
            label = f"{a['asset_type']} (#{a['id']})" if a['asset_type'] else f"{a['asset_name']} (#{a['id']})"
            projection[aid] = series
            labels[aid] = label

        total_series = [round(float(x), 2) for x in total_series]

        return jsonify({
            "years": list(range(1, years + 1)),
            "projection": projection,
            "labels": labels,
            "total": total_series
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------
# Investment Pathway (randomForest)
# -------------------------
@app.route("/investment_path", methods=["GET"])
def investment_path():
    defaults = {
        "age": 18,
        "monthly_income": 0,
        "current_savings": 0,
        "investment_horizon_years": 1,
        "credit_score": 650,
        "expenses_per_month": 0,
        "debt_amount": 0,
        "marital_status": 0,
        "dependents": 0,
        "has_existing_investments": 0,
    }
    return render_template("investment_path.html", defaults=defaults, result=None)

@app.route("/predict_investment", methods=["POST"])
def predict_investment():
    if investment_model is None:
        flash("Model not loaded. Please check the investment model file.", "error")
        return redirect(url_for("investment_path"))

    try:
        # Collect inputs
        age = int(request.form.get("age"))
        monthly_income = float(request.form.get("monthly_income"))
        current_savings = float(request.form.get("current_savings"))
        investment_horizon_years = float(request.form.get("investment_horizon_years"))
        credit_score = float(request.form.get("credit_score"))
        expenses_per_month = float(request.form.get("expenses_per_month"))
        debt_amount = float(request.form.get("debt_amount"))
        marital_status = int(request.form.get("marital_status"))
        dependents = int(request.form.get("dependents"))
        has_existing_investments = int(request.form.get("has_existing_investments"))
        investment_amount = float(request.form.get("investment_amount"))

        # Prepare input for the ML model
        X = np.array([[age, monthly_income, current_savings, investment_horizon_years,
                       credit_score, expenses_per_month, debt_amount, marital_status,
                       dependents, has_existing_investments]])

        # Predict allocation percentages
        preds = investment_model.predict(X)[0]
        preds = np.clip(preds, 0, 100)
        preds = (preds / preds.sum()) * 100

        # Extract asset allocation
        equity, debt, gold, real_estate = [round(float(p), 2) for p in preds]

        # Calculate split amounts
        investment_split = {
            "equity": round(investment_amount * (equity / 100), 2),
            "debt": round(investment_amount * (debt / 100), 2),
            "gold": round(investment_amount * (gold / 100), 2),
            "real_estate": round(investment_amount * (real_estate / 100), 2),
        }

        # Compute risk score
        risk_score = compute_risk_score(
            age, monthly_income, current_savings, investment_horizon_years,
            credit_score, expenses_per_month, debt_amount, marital_status,
            dependents, has_existing_investments
        )
        risk_label = get_risk_label(risk_score)

        # Response
        result = {
            "equity": equity,
            "debt": debt,
            "gold": gold,
            "real_estate": real_estate,
            "investment_amount": investment_amount,
            "investment_split": investment_split,
            "risk_score": risk_score,
            "risk_label": risk_label,
        }

        # Defaults
        defaults = {
            "age": age,
            "monthly_income": monthly_income,
            "current_savings": current_savings,
            "investment_horizon_years": investment_horizon_years,
            "credit_score": credit_score,
            "expenses_per_month": expenses_per_month,
            "debt_amount": debt_amount,
            "marital_status": marital_status,
            "dependents": dependents,
            "has_existing_investments": has_existing_investments,
            "investment_amount": investment_amount,
        }

        return render_template("investment_path.html", result=result, defaults=defaults)

    except Exception as e:
        flash(f"Error during prediction: {e}", "error")
        return redirect(url_for("investment_path"))




# ----------------------------------------------------
# RETIREMENT READINESS (Using MLR Model)
# ----------------------------------------------------
@app.route('/retirement')
def retirement_page():
    return render_template('retirement_readiness.html')

@app.route('/predict_retirement', methods=['POST'])
def predict_retirement():
    score = None
    status = None
    color = None
    
    try:
        # Collect inputs
        age = float(request.form['age'])
        retirement_age = float(request.form['retirement_age'])
        income = float(request.form['income'])
        expenses = float(request.form['expenses'])
        savings = float(request.form['savings'])
        investment = float(request.form['investment'])
        inflation = float(request.form['inflation'])
        return_rate = float(request.form['return_rate'])

        # Validation
        if not (18 <= age <= 70) or retirement_age < age or income <= 0:
            status, color, score = "Invalid Input", "#e74c3c", "Error"
        else:
            features = np.array([[age, retirement_age, income, expenses, savings, investment, inflation, return_rate]])

            if retirement_linear_model_indian:
                # Predict using the MLR model
                raw_score = retirement_linear_model_indian.predict(features)[0]
                
                # ✅ MLR Specific: Clamp score between 0 and 100
                score = round(max(0, min(100, raw_score)), 2)

                # Assign Status based on the score
                if score >= 80:
                    status, color = "✅ Excellent – You’re well-prepared!", "#2ecc71"
                elif score >= 60:
                    status, color = "🟡 Good – On track, but could improve.", "#f1c40f"
                else:
                    status, color = "🔴 Needs Attention – Review your plan.", "#e74c3c"
            else:
                status, color, score = "Model Not Loaded", "#e74c3c", "Error"
        
        return render_template(
            'retirement_readiness.html',
            age=age, retirement_age=retirement_age, income=income,
            expenses=expenses, savings=savings, investment=investment,
            inflation=inflation, return_rate=return_rate,
            score=score, status=status, color=color
        )

    except Exception as e:
        return render_template('retirement_readiness.html', score="Error", status=f"Error: {str(e)}", color="#e74c3c")

# -------------------------
# Learn more
# -------------------------
@app.route('/learn_more')
def learn_more():
    return "<h2 style='text-align:center;margin-top:100px;'>📘 Learn More — Coming Soon!</h2>"

# ===============================
# RUN APP
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
