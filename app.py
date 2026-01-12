from email.mime import text
import os
import traceback
from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
    Response,
)
from dotenv import load_dotenv
from supabase import create_client
from supabase_auth import User
from image import smiles_to_svg
import pandas as pd
from match_compounds import match_compounds
import google.genai as genai
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import numpy as np
from scipy.spatial.distance import cosine
from utils import smiles_to_vector, rocchio_update

FILTERED_PATH = "datasets/compounds_filtered.csv"
df = pd.read_csv(FILTERED_PATH)
df = df[["Name", "Smiles", "Molecular Weight", "AlogP", "HBA", "HBD"]]
compounds = df.to_dict(orient="records")

# Precompute vectors 
MOLECULE_CACHE = []
for row in compounds:
    smiles = str(row.get("Smiles", "")).strip()
    name = str(row.get("Name", "Unknown"))
    if not smiles:
        continue

    vec = smiles_to_vector(smiles)
    if vec is None:
        continue

    MOLECULE_CACHE.append( #store for matching
        {
            "name": name,
            "smiles": smiles,
            "vector": vec,
            "props": {
                "molecular_weight": row.get("Molecular Weight"),
                "lipophilicity": row.get("AlogP"),
                "hydrogen_bonding_acceptors": row.get("HBA"),
                "hydrogen_bonding_donors": row.get("HBD"),
            },
        }
    )

print(f"Cache Ready: {len(MOLECULE_CACHE)} compounds.")

load_dotenv()
# Configuring Gemini API
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# load keys
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
FLASK_SECRET_KEY = os.environ["FLASK_SECRET_KEY"]

# flask session
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY


# helpers for supabase
def sb_public():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def sb_service():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def sb_user():
    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    sb.auth.set_session(session["access_token"], session["refresh_token"])
    return sb


def is_logged_in():
    return "access_token" in session and "refresh_token" in session


def require_login():
    if not is_logged_in():
        print(f"DEBUG: require_login failed. Session keys: {list(session.keys())}")
        return redirect(url_for("login"))
    return None


def get_user_id():
    u = sb_user().auth.get_user()
    return u.user.id


def save_match(
    sb_client,
    user_id: str,
    compound_name: str,
    smiles: str,
    properties: dict | None = None,
):
    data = {
        "user_id": user_id,
        "compound_name": compound_name,
        "smiles": smiles,
    }

    if properties:
        data.update(
            {
                "molecular_weight": str(properties.get("molecular_weight", "")),
                "lipophilicity": str(properties.get("lipophilicity", "")),
                "hydrogen_bonding_acceptors": str(
                    properties.get("hydrogen_bonding_acceptors", "")
                ),
                "hydrogen_bonding_donors": str(
                    properties.get("hydrogen_bonding_donors", "")
                ),
            }
        )

    check_res = (
        sb_service()
        .table("matches")
        .select("compound_name")
        .eq("compound_name", data["compound_name"])
        .eq("user_id", user_id)
        .execute()
    )
    compound_exists = len(check_res.data) > 0

    if not compound_exists:
        sb_client.table("matches").insert(data).execute()

def pick_next_match(user_vec: np.ndarray, seen_smiles: set[str], allowed_smiles: set[str] | None):
    # filter out seen smiles from allowed smiles
    available = []
    for m in MOLECULE_CACHE:
        if m["smiles"] in seen_smiles:
            continue
        if allowed_smiles is not None and m["smiles"] not in allowed_smiles:
            continue
        available.append(m)

    if not available:
        return None

    #If user vector is basically all zeros -> can't compute meaningful similarity
    norm = np.linalg.norm(user_vec)
    cold = (
        np.all(user_vec == 0)
        or norm < 1e-10
        or np.any(np.isnan(user_vec))
        or np.any(np.isinf(user_vec))
    )
    #so pick random one
    if cold:
        import random
        return random.choice(available)

    # otherwise: best cosine distance
    best = None
    best_dist = float("inf")
    for m in available:
        try:
            d = cosine(user_vec, m["vector"])  # 0=similar, 1=different
        except Exception:
            continue
        if np.isnan(d) or np.isinf(d):
            continue
        if d < best_dist:
            best_dist = d
            best = m

    return best


@app.route("/", methods=["GET"])
def root():
    return render_template("landing.html")


@app.route("/onboarding", methods=["GET", "POST"])
def onboarding():
    guard = require_login()
    if guard:
        return guard
    if request.method == "POST":
        try:
            molw = request.form.get("molw")
            lip = request.form.get("lip")
            hba = request.form.get("hba")
            hbd = request.form.get("hbd")

            user_id = get_user_id()

            update_data = {
                "user_id": user_id,
                "molecular_weight": molw,
                "lipophilicity": lip,
                "hydrogen_bonding_acceptors": hba,
                "hydrogen_bonding_donors": hbd,
                "index": 0,
            }

            # Check if user exists in prefs
            try:
                check_res = (
                    sb_service()
                    .table("prefs")
                    .select("user_id")
                    .eq("user_id", user_id)
                    .execute()
                )
                user_exists = len(check_res.data) > 0
            except Exception as e:
                user_exists = False

            if user_exists:
                response = (
                    sb_service()
                    .table("prefs")
                    .update(update_data)
                    .eq("user_id", user_id)
                    .execute()
                )
            else:
                response = sb_service().table("prefs").insert(update_data).execute()

        except Exception as e:
            flash("Something went wrong. Please try again.", "error")
            return render_template("onboarding.html")
        return redirect(url_for("matching"))
    return render_template("onboarding.html")


@app.route("/matching", methods=["GET"])
def matching():
    guard = require_login()
    if guard:
        return guard
    res = {
        "molecular_weight": None,
        "lipophilicity": None,
        "hydrogen_bonding_acceptors": None,
        "hydrogen_bonding_donors": None,
    }

    prefs = get_preferences()

    compounds = match_compounds(
        prefs["molecular_weight"],
        prefs["lipophilicity"],
        prefs["hydrogen_bonding_acceptors"],
        prefs["hydrogen_bonding_donors"],
    )

    if "index" not in session:
        session["index"] = 0
        session["accepted"] = []

    idx = prefs["index"]

    if idx >= len(compounds):
        return render_template("matching.html", done=True, res=res)

    compound = compounds[idx]

    res = set_match_info(prefs, compound)

    return render_template(
        "matching.html",
        name=compound.get("Name"),
        smiles=compound.get("Smiles"),
        molecular_weight=compound.get("Molecular Weight"),
        lipophilicity=compound.get("AlogP"),
        hba=compound.get("HBA"),
        hbd=compound.get("HBD"),
        res=res,
        done=False,
    )
@app.route("/api/get-next-match", methods=["GET"])
#Filters first, then ranks inside the filtered set
def api_get_next_match():
    guard = require_login()
    if guard:
        return jsonify({"error": "unauthorized"}), 401

    user_id = get_user_id()

    # 1) Hard preferences 
    pref_res = sb_service().table("prefs").select("*").eq("user_id", user_id).execute()
    prefs = pref_res.data[0] if pref_res.data else {}

    molw = prefs.get("molecular_weight", "na")
    lip  = prefs.get("lipophilicity", "na")
    hba  = prefs.get("hydrogen_bonding_acceptors", "na")
    hbd  = prefs.get("hydrogen_bonding_donors", "na")

    # Use your existing filter engine to restrict candidates
    filtered = match_compounds(molw, lip, hba, hbd)
    allowed_smiles = set(str(c["Smiles"]).strip() for c in filtered if c.get("Smiles"))

    # 2) Load user vector + seen list
    uv = sb_service().table("user_vectors").select("*").eq("user_id", user_id).execute()

    if not uv.data:
        zero_vec = [0.0] * 2048
        sb_service().table("user_vectors").insert(
            {"user_id": user_id, "preference_vector": zero_vec, "seen_smiles": []}
        ).execute()
        user_vec = np.array(zero_vec, dtype=float)
        seen_smiles = set()
    else:
        row = uv.data[0]
        user_vec = np.array(row["preference_vector"], dtype=float)
        seen_smiles = set(row["seen_smiles"] or [])

    # 3) Pick next best match
    best = pick_next_match(user_vec, seen_smiles, allowed_smiles)

    if not best:
        return jsonify({"done": True})

    return jsonify(
        {
            "done": False,
            "name": best["name"],
            "smiles": best["smiles"],
            "properties": best["props"],
        }
    )

@app.route("/api/match", methods=["POST"])
def api_match():
    guard = require_login()
    if guard:
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json()
    action = payload.get("action")
    smiles = (payload.get("smiles") or "").strip()
    name = payload.get("name") or "Unknown"

    prefs = get_preferences()

    compounds = match_compounds(
        prefs["molecular_weight"],
        prefs["lipophilicity"],
        prefs["hydrogen_bonding_acceptors"],
        prefs["hydrogen_bonding_donors"],
    )

    idx = prefs["index"]

    if idx >= len(compounds):
        return jsonify({"done": True})

    compound = compounds[idx]
    user_id = get_user_id()

    # 1) Load current user vector row
    uv = sb_service().table("user_vectors").select("*").eq("user_id", user_id).execute()
    if not uv.data:
        return jsonify({"error": "user_vectors row missing"}), 400

    row = uv.data[0]
    current_vec = np.array(row["preference_vector"], dtype=float)
    seen = row["seen_smiles"] or []

    # 2) convert smile to vector
    mol_vec = smiles_to_vector(smiles)
    if mol_vec is None:
        return jsonify({"error": "invalid smiles"}), 400

    # 3) Rocchio update

    if action == "like":
        properties = {
            "molecular_weight": compound.get("Molecular Weight"),
            "lipophilicity": compound.get("AlogP"),
            "hydrogen_bonding_acceptors": compound.get("HBA"),
            "hydrogen_bonding_donors": compound.get("HBD"),
        }



        save_match(
            sb_user(),
            user_id,
            compound.get("Name"),
            compound.get("Smiles"),
            properties,
        )

        new_vec = rocchio_update(current_vec, [mol_vec], [])
    else:
        new_vec = rocchio_update(current_vec, [], [mol_vec])

    # seen smiles
    if smiles not in seen:
        seen.append(smiles)
    # update user vectors
    sb_service().table("user_vectors").update(
        {"preference_vector": new_vec.tolist(), "seen_smiles": seen}
    ).eq("user_id", user_id).execute()

    update_index(idx + 1)

    next_compound = compounds[idx + 1]
    return jsonify(
        {
            "done": False,
            "name": next_compound["Name"],
            "smiles": next_compound["Smiles"],
        }
    )
    #Save match to matches table if liked 
    # if action == "like":
    #     props = None
    #     for m in MOLECULE_CACHE:
    #         if m["smiles"] == smiles:
    #             props = m["props"]
    #             break

    #     save_match(sb_user(), user_id, name, smiles, props)

    # return jsonify({"status": "ok"})


def set_match_info(prefs, compound):
    res = {
        "molecular_weight": None,
        "lipophilicity": None,
        "hydrogen_bonding_acceptors": None,
        "hydrogen_bonding_donors": None,
    }

    smiles = compound["Smiles"]

    mol = Chem.MolFromSmiles(smiles)

    if prefs["molecular_weight"] != "n/a":
        res["molecular_weight"] = Descriptors.MolWt(mol)
    if prefs["lipophilicity"] != "n/a":
        res["lipophilicity"] = Crippen.MolLogP(mol)
    if prefs["hydrogen_bonding_acceptors"] != "n/a":
        res["hydrogen_bonding_acceptors"] = Lipinski.NumHAcceptors(mol)
    if prefs["hydrogen_bonding_donors"] != "n/a":
        res["hydrogen_bonding_donors"] = Lipinski.NumHDonors(mol)

    return res


# def get_index():
#     res = (
#         sb_service()
#         .table("prefs")
#         .select("index")
#         .eq("user_id", get_user_id())
#         .single()
#         .execute()
#     )
#     return res.data["index"]


def update_index(new_index):
    return (
        sb_service()
        .table("prefs")
        .update({"index": new_index})
        .eq("user_id", get_user_id())
        .execute()
    )


@app.route("/compound-image", methods=["GET"], endpoint="get_compound_image")
def get_compound_image():
    smiles = request.args.get("smiles", "")
    try:
        return Response(smiles_to_svg(smiles), mimetype="image/svg+xml")
    except Exception:
        return "Invalid SMILES", 400


@app.route("/profile", methods=["GET"])
def profile():
    guard = require_login()
    if guard:
        return guard
    if request.method == "GET":
        try:
            prefResponse = (
                sb_service()
                .table("prefs")
                .select("*")
                .eq("user_id", get_user_id())
                .execute()
            )

            print(prefResponse)

            matchResponse = (
                sb_service()
                .table("matches")
                .select("*")
                .eq("user_id", get_user_id())
                .execute()
            )

            return render_template(
                "profile.html", prefs=prefResponse.data, matches=matchResponse.data
            )
        except Exception as e:
            print(e)
            return render_template("profile.html")
    return render_template("profile.html")


def get_preferences():
    response = (
        sb_service().table("prefs").select("*").eq("user_id", get_user_id()).execute()
    )

    res = {
        "molecular_weight": "n/a",
        "lip": "n/a",
        "hba": "n/a",
        "hbd": "n/a",
        "index": 0,
    }

    if response.data:
        res = response.data[0]

    return res


@app.route("/api/profile-chat", methods=["POST"])
def profile_chat():
    guard = require_login()
    if guard:
        return guard

    prompt_type = request.json.get("action")

    user_id = get_user_id()

    print(f"DEBUG: Prompt received: {prompt_type}, user_id: {user_id}")

    prefs_res = sb_service().table("prefs").select("*").eq("user_id", user_id).execute()
    matches_res = (
        sb_service().table("matches").select("*").eq("user_id", user_id).execute()
    )

    print(f"DEBUG: prefs_res: {prefs_res.data}, matches_res: {matches_res.data}")
    prefs = prefs_res.data[0] if prefs_res.data else {}
    matches = matches_res.data

    if not matches:
        return jsonify({"reply": "You have not matched with any compounds yet!"})

    context = {
        "preferences": prefs,
        "matches": [
            {
                "name": m["compound_name"],
                "molecular_weight": m["molecular_weight"],
                "lipophilicity": m["lipophilicity"],
                "hba": m["hydrogen_bonding_acceptors"],
                "hbd": m["hydrogen_bonding_donors"],
            }
            for m in matches
        ],
    }

    system_prompt = f"""
            You are a playful scientific personality analyst.
            You ONLY use the data provided.
            Do NOT invent compounds or preferences.
            Do NOT use markdown styling.

        User data:
        {context}

        User request:
        {prompt_type}

        Respond in a fun but concise card-style explanation, 
        like what makes the user unique scientifically, 
        based on their compound matches and preferences, 
    no more than 150 words.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=system_prompt
    )
    reply_text = response.text
    print(f"DEBUG: Gemini response: {response}")
    return jsonify({"reply": reply_text})


# auth routes
@app.get("/signup")
def signup():
    return render_template("signup.html")


@app.post("/signup")
def signup_post():  # create user in supabase
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not email or not password:
        flash("Email and password required.")
        return redirect(url_for("signup"))

    try:
        resp = sb_public().auth.sign_up({"email": email, "password": password})
    except Exception as e:
        flash(f"Signup failed: {e}")
        return redirect(url_for("signup"))  # error

    if getattr(resp, "session", None) is None:
        flash("Account created. Now log in (and confirm email if required).")
        return redirect(url_for("login"))  # for authentictaed login later

    # store token in flask session cookies
    session["access_token"] = resp.session.access_token
    session["refresh_token"] = resp.session.refresh_token
    # redirect to onboarding
    return redirect(url_for("onboarding"))


@app.get("/login")
def login():
    return render_template("login.html")


@app.post("/login")
def login_post():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    # authenticate
    if not email or not password:
        flash("Email and password required.")
        return redirect(url_for("login"))
    try:
        resp = sb_public().auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception:
        flash("Login failed. Check email/password.")
        return redirect(url_for("login"))

    session["access_token"] = resp.session.access_token
    session["refresh_token"] = resp.session.refresh_token
    return redirect(url_for("profile"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
