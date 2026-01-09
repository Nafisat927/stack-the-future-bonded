import os
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
    jsonify,
)
from dotenv import load_dotenv
from supabase import create_client
from supabase_auth import User
from image import smiles_to_svg
import pandas as pd

FILTERED_PATH = "datasets/compounds_filtered.csv"
df = pd.read_csv(FILTERED_PATH)
df = df[["Name", "Smiles", "Molecular Weight", "AlogP", "HBA", "HBD"]]
compounds = df.to_dict(orient="records")

load_dotenv()

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

    sb_client.table("matches").insert(data).execute()


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

            update_data = {}
            if molw != "na":
                update_data["molecular_weight"] = molw
            if lip != "na":
                update_data["lipophilicity"] = lip
            if hba != "na":
                update_data["hydrogen_bonding_acceptors"] = hba
            if hbd != "na":
                update_data["hydrogen_bonding_donors"] = hbd
            update_data["user_id"] = user_id  # get user

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
                if len(update_data) > 1:  # at least 1 preference
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
        return redirect(url_for("onboarding"))
    return render_template("onboarding.html")


@app.route("/matching")
def matching():
    guard = require_login()
    if guard:
        return guard

    user_id = get_user_id()

    if "index" not in session:
        session["index"] = 0
        session["accepted"] = []

    idx = session["index"]

    if idx >= len(compounds):
        return render_template("matching.html", done=True)

    compound = compounds[idx]

    if request.method == "POST":
        action = request.form.get("action")

        if action == "like":
            session["accepted"].append(compound)

            properties = {
                "molecular_weight": compound.get("Molecular Weight"),
                "lipophilicity": compound.get("AlogP"),
                "hydrogen_bonding_acceptors": compound.get("HBA"),
                "hydrogen_bonding_donors": compound.get("HBD"),
            }

            try:
                save_match(
                    sb_user(),
                    user_id,
                    compound.get("Name"),
                    compound.get("Smiles"),
                    properties,
                )
            except Exception as e:
                print("Error saving match:", e)

        session["index"] += 1

        return redirect(url_for("matching"))

    return render_template(
        "matching.html",
        name=compound.get("Name"),
        smiles=compound.get("Smiles"),
        molecular_weight=compound.get("Molecular Weight"),
        lipophilicity=compound.get("AlogP"),
        hba=compound.get("HBA"),
        hbd=compound.get("HBD"),
        done=False,
    )


@app.route("/api/match", methods=["POST"])
def api_match():
    guard = require_login()
    if guard:
        return guard

    idx = session.get("index", 0)

    if idx >= len(compounds):
        return jsonify({"done": True})

    session["index"] = idx + 1

    next_compound = compounds[idx]
    return jsonify(
        {
            "done": False,
            "name": next_compound["Name"],
            "smiles": next_compound["Smiles"],
        }
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
    return redirect(url_for("onboarding"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
