from flask import Blueprint, redirect, render_template, session, url_for

from services.auth_utils import login_required

page_bp = Blueprint("pages", __name__)


@page_bp.route("/")
def home():
    return redirect(
    url_for("pages.dashboard" if "user_id" in session else "auth.login")
)

@page_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@page_bp.route("/history")
@login_required
def history_page():
    return render_template("history.html")


@page_bp.route("/drivers")
@login_required
def drivers_page():
    return render_template("drivers.html")


@page_bp.route("/verification")
@login_required
def verification_page():
    return render_template("verification.html")
