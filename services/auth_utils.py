
from functools import wraps

from flask import jsonify, redirect, request, session, url_for


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({
                    "success": False,
                    "message": "Chưa đăng nhập."
                }), 401
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper
