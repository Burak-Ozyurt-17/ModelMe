import sqlite3, os, secrets, shutil, time,json

from datetime import datetime
from flask import Flask, render_template, request, redirect, session,jsonify,send_file
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import error, login_required

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


CATEGORIES = ["Image Classification",
    "Object Detection",
    "Semantic Segmentation",
    "Pose Estimation",
    "OCR / Text Detection",
    "Style Transfer",
    "Background Removal",
    "Face Recognition",
    "Custom Vision Task"
]

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")


@app.errorhandler(500)
def not_found(e):
    return render_template("500.html")


@app.errorhandler(405)
def not_found(e):
    return render_template("405.html")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/homepage")
@login_required
def homepage():
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        examples = db.execute("SELECT title, color,model_hash FROM examples ORDER BY model_id").fetchall()
        models = db.execute(
            "SELECT * FROM models WHERE user_id = ?",
            (session.get("user_id"),)
        ).fetchall()
        published_models = db.execute(
            "SELECT * FROM published_models WHERE user_id = ?",
            (session.get("user_id"),)
        ).fetchall()
    return render_template("homepage.html", examples=examples, models=models,published_models=published_models)


@app.route("/register", methods=["GET", "POST"])
def register():
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        
        if not username:
            return error("Missing Username")
        elif not password:
            return error("Missing Password")
        elif not confirmation:
            return error("Missing Confirmation")
        elif confirmation != password:
            return error("Unmatching Password Confirmation")

        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            db = con.cursor()
            
            # Fixed: Properly check if username exists
            existing_user = db.execute("SELECT username FROM users WHERE username = ?", (username,)).fetchone()
            if existing_user:
                return error("Username has already been taken")

            try:
                db.execute("INSERT INTO users (username,hash) VALUES(?,?)",
                           (username, generate_password_hash(password)))
                con.commit()
                
                user = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
                session["user_id"] = user["id"]
                
            except sqlite3.IntegrityError:
                return error("Username has already been taken")

        return redirect("/homepage")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if request.method == "POST":
        if not request.form.get("username"):
            return error("Missing Username")
        elif not request.form.get("password"):
            return error("Missing Password")
            
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            db = con.cursor()

            row = db.execute(
                "SELECT * FROM users WHERE username = ?",
                (request.form.get("username"),)
            ).fetchone()

            if row is None or not check_password_hash(row["hash"], request.form.get("password")):
                return error("Invalid username and/or password")
                
            session["user_id"] = row["id"]

        return redirect("/homepage")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/search")
@login_required
def search():
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        search_term = request.args.get("model")
        like_pattern = f"%{search_term}%"
        MODELS = db.execute(
            "SELECT * FROM published_models JOIN users ON published_models.user_id = users.id WHERE title LIKE ? OR description LIKE ? OR category LIKE ? OR username LIKE ?",
            (like_pattern, like_pattern, like_pattern, like_pattern)
        ).fetchall()
    return render_template("search.html", models=MODELS, args=request.args.get("model"))

@app.route("/profile", methods=["POST", "GET"])
@login_required
def profile():
    user_id = session.get("user_id")
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        row = db.execute(
            "SELECT username FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        models = db.execute(
            "SELECT * FROM published_models WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return render_template("profile.html", row=row, models=models)

@app.route("/about")
@login_required
def about():
    return render_template("about.html")


def create_model(db, con, name, user_id):
    while True:
        model_id = secrets.token_hex(8)
        exists = db.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if not exists:
            break

    default_description = "No description provided"
    default_category = "Uncategorized"

    db.execute("INSERT INTO models (user_id, model_id, title, description, category, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
           (user_id, model_id, name, default_description, default_category, datetime.now()))
    con.commit()
    return model_id


@app.route("/train", methods=["POST"])
@login_required
def train():
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()

        name = request.form.get("projectname")
        if name:
            model_id = create_model(db, con, name, session.get("user_id"))
            db.execute("INSERT INTO classes (model_id,class_name,class_number) VALUES(?,?,?)",           
                (model_id,"Class 1",1,))
            con.commit()
            return redirect(f"/model/{model_id}")   
        return redirect("/homepage")

def verify_model_ownership(model_id, user_id):
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        model = db.execute(
            "SELECT user_id FROM models WHERE model_id = ?",
            (model_id,)
        ).fetchone()
        return model and model["user_id"] == user_id

@app.route("/model/<model_id>",methods=["POST","GET"])
@login_required
def view_model(model_id):
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        model = db.execute(
            "SELECT * FROM models WHERE model_id = ?",
            (model_id,)
        ).fetchone()
        published_model = db.execute(
            "SELECT * FROM published_models WHERE model_id = ?",
            (model_id,)
        ).fetchone()
        classes = db.execute("SELECT * FROM classes WHERE model_id = ?",(model_id,)).fetchall()
        if not model and not published_model:
            return error("Model not found")
        if model and model["user_id"] == session.get("user_id"):
            return render_template("train.html", 
                                 categories=CATEGORIES, 
                                 model_id=model_id, 
                                 model=model,
                                 classes=classes,
                                 )
        else:
            creator = db.execute(
                "SELECT username FROM users WHERE id = (SELECT user_id FROM published_models WHERE model_id = ?)",
                (model_id,)
            ).fetchone()[0]
            return render_template("view.html", model=published_model or model, model_id = model_id ,creator = creator)
    
@app.route("/view/<model_id>")
@login_required
def only_view_model(model_id):
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        model = db.execute(
            "SELECT * FROM models WHERE model_id = ?",
            (model_id,)
        ).fetchone()
        published_model = db.execute(
            "SELECT * FROM published_models WHERE model_id = ?",
            (model_id,)
        ).fetchone()
        isTrained  = db.execute(
            "SELECT trained FROM models WHERE model_id = (?)",
            (model_id,)
        ).fetchone()[0]
        if not model and not published_model:
            return error("Model not found")
        if isTrained != 0:
            return redirect(f"/model/{model_id}")
        return error("Model file not found")
        
@app.route("/example/<model_hash>")
@login_required
def view_example(model_hash):
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        example = db.execute(
            "SELECT * FROM examples WHERE model_hash = ?",
            (model_hash,)
        ).fetchone()
        if not example:
            return error("Example not found")
    return render_template("view.html", model=example,model_id=model_hash,creator = "ModelMe")

@app.route("/delete/<model_id>",methods=["POST","GET"])
@login_required
def delete_model(model_id):
    if not verify_model_ownership(model_id, session.get("user_id")):
        return error("Unauthorized")
        
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        db.execute(
            "DELETE FROM models WHERE model_id = ?",
            (model_id,)
        )
        db.execute(
            "DELETE FROM published_models WHERE model_id = ?",
            (model_id,)
        )
        db.execute(
            "DELETE FROM classes WHERE model_id = ?",
            (model_id,)
        )
        con.commit()
        for _ in range(5):
                try:
                    shutil.rmtree(f"static/models/{model_id}")
                    break
                except (PermissionError,FileNotFoundError):
                    time.sleep(0.2)
        return redirect("/homepage")


@app.route("/publish/<model_id>",methods=["POST","GET"])
@login_required
def publish_model(model_id):
    if not verify_model_ownership(model_id, session.get("user_id")):
        return error("Unauthorized")
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        db.execute(
            "INSERT INTO published_models (user_id,model_id,title,description,category,timestamp) VALUES(?,?,?,?,?,?) ",
            (session.get("user_id"),model_id,request.form.get("projectname"),request.form.get("description"),request.form.get("category"),datetime.now()),
        )
        con.commit()
    return redirect("/homepage")

@app.route("/add_class/<model_id>",methods=["POST"])
@login_required
def add_class(model_id):
    with sqlite3.connect("database.db") as con:
        if request.headers.get("Content-Type") != "application/json":
            error(404)
        con.row_factory = sqlite3.Row
        db = con.cursor()
        result = db.execute("SELECT MAX(class_number) FROM classes WHERE model_id = ?", (model_id,)).fetchone()
        max_num = result[0] if result and result[0] is not None else 0
        default_num = int(max_num +1)
        default_name = "Class " + str(max_num + 1)
        db.execute("INSERT INTO classes (model_id,class_name,class_number) VALUES(?,?,?)",
        (model_id,default_name,default_num))
        con.commit()
    return jsonify(class_number=default_num,class_name=default_name)

@app.route("/delete_class/<model_id>/<class_number>",methods=["POST","GET"])
@login_required
def delete_class(model_id,class_number):
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        count = db.execute("SELECT COUNT(*) FROM classes WHERE model_id = ?",(model_id,)).fetchone()[0]
        if (count != 1):
            db.execute("DELETE FROM classes WHERE model_id = ? AND class_number=?",
            (model_id,class_number))
            con.commit()
    return jsonify(class_number = class_number)

@app.route("/rename_class/<model_id>/<int:class_number>/<class_name>",methods=["GET","POST"])
@login_required
def rename_class(model_id,class_number,class_name):
    if not verify_model_ownership(model_id, session.get("user_id")):
        return error("Unauthorized")
    with sqlite3.connect("database.db") as con:
        con.row_factory = sqlite3.Row
        db = con.cursor()
        db.execute("UPDATE classes SET class_name = ? WHERE model_id = ? AND class_number = ?",
            (class_name,model_id,class_number))
        con.commit()
    return jsonify(class_name = class_name,class_number=class_number)

@app.route("/save_model/<model_id>", methods=["POST"])
def save_model(model_id):
    os.makedirs(f"static/models/{model_id}", exist_ok=True)
    for f in request.files.values():
        f.save(os.path.join(f"static/models/{model_id}", f.filename))
    with sqlite3.connect("database.db") as con:
        db=con.cursor()
        db.execute("UPDATE models SET trained = 1 WHERE model_id = ?",(model_id,))
        con.commit()
    return {"status": "ok"}

@app.route("/model_exists/<model_id>")
def model_exists(model_id):
    import os
    model_path = f"static/models/{model_id}/model.json"
    return {"exists": os.path.exists(model_path)}

@app.route("/reset_model/<model_id>")
def reset_model(model_id):
    try:
        path = f"static/models/{model_id}"
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    return {"status":"success"}

@app.route("/save_class_names/<model_id>", methods=['POST'])
def save_class_names(model_id):
    names = request.get_json()
    path = os.path.join("static/models", model_id, "class_names.json")
    with open(path, "w") as f:
        json.dump(names, f)
    return {"status": "ok"}

@app.route("/models/<model_id>/class_names.json")
def get_class_names(model_id):
    path = os.path.join("static/models", model_id, "class_names.json")
    if os.path.exists(path):
        return send_file(path, mimetype="application/json")
    else:
        return {"error": "not found"}, 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
