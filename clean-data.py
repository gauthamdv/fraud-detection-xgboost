from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        amount = request.form.get("amount")
        result = f"Received amount: {amount}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
