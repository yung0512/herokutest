from flask import Flask
app = Flask(_name_)

@app.route("/")
def home():
    return "Hello Flask 2"

@app.route("/test")
def test():
    return "This is Test"

if _name_ =="__main__":
    app.run()