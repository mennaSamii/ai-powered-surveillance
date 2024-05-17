from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Assuming you have an index.html file

if __name__ == '__main__':
    app.run(debug=True)