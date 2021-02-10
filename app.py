from flask import Flask
from flask import  render_template_string


app = Flask(__name__)

with open('lda_vis.html', 'r') as f:
    li = str(f.read())


@app.route('/')
def home():
    return render_template_string(li)


if __name__ == "__main__":
    app.run()
