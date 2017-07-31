# coding=utf-8
from flask import Flask, request

from functions import text_recog

app = Flask(__name__)


@app.route('/')
def main_form():
    return """
        <head>JOAOZINHO</head>
        <body>
            <form action="submit" id="textform" method="post">
                    <textarea name="text" rows="20" cols="70"></textarea>
                    </br>
                <input type="submit" value="Classificar">
            </form>
        </body>
        """


@app.route('/submit', methods=['POST'])
def submit_textarea():
    return text_recog(request.form['text'])

if __name__ == '__main__':
    app.run()