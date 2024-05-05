from dotenv import load_dotenv
load_dotenv(dotenv_path='config.env')
from flask import Flask, render_template, request, redirect, url_for
from RAG.RAG import get_RAG_model


import logging

app = Flask(__name__)
rag = get_RAG_model()
app.logger.setLevel(logging.DEBUG)
@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        if 'search' in request.form:

            question = request.form['question']
            answer = str(rag.find_documents(question))
            return render_template('index.html', question=question, answer=answer)
        elif 'clear' in request.form:

            return redirect(url_for('index'))
    return render_template('index.html', question='', answer='')

if __name__ == '__main__':
    app.run(debug=True)
