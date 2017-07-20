from flask import Flask, render_template, request, redirect, make_response

app = Flask(__name__)

app.vars = {}

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        product = request.form.get('product')

        import datetime
        from bokeh.plotting import figure
        from bokeh.embed import components
        from api_data import get_data
        from topic_model_svd import get_reviews

        path='./data/iphone6.csv'
        reviews = get_reviews(path)

        return render_template('plot.html', reviews=reviews)


if __name__ == '__main__':
    app.run(debug=True)
