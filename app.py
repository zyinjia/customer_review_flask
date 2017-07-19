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
        app.vars['name'] = request.form['ticker']

        import datetime
        from bokeh.plotting import figure
        from bokeh.embed import components
        from api_data import get_data

        df = get_data(app.vars['name'])
        p = figure(title='Data from Quandle WIKI set',
                   x_axis_label='Date',
                   x_axis_type='datetime',
                   y_axis_label='Open Price ($)',
                   toolbar_sticky=False)
        p.circle(df['datetime'], df['open'])
        script, div = components(p)

        return render_template('plot.html', script=script, div=div, ticker=app.vars['name'])


if __name__ == '__main__':
    app.run(debug=True)
