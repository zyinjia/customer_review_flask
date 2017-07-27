from flask import Flask, render_template, request, redirect, make_response, json


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

        from bokeh.embed import components
        from svd_plot import plot_bokeh, get_plotsvd_data
        import pandas as pd

        path='./html_data/iphone6.csv'
        df = pd.read_csv(path, encoding='utf-8')
        df = df.rename(columns={'Unnamed: 0':'TopicIndex'})
        topics = df.T.to_dict().values()
        for item in topics:
            item['Reviews'] = [ review for review in item['Reviews'].split("\n\n")]

        svd_topic, svd_s = get_plotsvd_data('./html_data/iphone6.pickle' )
        p = plot_bokeh(svd_topic, svd_s)
        plot_script, plot_div = components(p)

        return render_template('plot.html',
                                product = product,
                                topics=topics,
                                plot_script=plot_script,
                                plot_div=plot_div )

if __name__ == '__main__':
    app.run(debug=True)
