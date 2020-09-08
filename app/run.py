import json
import plotly

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Prepare the data for the first graph
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    ## Prepare the data for the second graph
    ## Distribution of the message category

    #selection of the message category columns only   
    df_columns_total  = df.iloc[:,4:]
    # calculation of the total counts per column
    my_sum = pd.DataFrame(df_columns_total.sum())
    my_sum = my_sum.reset_index()
    my_sum.columns = ['message', 'number']
    my_sum.sort_values('number', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
    x_val = my_sum['message'].values
    y_val = my_sum['number'].values

    ## prepare the data for the third graph
    ## Check in how many genres each meassage category is
    df_grouped  = df.groupby('genre').sum()
    df_grouped_tr = df_grouped.transpose().reset_index()
    df_grouped_tr.columns = ['category','direct','news','social']
    
    
    y_count1 = df_grouped_tr.groupby('category').sum()['direct']
    y_count2 = df_grouped_tr.groupby('category').sum()['news']
    y_count3 = df_grouped_tr.groupby('category').sum()['social']
    x_value = list(y_count1.index)
   
    

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_val,
                    y=y_val
                )
            ],

            'layout': {
                
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': " Messages Count",
                    'titlefont' :{'size':13},
                    
                },
                'xaxis': {
                    'title': "Message Cat",
                    'titlefont' :{'size':13},
                    'tickfont'  :{'size':8}
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_value,
                    y=y_count1,
                    name='direct'
                ),
                Bar(
                    x=x_value,
                    y=y_count2,
                    name='news'
                ),
                Bar(
                    x=x_value,
                    y=y_count3,
                    name='social'
                )
            ],

            'layout': {
                
                'title': 'Message Genre per Message Category',
                'yaxis': {
                    'title': " Messages Count",
                    'titlefont' :{'size':13},
                    
                },
                'xaxis': {
                    'title': "Message Category",
                    'titlefont' :{'size':13},
                    'tickfont'  :{'size':8}
                }
            }
        }
    ]
    


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
 
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()