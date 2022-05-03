import dash
from dash import dcc,html
import pickle
import json
from dash.dependencies import Input, Output, State

########### Define your variables ######
myheading1 = 'Predicting Glass Type'
myheading2 = 'Model to predict whether the type of glass is suitable as window glass based on input features'
myheading3 = 'Features are Element Composition, Refractive Index and Threshold percentage'
tabtitle = 'Glass Type Prediction'
sourceurl = 'https://git.generalassemb.ly/intuit-ds-15/09-logistic-regression-classifier/tree/master/data'
githublink = 'https://github.com/vharkar/503-log-reg-loans-simple'

########### open the json file ######
with open('assets/rocaucglass.json', 'r') as f:
    fig=json.load(f)

########### open the pickle file ######
filename = open('analysis/glass_type_model.pkl', 'rb')
unpickled_model = pickle.load(filename)
filename.close()

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),
    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),

    html.Div([
        html.Div([
                html.H3(myheading2),
                html.H4(myheading3),
                dcc.Graph(figure=fig, id='fig1')
            ], className='five columns'),
        html.Div([
                html.H4("Features"),
                html.Div('Refractive Index:'),
                dcc.Input(id='RI', value=1.50, type='number', min=1.50, max=1.54, step=0.05),
                html.Div('Sodium (in weight percent):'),
                dcc.Input(id='Na', value=13.40, type='number', min=10.75, max=17.50, step=0.05),
                html.Div('Magnesium (in weight percent):'),
                dcc.Input(id='Mg', value=2.70, type='number', min=0.00, max=4.50, step=0.05),
                html.Div('Aluminium (in weight percent):'),
                dcc.Input(id='Al', value=1.45, type='number', min=0.30, max=3.50, step=0.05),
                html.Div('Silicon (in weight percent):'),
                dcc.Input(id='Si',value=72.65, type='number', min=69.80, max=75.40, step=0.05),
                html.Div('Potassium (in weight percent):'),
                dcc.Input(id='K', value=0.55, type='number', min=0.00, max=6.20, step=0.05),
                html.Div('Calcium (in weight percent):'),
                dcc.Input(id='Ca', value=8.95, type='number', min=5.45, max=16.20, step=0.05),
                html.Div('Barium (in weight percent):'),
                dcc.Input(id='Ba', value=0.15, type='number', min=0.00, max=3.15, step=0.05),
                html.Div('Iron (in weight percent):'),
                dcc.Input(id='Fe', value=0.05, type='number', min=0.00, max=0.50, step=0.05),
                html.Div('Threshold:'),
                dcc.Input(id='Threshold', value=80, type='number', min=50, max=90, step=5),

            ], className='six columns'),
            html.Div([
                html.H3('Predictions'),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
                html.Br(),
                html.Div('Probability of Window Glass:'),
                html.Div(id='YesProb'),
                html.Br(),
                html.Div('Probability of Not Window Glass:'),
                html.Div(id='NoProb')
            ], className='three columns')
        ], className='twelve columns',
    )]
)


######### Define Callback
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
     Output(component_id='YesProb', component_property='children'),
     Output(component_id='NoProb', component_property='children'),
    ],
    [Input(component_id='RI', component_property='value'),
     Input(component_id='Na', component_property='value'),
     Input(component_id='Mg', component_property='value'),
     Input(component_id='Al', component_property='value'),
     Input(component_id='Si', component_property='value'),
     Input(component_id='K', component_property='value'),
     Input(component_id='Ca', component_property='value'),
     Input(component_id='Ba', component_property='value'),
     Input(component_id='Fe', component_property='value'),
     Input(component_id='Threshold', component_property='value')
    ])
def prediction_function(RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Threshold):
    try:
        data = [[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]]
        rawprob= 100 * unpickled_model.predict_proba(data)[0][1]
        func = lambda y: 'WindowGlass' if int(rawprob) > Threshold else 'NotWindowGlass'
        formatted_y = func(rawprob)
        yes_prob = unpickled_model.predict_proba(data)[0][1] * 100
        no_prob = unpickled_model.predict_proba(data)[0][0] * 100
        formatted_yes_prob = "{:,.2f}%".format(yes_prob)
        formatted_no_prob = "{:,.2f}%".format(no_prob)
        return formatted_y, formatted_yes_prob, formatted_no_prob
    except:
        return "inadequate inputs", "inadequate inputs"

############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
