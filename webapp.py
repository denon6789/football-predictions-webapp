import pandas as pd
from flask import Flask, render_template_string
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Football Predictions</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8f9fa; }
        .container { margin-top: 40px; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="mb-4">Football Match Predictions</h2>
    {% if table_html %}
        {{ table_html|safe }}
    {% else %}
        <div class="alert alert-warning">No predictions available.</div>
    {% endif %}
</div>
</body>
</html>
'''

@app.route("/")
def show_predictions():
    # Try to load predictions from the most recent file
    # For demo, use the last raw_matches.csv or allow user to specify a file
    csv_file = "raw_matches.csv"
    if not os.path.exists(csv_file):
        return render_template_string(HTML_TEMPLATE, table_html=None)
    df = pd.read_csv(csv_file)
    # Show only relevant columns if present
    columns = [c for c in ['date','home_team','home_score','away_score','away_team','result'] if c in df.columns]
    if not columns:
        table_html = df.to_html(classes="table table-striped", index=False)
    else:
        table_html = df[columns].to_html(classes="table table-striped", index=False)
    return render_template_string(HTML_TEMPLATE, table_html=table_html)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
