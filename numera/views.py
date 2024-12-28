from django.shortcuts import render
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import bernoulli, binom, uniform, poisson, expon, norm

# Page d'accueil avec statistiques
def home(request):
    stats = {}
    columns = []
    numeric_columns = []
    categorical_columns = []
    result = None
    operation = None
    column = None

    # Charger le fichier CSV
    if 'file_data' in request.session:
        file_data = base64.b64decode(request.session['file_data'])
        df = pd.read_csv(io.BytesIO(file_data))
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
        stats = {
            'mean': df.mean(numeric_only=True).to_dict(),
            'median': df.median(numeric_only=True).to_dict(),
            'mode': df.mode().iloc[0].to_dict(),
            'std': df.std(numeric_only=True).to_dict(),
            'variance': df.var(numeric_only=True).to_dict(),
            'range': {col: (df[col].max() - df[col].min()) for col in df.select_dtypes(include=['number']).columns}
        }

    if request.method == "POST":

        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            if uploaded_file.name.endswith('.csv'):
                file_data = uploaded_file.read()
                request.session['file_data'] = base64.b64encode(file_data).decode('utf-8')
                df = pd.read_csv(io.BytesIO(file_data))
                columns = df.columns.tolist()
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
                stats = {
                    'mean': df.mean(numeric_only=True).to_dict(), 
                    'median': df.median(numeric_only=True).to_dict(),
                    'mode': df.mode().iloc[0].to_dict(),
                    'std': df.std(numeric_only=True).to_dict(),
                    'variance': df.var(numeric_only=True).to_dict(),
                    'range': {col: (df[col].max() - df[col].min()) for col in df.select_dtypes(include=['number']).columns}
                }

        # Traitement des opérations sur une colonne
        elif 'operation' in request.POST and 'column' in request.POST:
            operation = request.POST['operation']
            column = request.POST['column']

            if 'file_data' in request.session:
                file_data = base64.b64decode(request.session['file_data'])
                df = pd.read_csv(io.BytesIO(file_data))
                if column in df.columns:
                    if operation == "mean":
                        result = df[column].mean()
                    elif operation == "median":
                        result = df[column].median()
                    elif operation == "mode":
                        result = df[column].mode().iloc[0]
                    elif operation == "std":
                        result = df[column].std()
                    elif operation == "variance":
                        result = df[column].var()
                    elif operation == "range":
                        result = df[column].max() - df[column].min()

    return render(request, 'numera/home.html', {
        'stats': stats,
        'columns': columns,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'result': result,
        'operation': operation,
        'column': column,
    })

# Page de visualisation
def visualisation(request):
    plot_url = None
    columns = []
    numeric_columns = []
    categorical_columns = []

    if 'file_data' in request.session:
        file_data = base64.b64decode(request.session['file_data'])
        df = pd.read_csv(io.BytesIO(file_data))
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

        if request.method == "POST":
            column = request.POST['column']
            plot_type = request.POST['plot_type']

            plt.figure(figsize=(10, 6))
            if plot_type == "bar":
                df[column].value_counts().plot(kind="bar", color="skyblue")
            elif plot_type == "kde":
                df[column].plot(kind="kde", color="orange")
            elif plot_type == "hist":
                df[column].plot(kind="hist", bins=20, color="green")
            elif plot_type == "line":
                df[column].plot(kind="line", color="purple")
            elif plot_type == "box":
                df.boxplot(column=[column])
            elif plot_type == "scatter" and len(numeric_columns) > 1:
                sns.scatterplot(data=df, x=column, y=numeric_columns[0], palette="viridis")
            elif plot_type == "heatmap":
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            elif plot_type == "pie":
                df[column].value_counts().plot(kind="pie", autopct='%1.1f%%', colors=sns.color_palette("pastel"))

            plt.title(f"{plot_type.capitalize()} plot for {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            # Sauvegarder le graphique
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

    return render(request, 'numera/visualisation.html', {
        'columns': columns,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'plot_url': plot_url,
    })

# Page des distributions
def distribution(request):
    plot_url = None
    message = None
    columns = []
    numeric_columns = []

    # Vérifier si un fichier est déjà chargé
    if 'file_data' in request.session:
        file_data = base64.b64decode(request.session['file_data'])
        df = pd.read_csv(io.BytesIO(file_data))
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        if request.method == "POST":
            column = request.POST.get('column', None)
            distribution_type = request.POST.get('distribution_type', '')
            params = {
                'n': int(request.POST.get('n', 10)),
                'p': float(request.POST.get('p', 0.5)),
                'lambda': float(request.POST.get('lambda', 1.0)),
                'a': float(request.POST.get('a', 0)),
                'b': float(request.POST.get('b', 1))
            }

            if column and column in numeric_columns:
                data = df[column].dropna()

                x = np.linspace(min(data), max(data), 500)

                plt.figure(figsize=(10, 6))
                if distribution_type == "bernoulli":
                    rv = bernoulli(params['p'])
                    plt.bar([0, 1], rv.pmf([0, 1]), color='skyblue')
                elif distribution_type == "binomial":
                    rv = binom(params['n'], params['p'])
                    plt.bar(range(params['n'] + 1), rv.pmf(range(params['n'] + 1)), color='skyblue')
                elif distribution_type == "poisson":
                    rv = poisson(params['lambda'])
                    plt.bar(range(20), rv.pmf(range(20)), color='skyblue')
                elif distribution_type == "uniform":
                    plt.plot(x, uniform.pdf(x, loc=params['a'], scale=params['b'] - params['a']), color='green')
                elif distribution_type == "exponential":
                    plt.plot(x, expon.pdf(x, scale=1 / params['lambda']), color='orange')
                elif distribution_type == "normal":
                    plt.plot(x, norm.pdf(x, loc=data.mean(), scale=data.std()), color='purple')

                plt.title(f"Distribution: {distribution_type.capitalize()} pour {column}")
                plt.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
            else:
                message = "Veuillez sélectionner une colonne numérique valide."
    else:
        message = "Veuillez charger un fichier CSV avant de procéder."

    return render(request, 'numera/distribution.html', {
        'columns': columns,
        'numeric_columns': numeric_columns,
        'plot_url': plot_url,
        'message': message,
    })
