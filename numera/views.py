from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import json
from scipy.stats import bernoulli, binom, uniform, poisson, expon, norm
from django.http import JsonResponse
from .forms import CustomUserCreationForm
from .models import UserActivity, DatasetUpload

# Authentication Views
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def profile_view(request):
    return render(request, 'registration/profile.html')


# Dashbord 
@login_required
def dashboard(request):
    # Get user activity stats
    user_stats = {
        'visualizations_count': UserActivity.objects.filter(
            user=request.user, 
            activity_type='visualization'
        ).count(),
        'distributions_count': UserActivity.objects.filter(
            user=request.user, 
            activity_type='distribution'
        ).count(),
        'datasets_count': DatasetUpload.objects.filter(
            user=request.user
        ).count(),
        'recent_activities': UserActivity.objects.filter(
            user=request.user
        ).order_by('-timestamp')[:5],
    }

    return render(request, 'numera/dashboard.html', user_stats)

# Page d'accueil avec statistiques
@login_required
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
                  # Record dataset upload
                DatasetUpload.objects.create(
                    user=request.user,
                    filename=uploaded_file.name
                )
                UserActivity.objects.create(
                    user=request.user,
                    activity_type='upload',
                    description=f'A importé le fichier {uploaded_file.name}'
                )
                # the rest of the page 
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
@login_required
def visualisation(request):
    plot_url = None
    columns = []
    numeric_columns = []
    categorical_columns = []
    plot_type = None
    column = None

    if 'file_data' in request.session:
        file_data = base64.b64decode(request.session['file_data'])
        df = pd.read_csv(io.BytesIO(file_data))
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

        if request.method == "POST" and 'column' in request.POST and 'plot_type' in request.POST:
            column = request.POST['column']
            plot_type = request.POST['plot_type']

            # Record visualization activity
            UserActivity.objects.create(
                user=request.user,
                activity_type='visualization',
                description=f'A créé un graphique {plot_type} pour {column}'
            )

            try:
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

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
            finally:
                plt.close('all')

    return render(request, 'numera/visualisation.html', {
        'columns': columns,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'plot_url': plot_url,
        'plot_type': plot_type,
        'selected_column': column
    })
# Page des distributions
@login_required
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
            # Record distribution activity
                UserActivity.objects.create(
                user=request.user,
                activity_type='distribution',
                description=f'A analysé la distribution {distribution_type} pour {column}'
            )    
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


# Profile Statistics
@login_required
def profile_view(request):
    context = {
        'visualizations_count': UserActivity.objects.filter(
            user=request.user, 
            activity_type='visualization'
        ).count(),
        'distributions_count': UserActivity.objects.filter(
            user=request.user, 
            activity_type='distribution'
        ).count(),
        'datasets_count': DatasetUpload.objects.filter(
            user=request.user
        ).count(),
        'recent_activities': UserActivity.objects.filter(
            user=request.user
        ).order_by('-timestamp')[:5],
    }
    return render(request, 'registration/profile.html', context)



# destribution2
@login_required
def distribution2(request):
    plot_url = None
    message = None
    columns = []
    numeric_columns = []

    if 'file_data' in request.session:
        file_data = base64.b64decode(request.session['file_data'])
        df = pd.read_csv(io.BytesIO(file_data))
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        if request.method == "POST":
            column = request.POST.get('column', None)
            distribution_type = request.POST.get('distribution_type', '')
            
            # Get distribution parameters
            params = {
                'n': int(request.POST.get('n', 10)),
                'p': float(request.POST.get('p', 0.5)),
                'lambda': float(request.POST.get('lambda', 1.0)),
                'a': float(request.POST.get('a', 0)),
                'b': float(request.POST.get('b', 1))
            }

            if column and column in numeric_columns:
                try:
                    # Record analysis activity
                    UserActivity.objects.create(
                        user=request.user,
                        activity_type='distribution',
                        description=f'Analyse de distribution {distribution_type} pour {column}'
                    )

                    # Get data and plot
                    data = df[column].dropna()
                    plt.figure(figsize=(10, 6))
                    
                    # Plot histogram of actual data
                    plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='Données')

                    # Plot theoretical distribution
                    x = np.linspace(min(data), max(data), 500)
                    if distribution_type == "normal":
                        mu, sigma = data.mean(), data.std()
                        plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label=f'Normale(μ={mu:.2f}, σ={sigma:.2f})')
                    
                    elif distribution_type == "uniform":
                        plt.plot(x, uniform.pdf(x, params['a'], params['b'] - params['a']), 
                               'r-', label=f'Uniforme(a={params["a"]}, b={params["b"]})')
                    
                    elif distribution_type == "exponential":
                        plt.plot(x, expon.pdf(x, scale=1/params['lambda']), 
                               'r-', label=f'Exponentielle(λ={params["lambda"]})')
                    
                    elif distribution_type == "poisson":
                        x_poisson = np.arange(0, max(20, int(data.max()) + 1))
                        plt.bar(x_poisson, poisson.pmf(x_poisson, params['lambda']), 
                               alpha=0.7, color='red', label=f'Poisson(λ={params["lambda"]})')
                    
                    elif distribution_type == "binomial":
                        x_binom = np.arange(0, params['n'] + 1)
                        plt.bar(x_binom, binom.pmf(x_binom, params['n'], params['p']), 
                               alpha=0.7, color='red', 
                               label=f'Binomiale(n={params["n"]}, p={params["p"]})')
                    
                    elif distribution_type == "bernoulli":
                        x_bern = [0, 1]
                        plt.bar(x_bern, [1-params['p'], params['p']], 
                               alpha=0.7, color='red', 
                               label=f'Bernoulli(p={params["p"]})')

                    plt.title(f'Distribution {distribution_type} - {column}')
                    plt.xlabel('Valeurs')
                    plt.ylabel('Densité')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save plot
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
                    buf.close()

                except Exception as e:
                    message = f"Erreur lors de l'analyse: {str(e)}"
            else:
                message = "Veuillez sélectionner une colonne numérique valide."
    else:
        message = "Veuillez d'abord charger un fichier CSV."

    return render(request, 'numera/distribution.html', {
        'columns': columns,
        'numeric_columns': numeric_columns,
        'plot_url': plot_url,
        'message': message
    })
    
    #visualisation2 
@login_required
def visualisation2(request):
    PLOT_TYPES = {
        'scatter': {
            'name': 'Nuage de Points',
            'icon': 'braille',
            'min_cols': 2,
            'max_cols': 2,
            'description': 'Relation entre deux variables'
        },
        'bar': {
            'name': 'Diagramme en Barres',
            'icon': 'chart-bar',
            'min_cols': 1,
            'max_cols': 5,
            'description': 'Comparaison de valeurs'
        },
        'line': {
            'name': 'Graphique Linéaire',
            'icon': 'chart-line',
            'min_cols': 1,
            'max_cols': 5,
            'description': 'Évolution temporelle'
        },
        'box': {
            'name': 'Boîte à Moustaches',
            'icon': 'box',
            'min_cols': 1,
            'max_cols': 5,
            'description': 'Distribution et outliers'
        },
        'histogram': {
            'name': 'Histogramme',
            'icon': 'chart-bar',
            'min_cols': 1,
            'max_cols': 1,
            'description': 'Distribution des données'
        },
        'kde': {
            'name': 'Densité (KDE)',
            'icon': 'chart-area',
            'min_cols': 1,
            'max_cols': 2,
            'description': 'Distribution continue'
        },
        'violin': {
            'name': 'Violin Plot',
            'icon': 'wave-square',
            'min_cols': 1,
            'max_cols': 3,
            'description': 'Distribution par groupe'
        },
        'correlation': {
            'name': 'Corrélation',
            'icon': 'th',
            'min_cols': 2,
            'max_cols': 10,
            'description': 'Relations entre variables'
        },
        'heatmap': {
            'name': 'Heatmap',
            'icon': 'th-large',
            'min_cols': 2,
            'max_cols': 10,
            'description': 'Motifs dans les données'
        },
        'pair': {
            'name': 'Pair Plot',
            'icon': 'border-all',
            'min_cols': 2,
            'max_cols': 5,
            'description': 'Relations multivariées'
        }
    }

    if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            plot_type = request.POST.get('plot_type')
            columns = request.POST.getlist('columns')
            
            if not plot_type in PLOT_TYPES:
                return JsonResponse({'success': False, 'error': 'Type de graphique invalide'})
            
            plot_config = PLOT_TYPES[plot_type]
            if not (plot_config['min_cols'] <= len(columns) <= plot_config['max_cols']):
                return JsonResponse({
                    'success': False, 
                    'error': f"Sélectionnez entre {plot_config['min_cols']} et {plot_config['max_cols']} colonnes"
                })

            df = pd.read_csv(io.BytesIO(base64.b64decode(request.session['file_data'])))
            
            plt.switch_backend('Agg')
            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                if plot_type == 'scatter':
                    sns.scatterplot(data=df, x=columns[0], y=columns[1], ax=ax)
                elif plot_type == 'bar':
                    df[columns].plot(kind='bar', ax=ax)
                elif plot_type == 'line':
                    df[columns].plot(kind='line', ax=ax)
                elif plot_type == 'box':
                    df[columns].boxplot(ax=ax)
                elif plot_type == 'histogram':
                    df[columns[0]].hist(bins=30, ax=ax)
                elif plot_type == 'kde':
                    sns.kdeplot(data=df[columns], ax=ax)
                elif plot_type == 'violin':
                    sns.violinplot(data=df[columns], ax=ax)
                elif plot_type == 'correlation':
                    sns.heatmap(df[columns].corr(), annot=True, ax=ax)
                elif plot_type == 'heatmap':
                    sns.heatmap(df[columns], ax=ax)
                elif plot_type == 'pair':
                    plt.close()
                    g = sns.pairplot(df[columns])
                    fig = g.fig

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close('all')
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                return JsonResponse({'success': True, 'plot': plot_url})

            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    columns = []
    if 'file_data' in request.session:
        df = pd.read_csv(io.BytesIO(base64.b64decode(request.session['file_data'])))
        columns = df.columns.tolist()

    return render(request, 'numera/visualisation2.html', {
        'plot_types': PLOT_TYPES,
        'columns': columns
    })
    
# Hypothesis Testing

def generate_test_plots(values, test_type, null_value, alpha):
    plt.figure(figsize=(12, 6))
    sns.histplot(values, kde=True, color='skyblue')
    plt.axvline(null_value, color='red', linestyle='--', label='Null Value')
    plt.title(f'{test_type.capitalize()} Test Distribution')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return plot_url

@login_required
def hypothesis(request):
    # Initialize result variables
    test_statistic = None
    p_value = None
    conclusion = None
    data_column = None

    if request.method == 'POST':
        # Get form data
        data_column = request.POST.get('column')
        null_value = float(request.POST.get('null_value'))
        significance_level = float(request.POST.get('significance_level'))
        test_type = request.POST.get('test_type')

        # Retrieve the uploaded CSV file
        if request.FILES.get('csv_file'):
            file = request.FILES['csv_file']
            df = pd.read_csv(file)

            # Validate column existence
            if data_column in df.columns:
                data = df[data_column]

                # Perform hypothesis test based on the selected type
                if test_type == 'z_test':
                    # Calculate Z-test statistic
                    sample_mean = data.mean()
                    population_std = data.std()  # Assumes population std. If unknown, use sample std.
                    n = len(data)
                    z_stat = (sample_mean - null_value) / (population_std / (n ** 0.5))
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                elif test_type == 't_test':
                    # Perform a one-sample t-test
                    sample_mean = data.mean()
                    sample_std = data.std()
                    n = len(data)
                    t_stat, p_value = stats.ttest_1samp(data, null_value)

                # Determine the conclusion based on the p-value
                if p_value <= significance_level:
                    conclusion = "Reject the null hypothesis"
                else:
                    conclusion = "Fail to reject the null hypothesis"

                test_statistic = z_stat if test_type == 'z_test' else t_stat

    return render(request, 'numera/hypothesis.html', {  # Note the added 'numera/' prefix
        'test_statistic': test_statistic,
        'p_value': p_value,
        'conclusion': conclusion,
        'data_column': data_column,
        'columns': request.session.get('columns', [])
    })