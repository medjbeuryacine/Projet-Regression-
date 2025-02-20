import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prix_euro(df):
    ECHANGE = 0.011
    df["prix_euro"] = df["Price"] * 100000 * ECHANGE  # Conversion
    df = df.drop(columns=["Price"])
    return df

prix_euro_transformer = FunctionTransformer(prix_euro, validate=False)

def supprimer_lignes_cheres(df):
    # Supprimer les lignes où 'prix_euro' > 66 000
    df = df.drop(df[df["prix_euro"] >= 66000].index, axis=0)
    return df  # Garde uniquement les lignes <= 66 000€

# Transformer pour la pipeline
supprimer_lignes_cheres_transformer = FunctionTransformer(supprimer_lignes_cheres, validate=False)

def split_name(nom_voiture):
    # Diviser le nom en mots
    mots = nom_voiture.split()
   
    # Si le nom contient "Land Rover", on prend les deux premiers mots comme Brand et le reste comme Model
    if 'Land' in mots and 'Rover' in mots:
        brand = ' '.join(mots[:2])
        if 'Range' in mots and 'Rover' in mots:
            model = ' '.join(mots[2:4])
        else:
            model = ' '.join(mots[2:3])
    else:
        brand = mots[0]
        model = mots[1] if len(mots) > 1 else ''
    return pd.Series([brand, model])

def ajouter_colonnes_marque_modele(df):
    # Appliquer la fonction split_name pour extraire 'Brand' et 'Model' à partir de 'Name'
    df[['Brand', 'Model']] = df['Name'].apply(split_name)
    return df


split_colonne_transformer = FunctionTransformer(ajouter_colonnes_marque_modele, validate=False)

def lowercase_categorical_columns(df):
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].str.lower()
    return df

lowercase_transformer = FunctionTransformer(lowercase_categorical_columns, validate=False)

def drop_duplicated(df):
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates(inplace=True)
        return df
    return df

supprimer_doublons_transformer = FunctionTransformer(drop_duplicated, validate=False)

def convert_to_float(df):
    colonnes = ['Mileage', 'Engine', 'Power']
    for col in colonnes:
        if col in df.columns:
            if df[col].dtype == object:
                # Si la colonne contient une virgule, on applique la transformation
                df[col] = df[col].str.extract(r'([0-9.]+)').astype(float)

            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df  

convertir_numerique_transformer = FunctionTransformer(convert_to_float, validate=False)

def trouver_voitures_similaires(df):
    # Ajouter une colonne 'Similar_Cars' vide
    df['Similar_Cars'] = None
    
    for index, row in df.iterrows():
        marque_voiture = row['Brand']
        modele_voiture = row['Model']
        fuel_type = row['Fuel_Type']
        transmission = row['Transmission']
        
        # Rechercher des voitures ayant la même marque, modèle, carburant et transmission
        similar_cars = df[
            (df['Brand'] == marque_voiture) & 
            (df['Model'] == modele_voiture) & 
            (df['Fuel_Type'] == fuel_type) & 
            (df['Transmission'] == transmission)
        ]
        
        # Si au moins 2 véhicules (lui-même + un autre) sont trouvés, on ajoute les résultats
        if len(similar_cars) >= 2:
            df.at[index, 'Similar_Cars'] = ', '.join(similar_cars['Name'].values)
    
    return df

voitures_similaires_transformer = FunctionTransformer(trouver_voitures_similaires, validate=False)

def drop_colonne(df):
    colonne_drop = ["New_Price", "Name", "Price"]
    # On vérifie que les colonnes à supprimer existent dans le DataFrame
    colonne_drop = [col for col in colonne_drop if col in df.columns]
    # Suppression des colonnes et retour du DataFrame modifié
    df = df.drop(columns=colonne_drop, axis=1)
    return df  # Assurez-vous de toujours retourner le DataFrame modifié

supprimer_colonne_transformer = FunctionTransformer(drop_colonne, validate=False)

def remplir_valeurs_manquantes(df):
    for index, row in df.iterrows():
        # Liste des colonnes à compléter
        list_colonnes = ['Mileage', 'Engine', 'Power', 'Seats']
        for col in list_colonnes:
            # Vérifier si la valeur dans la cellule est manquante (NaN)
            if pd.isnull(row[col]):  
                # Rechercher des véhicules similaires avec le même carburant et transmission
                similar_cars = df[df['Similar_Cars'].notnull()]  # Utiliser la colonne 'Similar_Cars' déjà remplie
                if not similar_cars.empty:
                    # Récupérer la valeur la plus fréquente (mode) pour la colonne en question
                    mode_value = similar_cars[col].mode()
                    if not mode_value.empty:  # Si un mode existe, utiliser la première valeur trouvée
                        df.at[index, col] = mode_value.iloc[0]  
                    else:  # Si aucun mode n'est trouvé, utiliser la médiane de la colonne entière
                        df.at[index, col] = df[col].median()  
                else:
                    # Si aucune voiture similaire n'est trouvée, remplir avec la médiane de la colonne
                    df.at[index, col] = df[col].median()  
    return df.drop("Similar_Cars", axis=1)

valeurs_manquantes_transformer = FunctionTransformer(remplir_valeurs_manquantes, validate=False)

def reduire_outliers(df):
    # Liste des colonnes à traiter
    list_colonnes = ['Mileage', 'Engine', 'Power', 'Seats', 'Kilometers_Driven']
    
    # Calcul des bornes IQR
    Q1 = df[list_colonnes].quantile(0.25)
    Q3 = df[list_colonnes].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    # Appliquer les bornes aux colonnes, réduisant les outliers
    df[list_colonnes] = df[list_colonnes].apply(
        lambda x: x.clip(lower=borne_inf[x.name], upper=borne_sup[x.name])  # Utilisation de apply pour chaque colonne
    )

    return df

outliers_transformer = FunctionTransformer(reduire_outliers, validate=False)

def transformer(df):

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols_onehot = ["Fuel_Type","Transmission","Owner_Type", "Brand", "Model"]


    processeur = ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(sparse_output=False), categorical_cols_onehot)
    ])

    df_transformer = processeur.fit_transform(df)

    return df_transformer

transformers = FunctionTransformer(transformer, validate=False) ## transformer pour mettre dans la pipeline

# Création de la pipeline
preprocessing_pipeline_1 = Pipeline([
    ("prix_euro", prix_euro_transformer),
    ("supprimer_lignes_cheres", supprimer_lignes_cheres_transformer),
    ("split_colonne", split_colonne_transformer),
    ("supprimer_doublons", supprimer_doublons_transformer),
    ("convertir_numerique", convertir_numerique_transformer),
    ("voitures_similaires", voitures_similaires_transformer),
    ("supprimer_colonne", supprimer_colonne_transformer),
    ("lowercase", lowercase_transformer), 
])

preprocessing_pipeline_2 = Pipeline([
    ("valeurs_manquantes", valeurs_manquantes_transformer),
    ("outliers", outliers_transformer),
    ("transformers", transformers)
])

# Charger le dataset
df = pd.read_csv("train.csv")

# Appliquer la première partie du pipeline avant le split
df_transformed_1 = preprocessing_pipeline_1.fit_transform(df)


# Séparation des features et de la cible
X = df_transformed_1.drop(columns=["prix_euro"])  # Features
y = df_transformed_1["prix_euro"]  # Target

# Split des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformation des données
X_train_transformed = preprocessing_pipeline_2.fit_transform(X_train)
X_test_transformed = preprocessing_pipeline_2.transform(X_test)

print("Colonnes de X_train_transformed:", X_train_transformed.shape)
print("Colonnes de X_test_transformed:", X_test_transformed.shape)

# Récupérer les colonnes des jeux d'entraînement et de test
columns_train = pd.DataFrame(X_train_transformed).columns
columns_test = pd.DataFrame(X_test_transformed).columns

# Aligner les colonnes des deux jeux
missing_train_columns = columns_test.difference(columns_train)
missing_test_columns = columns_train.difference(columns_test)

# Ajouter des colonnes manquantes (remplies de 0) dans les jeux de données
for col in missing_train_columns:
    X_train_transformed = np.c_[X_train_transformed, np.zeros(X_train_transformed.shape[0])]

for col in missing_test_columns:
    X_test_transformed = np.c_[X_test_transformed, np.zeros(X_test_transformed.shape[0])]

# Vérifier que les dimensions sont maintenant identiques
print(f"Dimensions après alignement: {X_train_transformed.shape}, {X_test_transformed.shape}")

# # Entraînement du modèle SVR
svr = SVR()
svr.fit(X_train_transformed, y_train)

# Prédiction
y_pred = svr.predict(X_test_transformed)

# Calcul des métriques
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
