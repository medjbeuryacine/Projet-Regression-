from fastapi import FastAPI, HTTPException, Form, Request, Response, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pymysql
from dotenv import load_dotenv
import os
from itsdangerous import URLSafeSerializer

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = "votre_clef_secrète_pour_signer_les_cookies"  # Utilisez une clé sécurisée et complexe
serializer = URLSafeSerializer(SECRET_KEY)

# ____________________________________________________CONNEXION A LA BASE DE DONNÉES_________________________________________________

def BDD():
    try:
        connexion = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            cursorclass=pymysql.cursors.DictCursor
        )
        return connexion
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problème de connexion à la base de données: {e}")

# ____________________________________________Vérification de l'utilisateur connecté_________________________________________________

def utilisateur_connecte(request: Request):
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=403, detail="Non autorisé")
    
    try:
        data = serializer.loads(token)
        return data.get("username")
    except Exception:
        raise HTTPException(status_code=403, detail="Token invalide ou expiré")

# _________________________________________________________Route d'accueil__________________________________________________________

@app.get("/", response_class=HTMLResponse)
async def accueil(request: Request):
    return templates.TemplateResponse("accueil.html", {"request": request, "message": "Bienvenue sur LINY"})

# _________________________________________________________Page de login____________________________________________________________

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), response: Response = None):
    connexion = BDD()
    try: 
        with connexion.cursor() as cursor:
            sql = "SELECT * FROM utilisateurs WHERE username = %s AND password = sha2(%s, 256)"
            cursor.execute(sql, (username, password))
            utilisateur = cursor.fetchone()

        if utilisateur:
            # Créer un token de session
            session_token = serializer.dumps({"username": username})
            response = RedirectResponse(url="/analyse", status_code=303)
            response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True)
            return response
        else:
            raise HTTPException(status_code=404, detail="Nom d'utilisateur ou mot de passe incorrect.")
    finally:
        connexion.close()

# ______________________________________________________Page d'analyse sécurisée_____________________________________________________

@app.get("/analyse", response_class=HTMLResponse)
async def analyse_page(request: Request, username: str = Depends(utilisateur_connecte)):
    return templates.TemplateResponse("analyse.html", {"request": request, "username": username})

# ______________________________________________________Page de prédiction sécurisée_________________________________________________

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request, username: str = Depends(utilisateur_connecte)):
    return templates.TemplateResponse("prediction.html", {"request": request, "username": username})

# _________________________________________________________Déconnexion______________________________________________________________

@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/")
    response.delete_cookie(key="session_token")
    return response
