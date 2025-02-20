from fastapi import FastAPI, HTTPException, Form, Request, Response, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pymysql
from itsdangerous import URLSafeSerializer

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = "projetliny"
serializer = URLSafeSerializer(SECRET_KEY)

# ____________________________________________________CONNEXION A LA BASE DE DONNÉES_________________________________________________
def BDD():
    try:
        connexion = pymysql.connect(
            host="192.168.20.139",
            user="root",
            password="devIA25",
            database="Projet_Regression",
            cursorclass=pymysql.cursors.DictCursor
        )
        return connexion
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de connexion : {e}")

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

# ____________________________________________Gestion des Flash Messages_________________________________________________
def get_flash_message(request: Request):
    """Récupère le message flash et le supprime après affichage"""
    message = request.cookies.get("flash_message")
    response = {"message": message}
    return response

def set_flash_message(response: Response, message: str):
    """Stocke un message flash temporaire"""
    response.set_cookie(key="flash_message", value=message, max_age=3)

# _________________________________________________________Route d'accueil__________________________________________________________

@app.get("/", response_class=HTMLResponse)
async def accueil(request: Request):
    flash_data = get_flash_message(request)
    return templates.TemplateResponse("accueil.html", {"request": request, **flash_data})

# _________________________________________________________Page de login____________________________________________________________

@app.post("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    flash_data = get_flash_message(request)
    return templates.TemplateResponse("login.html", {"request": request, **flash_data})

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
            response = RedirectResponse(url="/login", status_code=303)
            set_flash_message(response, "Nom d'utilisateur ou mot de passe incorrect.")
            return response
    finally:
        connexion.close()

# ______________________________________________________Page d'analyse sécurisée_____________________________________________________

@app.get("/analyse", response_class=HTMLResponse)
async def analyse_page(request: Request, username: str = Depends(utilisateur_connecte)):
    flash_data = get_flash_message(request)
    return templates.TemplateResponse("analyse.html", {"request": request, "username": username, **flash_data})

# ______________________________________________________Page de prédiction sécurisée_________________________________________________

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request, username: str = Depends(utilisateur_connecte)):
    flash_data = get_flash_message(request)
    return templates.TemplateResponse("prediction.html", {"request": request, "username": username, **flash_data})

# _________________________________________________________Déconnexion______________________________________________________________

@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login")
    response.delete_cookie(key="session_token")
    set_flash_message(response, "✅ Déconnexion réussie.")
    return response
