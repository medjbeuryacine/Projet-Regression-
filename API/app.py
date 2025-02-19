from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pymysql
import jinja2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ____________________________________________________CONNEXION A LA BASE DE DONNÉES_________________________________________________
def BDD():
    try:
        connexion = pymysql.connect(
        host = "192.168.20.139", # Adresse du serveur MySQL
        user = "root",      # Nom d'utilisateur MySQL
        password =  "devIA25",  # Mot de passe MySQL
        database = "Projet_Regression",   # Nom de la base de données
        cursorclass=pymysql.cursors.DictCursor
        )
        return connexion
    except Exception as e:
        raise HTTPException (status_code=500, detail=f"vous avez le problem: {e}")
    
try:
    BDD()
    print("connecte")
except Exception as e:
    print(e)


# _________________________________________________________faire_les_routes___________________________________________________

@app.get("/", response_class=HTMLResponse)
async def accueil(request:Request):
    return templates.TemplateResponse("accueil.html", {"request":request, "message": "Bienvenue sur LINY"})



#____________________________________________________________la_page_login______________________________________________________

@app.get("/login", response_class=HTMLResponse)
async def login_page(request:Request):
    return templates.TemplateResponse("login.html", {"request":request})


@app.post("/login")
async def login(username: str = Form(...) , password: str = Form(...)):
        connexion = BDD()
        try: 
            with connexion.cursor() as cursor:
                sql = "SELECT * FROM utilisateurs WHERE username = %s AND password = sha2(%s, 256)"
                cursor.execute(sql, (username, password))
                utilisateur = cursor.fetchone()

            if utilisateur:
                return RedirectResponse(url="/analyse", status_code=303) # si il connecte , il vas être diriger à la page d'analyse
            else:
                raise HTTPException(status_code=404 ,detail="Nom d'utilisateur ou mot de passe incorrect.")
        finally:
             connexion.close()



#____________________________________________________________la_page_graphique______________________________________________________

@app.get("/analyse", response_class=HTMLResponse)
async def analyse_page(request:Request):
    return templates.TemplateResponse("analyse.html", {"request":request})

async def analyse():
    pass





#____________________________________________________________la_page_prediction______________________________________________________

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request:Request):
    return templates.TemplateResponse("prediction.html", {"request":request})

async def prediction():
    pass





