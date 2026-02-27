import os
import sys
import numpy as np
import joblib
from flask import Flask, render_template, request

# ── Garante que o diretório do script é o working directory ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── Flask aponta explicitamente para a pasta templates ────────
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ── Carregar modelos ──────────────────────────────────────────
def carregar_modelo(nome):
    caminho = os.path.join(BASE_DIR, nome)
    if not os.path.exists(caminho):
        print(f"[ERRO] Arquivo não encontrado: {caminho}")
        sys.exit(1)
    return joblib.load(caminho)

modelo     = carregar_modelo("modelo.pkl")
scaler     = carregar_modelo("scaler.pkl")
le_gender  = carregar_modelo("le_gender.pkl")
le_workout = carregar_modelo("le_workout.pkl")
print("[OK] Modelos carregados com sucesso!")

# ── Mapeamentos ───────────────────────────────────────────────
NIVEL_MAP  = {1: "Iniciante", 2: "Intermediário", 3: "Avançado"}
NIVEL_COR  = {1: "#22c55e",   2: "#f59e0b",       3: "#ef4444"}
NIVEL_ICON = {1: "🟢",        2: "🟡",             3: "🔴"}

TREINOS = {
    ("Iniciante",     "hipertrofia", 3): ["Peito + Tríceps","Descanso","Costas + Bíceps","Descanso","Pernas + Glúteos","Descanso","Descanso"],
    ("Iniciante",     "hipertrofia", 4): ["Peito + Tríceps","Costas + Bíceps","Descanso","Pernas + Glúteos","Ombros + Core","Descanso","Descanso"],
    ("Iniciante",     "hipertrofia", 5): ["Peito + Tríceps","Costas + Bíceps","Pernas + Glúteos","Ombros + Core","Braços + Core","Descanso","Descanso"],
    ("Intermediário",  "hipertrofia", 4): ["Peito + Tríceps","Costas + Bíceps","Descanso","Pernas (Quad)","Ombros + Trapézio","Pernas (Post)","Descanso"],
    ("Intermediário",  "hipertrofia", 5): ["Peito","Costas","Pernas","Ombros + Trapézio","Braços + Core","Descanso","Descanso"],
    ("Intermediário",  "hipertrofia", 6): ["Peito + Tríceps","Costas + Bíceps","Pernas (Quad)","Ombros","Braços","Pernas (Post)","Descanso"],
    ("Avançado",      "hipertrofia", 5): ["Peito + Tríceps","Costas + Bíceps","Pernas (Quad)","Ombros + Trapézio","Pernas (Post + Glúteo)","Descanso","Descanso"],
    ("Avançado",      "hipertrofia", 6): ["Peito","Costas","Pernas (Quad)","Ombros","Braços","Pernas (Post)","Descanso"],
    ("Iniciante",     "força",       3): ["Supino + Acessórios","Descanso","Agachamento + Acessórios","Descanso","Terra + Acessórios","Descanso","Descanso"],
    ("Intermediário",  "força",       4): ["Supino (Força)","Agachamento (Força)","Descanso","Terra (Força)","Press + Assist.","Descanso","Descanso"],
    ("Avançado",      "força",       5): ["Supino (Força)","Agachamento (Força)","Terra (Força)","Press + Assist.","Acessórios Gerais","Descanso","Descanso"],
    ("Iniciante",     "definição",   3): ["Upper (Alta Rep)","Descanso","Lower (Alta Rep)","Descanso","Full Body + HIIT","Descanso","Descanso"],
    ("Intermediário",  "definição",   4): ["Peito + Tríceps (Alta Rep)","Costas + Bíceps (Alta Rep)","Descanso","Pernas + HIIT","Ombros + Core","Descanso","Descanso"],
    ("Avançado",      "definição",   5): ["Peito + Tríceps","Costas + Bíceps","Pernas","Ombros + Core","HIIT / Cardio","Descanso","Descanso"],
}

EXERCICIOS = {
    "Peito + Tríceps":             [("Supino Reto com Barra","4x8-12","Composto principal"),("Supino Inclinado Halteres","3x10-12","Porção superior"),("Crucifixo na Polia","3x12-15","Isolamento"),("Tríceps Corda (Polia)","4x12-15","Cabeça lateral"),("Tríceps Testa com Barra EZ","3x10-12","Cabeça longa")],
    "Costas + Bíceps":             [("Barra Fixa ou Puxada Alta","4x8-10","Largura das costas"),("Remada Curvada com Barra","4x8-10","Espessura das costas"),("Remada Unilateral Haltere","3x10-12","Isolamento dorsal"),("Rosca Direta com Barra","4x10-12","Bíceps braquial"),("Rosca Martelo","3x12","Braquial e antebraço")],
    "Pernas + Glúteos":            [("Agachamento Livre com Barra","4x8-12","Quadríceps + glúteo"),("Leg Press 45°","4x12-15","Volume de pernas"),("Cadeira Extensora","3x15","Isolamento quadríceps"),("Mesa Flexora","3x12-15","Isquiotibiais"),("Elevação Pélvica (Hip Thrust)","4x12","Glúteo máximo")],
    "Ombros + Trapézio":           [("Desenvolvimento com Halteres","4x10-12","Deltóide anterior"),("Elevação Lateral","4x12-15","Deltóide medial"),("Elevação Frontal Alternada","3x12","Deltóide anterior"),("Remada Alta com Barra","3x12","Trapézio + deltóide"),("Encolhimento com Halteres","4x15","Trapézio superior")],
    "Ombros + Core":               [("Desenvolvimento Militar","4x10-12","Ombros geral"),("Elevação Lateral","3x15","Deltóide medial"),("Face Pull","3x15","Deltóide posterior"),("Prancha","3x60s","Core estabilizador"),("Abdominal Crunch","4x20","Reto abdominal")],
    "Braços + Core":               [("Rosca Direta","4x10-12","Bíceps"),("Rosca Concentrada","3x12","Pico do bíceps"),("Tríceps Corda","4x12-15","Tríceps geral"),("Tríceps Francês","3x10-12","Cabeça longa"),("Abdominal Infra","4x20","Core inferior")],
    "Pernas (Quad)":               [("Agachamento Livre com Barra","5x5-8","Força + volume"),("Leg Press 45°","4x10-12","Quadríceps"),("Hack Squat","3x10-12","Vasto lateral"),("Cadeira Extensora","3x15-20","Finalização"),("Panturrilha em Pé","4x15-20","Gastrocnêmio")],
    "Pernas (Post)":               [("Stiff com Barra","4x10-12","Isquiotibiais"),("Mesa Flexora","4x12-15","Isolamento posterior"),("Agachamento Sumô","3x12","Adutores + glúteo"),("Elevação Pélvica","4x12-15","Glúteo máximo"),("Panturrilha Sentado","4x15-20","Sóleo")],
    "Pernas (Post + Glúteo)":      [("Stiff com Barra","4x10-12","Cadeia posterior"),("Mesa Flexora","4x12-15","Isquiotibiais"),("Elevação Pélvica","5x12","Glúteo máximo"),("Abdução de Quadril (Polia)","3x15","Glúteo médio"),("Panturrilha Sentado","3x20","Sóleo")],
    "Peito":                       [("Supino Reto com Barra","5x6-8","Composto força"),("Supino Inclinado Haltere","4x10-12","Superior"),("Crucifixo Polia","3x15","Isolamento"),("Crossover","3x15","Finalizador"),("Flexão Declinada","3xFalha","Inferior")],
    "Costas":                      [("Barra Fixa Lastrada","5x5-8","Força"),("Remada Curvada Barra","4x8-10","Espessura"),("Remada Unilateral Haltere","3x10-12","Unilateral"),("Pullover","3x12","Serrátil"),("Face Pull","3x15","Rotadores")],
    "Pernas":                      [("Agachamento Livre","5x5","Força máxima"),("Leg Press","4x10-12","Volume"),("Cadeira Extensora","3x15","Quadríceps"),("Mesa Flexora","3x15","Isquiotibiais"),("Panturrilha em Pé","5x20","Gastrocnêmio")],
    "Ombros":                      [("Desenvolvimento Halteres","5x8-10","Deltóide geral"),("Elevação Lateral","5x12-15","Deltóide medial"),("Elevação Frontal","3x12","Deltóide anterior"),("Face Pull","4x15","Posterior"),("Arnold Press","3x10","Rotação completa")],
    "Braços":                      [("Rosca Direta com Barra","4x10-12","Bíceps braquial"),("Rosca Martelo","3x12","Braquial"),("Tríceps Corda (Polia)","4x12-15","Cabeça lateral"),("Tríceps Testa","3x10","Cabeça longa"),("Rosca 21","3x21","Técnica de choque")],
    "Supino + Acessórios":         [("Supino Reto com Barra","3x5","Força máxima"),("Supino Inclinado Halteres","3x8-10","Assistência"),("Tríceps Corda","3x12","Assistência"),("Crucifixo","3x12","Assistência")],
    "Agachamento + Acessórios":    [("Agachamento Livre","3x5","Força máxima"),("Leg Press","3x10","Volume"),("Cadeira Extensora","3x15","Assistência"),("Panturrilha","4x20","Assistência")],
    "Terra + Acessórios":          [("Levantamento Terra","3x5","Força máxima"),("Remada Curvada","3x8","Assistência"),("Barra Fixa","3x8","Assistência"),("Hiperextensão","3x15","Lombar")],
    "Supino (Força)":              [("Supino Reto com Barra","5x3-5","Força máxima"),("Supino Inclinado","3x6-8","Assistência"),("Tríceps Corda","4x10","Assistência"),("Elevação Frontal","3x12","Assistência")],
    "Agachamento (Força)":         [("Agachamento Livre com Barra","5x3-5","Força máxima"),("Agachamento Búlgaro","3x8","Assistência unilateral"),("Leg Press","3x10","Volume"),("Extensora","3x15","Assistência")],
    "Terra (Força)":               [("Levantamento Terra","5x3-5","Força máxima"),("Stiff","3x8","Assistência posterior"),("Remada Curvada","4x8","Assistência dorsal"),("Barra Fixa Lastrada","3x6","Assistência")],
    "Press + Assist.":             [("Desenvolvimento Militar","4x6-8","Força ombros"),("Push Press","3x5","Potência"),("Elevação Lateral","4x12","Volume"),("Face Pull","3x15","Saúde do ombro")],
    "Acessórios Gerais":           [("Rosca Direta","4x10-12","Bíceps"),("Tríceps Corda","4x12","Tríceps"),("Elevação Lateral","3x15","Ombros"),("Abdominal Crunch","4x20","Core"),("Panturrilha","4x20","Gastrocnêmio")],
    "Upper (Alta Rep)":            [("Supino com Halteres","3x15-20","Peito"),("Puxada Alta","3x15","Costas"),("Rosca Direta","3x15","Bíceps"),("Tríceps Corda","3x15","Tríceps"),("Elevação Lateral","3x20","Ombros")],
    "Lower (Alta Rep)":            [("Leg Press","4x20","Quadríceps"),("Mesa Flexora","4x20","Isquiotibiais"),("Hip Thrust","4x20","Glúteos"),("Extensora","3x25","Finalização"),("Panturrilha","4x25","Gastrocnêmio")],
    "Full Body + HIIT":            [("Agachamento","3x15","Pernas"),("Supino Halteres","3x15","Peito"),("Remada Curvada","3x15","Costas"),("Desenvolvimento","3x15","Ombros"),("HIIT – Burpees","4x30s","Cardio finalizador")],
    "Peito + Tríceps (Alta Rep)":  [("Supino com Halteres","4x15-20","Peito – alta rep"),("Crucifixo Polia","3x20","Isolamento"),("Flexão de Braço","3xFalha","Calistenics"),("Tríceps Corda","4x20","Tríceps"),("Tríceps Testa","3x15","Cabeça longa")],
    "Costas + Bíceps (Alta Rep)":  [("Puxada Alta","4x15-20","Largura"),("Remada Baixa","4x15-20","Espessura"),("Pullover","3x20","Serrátil"),("Rosca Direta","4x15-20","Bíceps"),("Rosca Concentrada","3x20","Pico")],
    "Pernas + HIIT":               [("Agachamento Livre","4x15","Quad + glúteo"),("Stiff","3x15","Posterior"),("Elevação Pélvica","4x20","Glúteo"),("Panturrilha","4x25","Gastrocnêmio"),("HIIT – Pular Corda","5x60s","Cardio finalizador")],
    "HIIT / Cardio":               [("Aquecimento Leve","10 min","Preparo"),("Sprint 30s / Caminhada 90s","8 rounds","Intervalo alto"),("Jump Squat","4x20","Potência"),("Mountain Climbers","4x40s","Core + cardio"),("Alongamento","10 min","Recuperação")],
    "Descanso": [],
}

DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


def get_treino(nivel_nome, objetivo, freq):
    freq = min(max(int(freq), 3), 6)
    key = (nivel_nome, objetivo, freq)
    if key in TREINOS:
        return TREINOS[key]
    # Fallback: mesmo nível e objetivo, freq mais próxima
    candidatos = [(k, v) for k, v in TREINOS.items()
                  if k[0] == nivel_nome and k[1] == objetivo]
    if candidatos:
        candidatos.sort(key=lambda x: abs(x[0][2] - freq))
        return candidatos[0][1]
    # Fallback geral
    return TREINOS[("Intermediário", "hipertrofia", 4)]


def calcular_macros(peso, objetivo):
    obj = objetivo.lower()
    if obj in ["hipertrofia", "força"]:
        return {"calorias": int(peso*35), "proteina": int(peso*2.2), "carbo": int(peso*5.0), "gordura": int(peso*1.0)}
    else:
        return {"calorias": int(peso*28), "proteina": int(peso*2.5), "carbo": int(peso*3.0), "gordura": int(peso*0.8)}


# ── Rotas ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/resultado", methods=["POST"])
def resultado():
    # Coleta do formulário
    nome        = request.form.get("nome", "Atleta")
    age         = float(request.form.get("age", 25))
    gender      = request.form.get("gender", "Male")
    weight      = float(request.form.get("weight", 75))
    height      = float(request.form.get("height", 1.75))
    resting_bpm = float(request.form.get("resting_bpm", 65))
    avg_bpm     = float(request.form.get("avg_bpm", 140))
    max_bpm     = float(request.form.get("max_bpm", 175))
    fat_pct     = float(request.form.get("fat_pct", 18))
    freq        = int(request.form.get("freq", 4))
    duracao     = float(request.form.get("duracao", 1.0))
    workout_t   = request.form.get("workout_type", "Strength")
    agua        = float(request.form.get("agua", 3.0))
    objetivo    = request.form.get("objetivo", "hipertrofia")

    bmi          = round(weight / (height ** 2), 2)
    calorias_est = duracao * 500 + avg_bpm * 0.5

    # Encoding
    gen_enc = le_gender.transform([gender])[0]
    wk_enc  = le_workout.transform([workout_t])[0]

    X_user = np.array([[
        age, weight, height, bmi,
        max_bpm, avg_bpm, resting_bpm,
        duracao, calorias_est,
        fat_pct, agua, freq,
        gen_enc, wk_enc
    ]])
    X_scaled = scaler.transform(X_user)

    # Predição — modelo pode retornar 0,1,2 ou 1,2,3
    raw = int(modelo.predict(X_scaled)[0])
    nivel_idx = raw if raw in [1, 2, 3] else raw + 1
    nivel_idx = max(1, min(3, nivel_idx))

    nivel_nome = NIVEL_MAP[nivel_idx]
    nivel_cor  = NIVEL_COR[nivel_idx]
    nivel_icon = NIVEL_ICON[nivel_idx]

    # Probabilidades
    proba = None
    if hasattr(modelo, "predict_proba"):
        p     = modelo.predict_proba(X_scaled)[0]
        proba = [round(float(x) * 100, 1) for x in p]

    # Plano semanal
    divisao = get_treino(nivel_nome, objetivo, freq)
    plano = []
    for dia, treino in zip(DIAS, divisao):
        plano.append({
            "dia":       dia,
            "treino":    treino,
            "descanso":  treino == "Descanso",
            "exercicios": EXERCICIOS.get(treino, []),
        })

    # IMC
    if   bmi < 18.5: bmi_class, bmi_cor = "Abaixo do peso", "#60a5fa"
    elif bmi < 25.0: bmi_class, bmi_cor = "Normal",         "#22c55e"
    elif bmi < 30.0: bmi_class, bmi_cor = "Sobrepeso",      "#f59e0b"
    else:            bmi_class, bmi_cor = "Obesidade",      "#ef4444"

    macros = calcular_macros(weight, objetivo)

    return render_template("resultado.html",
        nome=nome, nivel_nome=nivel_nome, nivel_cor=nivel_cor, nivel_icon=nivel_icon,
        bmi=bmi, bmi_class=bmi_class, bmi_cor=bmi_cor,
        plano=plano, macros=macros, proba=proba,
        objetivo=objetivo.title(), freq=freq,
        weight=weight, fat_pct=fat_pct, avg_bpm=avg_bpm,
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  GymIQ Flask – Iniciando servidor...")
    print(f"  Pasta base: {BASE_DIR}")
    print(f"  Templates:  {os.path.join(BASE_DIR, 'templates')}")
    print("  Acesse:     http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
