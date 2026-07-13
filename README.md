# Spam Classifier — FastAPI + Flask + Docker + Render

Predicts whether a message is **spam** or **ham** (not spam).

- **Model**: TF-IDF + Logistic Regression *and* Linear SVM, trained on the SMS Spam
  Collection dataset. Both are trained and scored; the better one (by F1) is saved
  and used. In this run, **SVM won** (F1 = 0.948 vs 0.942 for Logistic Regression),
  though both scored ~98% accuracy.
- **Backend**: FastAPI, exposes a `/predict` JSON endpoint.
- **Frontend**: Flask, a simple web form that calls the backend.
- **Deployment**: Each service has its own Dockerfile; deploy both to Render as
  two separate web services.

```
spam_classifier/
├── model_training/
│   ├── spam.csv              # dataset
│   ├── train_model.py        # trains + compares LR and SVM, saves the winner
│   └── requirements.txt
├── backend/                  # FastAPI service
│   ├── main.py
│   ├── text_utils.py
│   ├── model.pkl             # trained model (already generated)
│   ├── vectorizer.pkl        # trained TF-IDF vectorizer (already generated)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── frontend/                 # Flask service
│   ├── app.py
│   ├── templates/index.html
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .dockerignore
├── render.yaml                # optional: one-click Render Blueprint for both services
└── .gitignore
```

The model is already trained and `model.pkl` / `vectorizer.pkl` are sitting in
`backend/`, so you can run the backend immediately without retraining anything.

---

## 1. Open the project in PyCharm

1. `File -> Open` and select the `spam_classifier` folder.
2. Create two run configurations (or one virtualenv per folder is optional — a
   single shared venv works fine for a project this size):
   - PyCharm: `File -> Settings -> Project -> Python Interpreter -> Add Interpreter
     -> Virtualenv Environment -> New`.
3. Mark `backend/` and `frontend/` as regular source folders (default is fine).

You don't need two separate PyCharm projects — one project with two run
configurations is simplest.

---

## 2. Run locally (no Docker)

**Backend (FastAPI)** — in a terminal (or PyCharm terminal):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! Claim your free prize now"}'
```

**Frontend (Flask)** — in a second terminal:
```bash
cd frontend
pip install -r requirements.txt
python app.py
```
Open **http://localhost:5000** in your browser. It defaults to talking to the
backend at `http://localhost:8000` (see `BACKEND_URL` env var in `app.py`).

### PyCharm run configurations
- **Backend**: Run `uvicorn` — Script path: use "Module name" = `uvicorn`,
  parameters: `main:app --reload --port 8000`, working directory: `backend/`.
- **Frontend**: Run `app.py` directly, working directory: `frontend/`.

---

## 3. Retrain the model (optional)

Only needed if you want to change the algorithm, features, or add data.
```bash
cd model_training
pip install -r requirements.txt
python train_model.py
```
This prints a full comparison (accuracy/precision/recall/F1) for both Logistic
Regression and SVM, then saves `model.pkl` + `vectorizer.pkl` for whichever won.
Copy those two files into `backend/` to update the deployed model.

---

## 4. Run with Docker (locally)

Each service is a separate image.

**Backend:**
```bash
cd backend
docker build -t spam-backend .
docker run -p 8000:8000 spam-backend
```

**Frontend:**
```bash
cd frontend
docker build -t spam-frontend .
docker run -p 5000:5000 -e BACKEND_URL=http://host.docker.internal:8000 spam-frontend
```
(`host.docker.internal` lets the frontend container reach the backend
container running on your host machine. On Linux, use `--network=host`
instead, or run both containers on a shared Docker network.)

Open **http://localhost:5000**.

---

## 5. Deploy to Render

You'll create **two** separate Web Services on Render, both from the same
GitHub repo, each pointed at a different subfolder.

### Push to GitHub first
```bash
cd spam_classifier
git init
git add .
git commit -m "Spam classifier: FastAPI + Flask"
git remote add origin <your-repo-url>
git push -u origin main
```

### Option A — manual setup (recommended the first time)

**Backend service:**
1. Render dashboard → **New +** → **Web Service** → connect your repo.
2. **Root Directory**: `backend`
3. **Runtime**: Docker (Render auto-detects the Dockerfile)
4. Deploy. Once live, copy its URL, e.g. `https://spam-classifier-backend.onrender.com`.

**Frontend service:**
1. **New +** → **Web Service** → same repo.
2. **Root Directory**: `frontend`
3. **Runtime**: Docker
4. Add an environment variable: `BACKEND_URL` = the backend URL from above.
5. Deploy.

Visit the frontend's Render URL — it should classify messages via the backend.

### Option B — Blueprint (`render.yaml`)
This repo includes a `render.yaml` at the root. In Render: **New +** →
**Blueprint** → select the repo → Render creates both services automatically.
You'll still need to manually set `BACKEND_URL` on the frontend service once
(Render dashboard → frontend service → Environment) to the backend's public URL,
since Render fills this in more reliably by hand than through blueprint wiring
across two independent web services.

---

## Notes on the model choice

Logistic Regression and Linear SVM are both strong, fast baselines for
text classification on small-to-medium datasets like this one (5,572 SMS
messages) — much cheaper to train and serve than a deep learning model, and
plenty accurate (~98%) for this task. `train_model.py` trains both and keeps
whichever scores higher on held-out F1, so you get the better of the two
without having to choose upfront. LinearSVC doesn't natively output
probabilities, so it's wrapped in `CalibratedClassifierCV` — this is why the
API can return a `spam_probability` either way, not just a hard label.
