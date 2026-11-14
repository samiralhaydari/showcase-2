# Putting MYR SupplyLens online

This folder is ready to push as it is. It contains the redesigned app, the trained
models from Drive (`artifacts/`) and a copy of the BNM rates as a fallback.

## 1. Push to GitHub

Create an empty repository on github.com (private is fine), then from this folder:

    git init
    git add .
    git commit -m "MYR SupplyLens: redesigned interface, hosted build"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<repo-name>.git
    git push -u origin main

Check on GitHub that `artifacts/models/` shows the `.keras` files. If it is empty,
an old `.gitignore` is still excluding them.

## 2. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with the same GitHub account.
2. Create app, choose the repository, branch `main`, main file `app.py`.
3. In Advanced settings choose Python 3.12.
4. Deploy. The first build takes a few minutes; after that you get a public link.

GitHub only stores the code. Streamlit Community Cloud is what runs it.

## What to expect

- **It sleeps.** With no visitors for a while the app goes to sleep and needs about a
  minute to wake. Open the link before a demo.
- **Saved invoices and suppliers are temporary.** They live in a local SQLite file that
  is reset whenever the app restarts or redeploys.
- **Rates refresh themselves.** The app re-downloads the BNM file when its copy is more
  than 12 hours old and falls back to the bundled copy if the download fails.
- **No TensorFlow on the server.** The app runs the trained networks with NumPy
  (`src/supplylens/lite_inference.py`), which cuts memory from about 740 MB to about
  280 MB. `tests/test_lite_inference.py` checks the output against Keras for every model
  file. Training is unchanged and still uses TensorFlow in Colab.

## After retraining in Colab

Copy the new files from Drive's `artifacts/models`, `artifacts/scalers` and
`artifacts/metadata` into this folder, then:

    git add artifacts
    git commit -m "Retrain <currencies>"
    git push

Streamlit Cloud redeploys automatically.

The current scalers were saved by two scikit-learn versions (USD V2.1 with 1.6.1, the older
bundles with 1.9.1). Both load and give the same numbers, but the server log will show a
version warning until every currency is retrained in the same Colab session.
