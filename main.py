import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      RepeatedStratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# HELPER: PENYIMPANAN FILE YANG AMAN (anti PermissionError di Windows)
import time


def safe_to_csv(dataframe, path, **kwargs):
    try:
        dataframe.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alt_path = path.replace(".csv", f"_{int(time.time())}.csv")
        print(f"[PERINGATAN] Tidak bisa menulis ke '{path}' (izin ditolak). "
              f"Kemungkinan file sedang terbuka di program lain (mis. Excel) "
              f"atau ada proses lain yang menguncinya.")
        print(f"             Menyimpan ke nama alternatif: '{alt_path}'")
        dataframe.to_csv(alt_path, **kwargs)
        return alt_path


def safe_savefig(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        alt_path = path.replace(".png", f"_{int(time.time())}.png")
        print(f"[PERINGATAN] Tidak bisa menulis ke '{path}' (izin ditolak). "
              f"Kemungkinan file sedang terbuka di program lain (mis. Photos/"
              f"image viewer) atau ada proses lain yang menguncinya.")
        print(f"             Menyimpan ke nama alternatif: '{alt_path}'")
        fig.savefig(alt_path, **kwargs)
        return alt_path

# MODE TESTING CEPAT
QUICK_TEST = True

# 1. LOAD DATASET
print("1. LOAD DATASET")


def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


df = read_csv_robust("dataset_ulasan_restoran.csv")
print(f"Jumlah data awal : {len(df)}")
print("Nama kolom tersedia di CSV:", list(df.columns))


def find_col(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


text_col = find_col(df, ["review_text", "review", "text", "ulasan",
                         "text_ulasan", "comment", "review_content"])
rating_col = find_col(df, ["rating", "star", "stars", "bintang", "score"])

if text_col is None:
    raise ValueError(
        f"Tidak dapat mendeteksi kolom teks secara otomatis.\n"
        f"Kolom yang tersedia di CSV: {list(df.columns)}\n"
        f"Silakan sesuaikan daftar 'candidates' di fungsi find_col()."
    )

df = df.rename(columns={text_col: "review_text"})
if rating_col is not None:
    df = df.rename(columns={rating_col: "rating"})

df["review_text"] = df["review_text"].astype(str)
INVALID_TEXT = {"nan", "none", "", "#name?", "#n/a", "#value!", "-"}
df = df[~df["review_text"].str.strip().str.lower().isin(INVALID_TEXT)].reset_index(drop=True)
print(f"Jumlah data setelah membuang teks kosong/rusak : {len(df)}\n")

# 2. TEXT PREPROCESSING (Bahasa Indonesia)
print("2. TEXT PREPROCESSING")

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{1,}", r"\1", text)
    return text


def preprocess_for_tfidf(text: str) -> str:
    text = clean_text(text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


print("Melakukan cleaning untuk analisis sentimen & fitur TF-IDF (mohon tunggu)...")
df["text_light"] = df["review_text"].apply(clean_text)
df["clean_text"] = df["review_text"].apply(preprocess_for_tfidf)

print("Contoh hasil preprocessing:")
for i in range(min(2, len(df))):
    print(f"  Asli   : {df['review_text'].iloc[i][:80]}...")
    print(f"  Bersih (TF-IDF) : {df['clean_text'].iloc[i][:80]}...\n")

# 3. PELABELAN SENTIMEN BERBASIS TEKS (LEXICON-BASED, BUKAN DARI RATING)
print("3. PELABELAN SENTIMEN BERBASIS TEKS (LEXICON-BASED)")

POSITIVE_LEXICON = {
    "enak": 2, "lezat": 2, "nikmat": 2, "mantap": 2, "sedap": 2, "gurih": 1,
    "puas": 2, "memuas": 2, "ramah": 2, "bersih": 1, "cepat": 1, "murah": 1,
    "nyaman": 1, "rekomendasi": 2, "suka": 1, "favorit": 1, "juara": 2,
    "top": 1, "oke": 1, "hangat": 1, "segar": 1, "hemat": 1, "cocok": 1,
    "istimewa": 2, "spesial": 1, "baik": 1, "senang": 1, "bagus": 2,
    "berkualitas": 1, "recommended": 2, "worth": 1, "banget": 1,
    "cinta": 1, "kece": 1, "kekinian": 1, "luas": 1, "empuk": 1, "renyah": 1,
    "wangi": 1, "menggugah": 1, "keren": 1, "sopan": 1,
    "terbaik": 2, "mantab": 2, "tanggap": 1, "improve": 1,
}

NEGATIVE_LEXICON = {
    "kecewa": 2, "buruk": 2, "jelek": 2, "kotor": 2, "lambat": 1, "mahal": 1,
    "hambar": 2, "basi": 2, "jorok": 2, "kasar": 2, "sombong": 1, "dingin": 1,
    "kosong": 1, "rusak": 2, "komplain": 1, "kapok": 2, "hancur": 1,
    "gagal": 1, "cuek": 1, "jutek": 1, "pahit": 1, "alot": 1, "bau": 1,
    "busuk": 2, "telat": 1, "payah": 1, "lelet": 1, "ngaret": 1, "lama": 1,
    "sial": 1, "amis": 1, "encer": 1, "sepi": 1, "tipis": 1,
    "kesal": 2, "tertipu": 2, "kumal": 2, "kucel": 2, "zonk": 2,
    "berantakan": 1, "brantakan": 1, "lalat": 2, "risih": 2, "keluh": 1,
    "tikus": 2, "panas": 1, "bermasalah": 1, "menipu": 2,
}

NEGATION_WORDS = {"tidak", "bukan", "tanpa", "jangan", "kurang",
                   "gak", "ga", "nggak", "enggak", "tak", "nda", "ndak"}

NEGATION_WINDOW = 2


def sentiment_score(stemmed_light_text: str) -> int:
    tokens = stemmed_light_text.split()
    score = 0
    for i, tok in enumerate(tokens):
        weight = None
        if tok in POSITIVE_LEXICON:
            weight = POSITIVE_LEXICON[tok]
        elif tok in NEGATIVE_LEXICON:
            weight = -NEGATIVE_LEXICON[tok]
        if weight is None:
            continue
        window = tokens[max(0, i - NEGATION_WINDOW):i]
        if any(w in NEGATION_WORDS for w in window):
            weight = -weight
        score += weight
    return score


print("Menghitung skor sentimen dari teks ulasan (lexicon-based)...")
df["text_stem_for_sentiment"] = df["text_light"].apply(stemmer.stem)
df["sentiment_score"] = df["text_stem_for_sentiment"].apply(sentiment_score)


def score_to_label(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "ambiguous"


df["label"] = df["sentiment_score"].apply(score_to_label)

print("\nContoh hasil pelabelan berbasis teks:")
preview_cols = ["review_text", "sentiment_score", "label"]
if "rating" in df.columns:
    preview_cols.insert(1, "rating")
print(df[preview_cols].head(8).to_string(index=False))

print(f"\nDistribusi label (sebelum filter ambiguous):\n{df['label'].value_counts()}\n")

if "rating" in df.columns:
    print("Cross-check label hasil teks vs rating bintang (HANYA untuk validasi, "
          "TIDAK dipakai dalam pelabelan/pemodelan):")
    print(pd.crosstab(df["label"], df["rating"]), "\n")

df = df[df["label"].isin(["positive", "negative"])].reset_index(drop=True)
print(f"Jumlah data setelah difilter (positive & negative) : {len(df)}")
print(f"Distribusi label final:\n{df['label'].value_counts()}\n")

if df["label"].nunique() < 2:
    raise ValueError(
        "Hanya ada 1 kelas tersisa setelah pelabelan berbasis teks. "
        "Perkaya kamus sentimen (POSITIVE_LEXICON / NEGATIVE_LEXICON) agar "
        "lebih banyak ulasan dapat diberi label."
    )

n_minority = df["label"].value_counts().min()
if n_minority < 5:
    print(f"[PERINGATAN] Kelas minoritas hanya memiliki {n_minority} sampel. "
          "Hasil evaluasi pada dataset sekecil ini akan memiliki varians "
          "tinggi -- sebaiknya perkaya kamus sentimen atau tambah data.\n")

le = LabelEncoder()
y_full = le.fit_transform(df["label"])
print(f"Mapping label: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

X_text_full = df["clean_text"].values

# 4-7. PSO FEATURE WEIGHTING (dengan skema evaluasi Repeated Stratified K-Fold)

N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 3
N_INNER_SPLITS = 3
N_INNER_REPEATS = 2
REG_LAMBDA = 0.08
MAX_TFIDF_FEATURES = 150

N_PARTICLES = 20
N_ITERATIONS = 30
C1, C2 = 1.5, 1.5
W_MIN, W_MAX = 0.4, 0.9
V_MAX_FRACTION = 0.25
STAGNATION_LIMIT = 5
REINIT_FRACTION = 0.3

PSO_MIN_IMPROVEMENT = 0.01  # ambang batas minimum perbaikan fitness agar bobot PSO diterima

PSO_TIE_MARGIN = 0.03  # jika |proba_kelas1 - proba_kelas0| < ini -> fallback ke baseline


def hybrid_predict(optimized_model, baseline_model, X_test_w, X_test_plain):
    """Prediksi PSO, kecuali untuk sampel yang probabilitasnya nyaris 50:50
    (margin < PSO_TIE_MARGIN) -- untuk sampel itu, pakai prediksi baseline.
    Valid secara metodologis: hanya memakai confidence margin (dari proba
    PSO) & prediksi baseline, keduanya dihitung TANPA melihat label asli
    (tidak ada data leakage). Diterapkan konsisten ke semua data test."""
    proba_opt = optimized_model.predict_proba(X_test_w)
    pred_opt = optimized_model.predict(X_test_w)
    pred_base = baseline_model.predict(X_test_plain)

    margin = np.abs(proba_opt[:, 1] - proba_opt[:, 0])
    is_tie = margin < PSO_TIE_MARGIN

    final_pred = np.where(is_tie, pred_base, pred_opt)
    return final_pred, proba_opt, is_tie

if QUICK_TEST:
    N_PARTICLES = 8
    N_ITERATIONS = 10


class Particle:
    def __init__(self, dim, v_max):
        self.position = np.random.uniform(0.0, 1.0, dim)
        self.velocity = np.random.uniform(-v_max, v_max, dim)
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.fitness = -np.inf


def make_fitness_function(X_train, y_train, n_splits=N_INNER_SPLITS,
                           n_repeats=N_INNER_REPEATS, reg_lambda=REG_LAMBDA):
    rskf_inner = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                          random_state=RANDOM_STATE)

    def fitness_function(weights: np.ndarray) -> float:
        weights = np.clip(weights, 0.0, 1.0)
        X_weighted = X_train * weights
        f1_scores = []
        for tr_idx, val_idx in rskf_inner.split(X_weighted, y_train):
            X_tr, X_val = X_weighted[tr_idx], X_weighted[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                      class_weight="balanced")
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, pred, average="macro", zero_division=0))
        mean_f1 = float(np.mean(f1_scores))
        penalty = reg_lambda * float(np.mean((weights - 1.0) ** 2))
        return mean_f1 - penalty

    return fitness_function


def pso_optimize(fitness_function, dim, n_particles=N_PARTICLES,
                  n_iterations=N_ITERATIONS, c1=C1, c2=C2,
                  w_min=W_MIN, w_max=W_MAX, verbose=True):
    v_max = V_MAX_FRACTION * 1.0
    swarm = [Particle(dim, v_max) for _ in range(n_particles)]
    global_best_position = None
    global_best_fitness = -np.inf
    history = []
    stagnation_counter = 0

    for it in range(n_iterations):
        inertia = w_max - (w_max - w_min) * (it / n_iterations)

        for p in swarm:
            p.fitness = fitness_function(p.position)
            if p.fitness > p.best_fitness:
                p.best_fitness = p.fitness
                p.best_position = p.position.copy()

        best_particle = max(swarm, key=lambda p: p.fitness)
        if best_particle.fitness > global_best_fitness + 1e-6:
            global_best_fitness = best_particle.fitness
            global_best_position = best_particle.best_position.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= STAGNATION_LIMIT:
            n_reinit = max(1, int(REINIT_FRACTION * n_particles))
            worst_particles = sorted(swarm, key=lambda p: p.fitness)[:n_reinit]
            for p in worst_particles:
                p.position = np.random.uniform(0.0, 1.0, dim)
                p.velocity = np.random.uniform(-v_max, v_max, dim)
            stagnation_counter = 0
            if verbose:
                print(f"    [stagnasi terdeteksi] {n_reinit} partikel di-reinisialisasi ulang")

        for p in swarm:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = c1 * r1 * (p.best_position - p.position)
            social = c2 * r2 * (global_best_position - p.position)
            p.velocity = inertia * p.velocity + cognitive + social
            p.velocity = np.clip(p.velocity, -v_max, v_max)

            new_position = p.position + p.velocity
            out_of_bounds = (new_position < 0.0) | (new_position > 1.0)
            new_position = np.clip(new_position, 0.0, 1.0)
            p.velocity = np.where(out_of_bounds, -0.5 * p.velocity, p.velocity)
            p.position = new_position

        history.append(global_best_fitness)
        if verbose:
            print(f"  Iterasi {it + 1:2d}/{n_iterations} | Best Fitness (F1 inner-CV - regularisasi) = {global_best_fitness:.4f}")

    return global_best_position, global_best_fitness, history


def run_one_fold(train_idx, test_idx, fold_label="", verbose=True):
    X_train_text = X_text_full[train_idx]
    X_test_text = X_text_full[test_idx]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, min_df=1)
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    # --- Baseline (SEBELUM OPTIMASI) ---
    baseline_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                         class_weight="balanced")
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, average="macro", zero_division=0)

    if verbose:
        print(f"\n--- {fold_label} ---")
        print(f"  Data train: {X_train.shape[0]} | Data test: {X_test.shape[0]} | Fitur: {X_train.shape[1]}")
        print(f"  Baseline (sebelum optimasi) -> Accuracy: {acc_base:.4f} | F1-macro: {f1_base:.4f}")
        print("  Menjalankan PSO...")

    # --- PSO (SESUDAH OPTIMASI) ---
    fitness_fn = make_fitness_function(X_train, y_train)
    dim = X_train.shape[1]
    best_weights, best_cv_fitness, history = pso_optimize(
        fitness_fn, dim=dim, verbose=verbose
    )

    # ------------------------------------------------------------------
    # PSO ACCEPTANCE GATE (lihat penjelasan di atas, dekat PSO_MIN_IMPROVEMENT)
    # ------------------------------------------------------------------
    neutral_weights = np.ones(dim)
    neutral_cv_fitness = fitness_fn(neutral_weights)
    pso_gain = best_cv_fitness - neutral_cv_fitness
    pso_accepted = pso_gain >= PSO_MIN_IMPROVEMENT

    if pso_accepted:
        final_weights = best_weights
    else:
        final_weights = neutral_weights

    if verbose:
        status = "DITERIMA" if pso_accepted else "DITOLAK (fallback ke bobot netral)"
        print(f"  [PSO Gate] Fitness netral (tanpa PSO): {neutral_cv_fitness:.4f} | "
              f"Fitness PSO: {best_cv_fitness:.4f} | Selisih: {pso_gain:+.4f} "
              f"(ambang: {PSO_MIN_IMPROVEMENT}) -> Bobot PSO {status}")

    X_train_w = X_train * final_weights
    X_test_w = X_test * final_weights
    optimized_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                          class_weight="balanced")
    optimized_model.fit(X_train_w, y_train)

    # Prediksi hybrid: PSO, kecuali untuk sampel bermargin sangat tipis ->
    # fallback ke baseline (lihat penjelasan di definisi hybrid_predict).
    y_pred_opt, proba_opt_test, tie_mask_test = hybrid_predict(
        optimized_model, baseline_model, X_test_w, X_test
    )
    n_tie = int(tie_mask_test.sum())
    acc_opt = accuracy_score(y_test, y_pred_opt)
    f1_opt = f1_score(y_test, y_pred_opt, average="macro", zero_division=0)

    if verbose:
        print(f"  PSO-Optimized (sesudah optimasi, hybrid tie-break) -> "
              f"Accuracy: {acc_opt:.4f} | F1-macro: {f1_opt:.4f} "
              f"(Best inner-CV F1: {best_cv_fitness:.4f}) | "
              f"{n_tie}/{len(y_test)} sampel di-fallback ke baseline (margin < {PSO_TIE_MARGIN})")

    return {
        "acc_base": acc_base, "f1_base": f1_base,
        "acc_opt": acc_opt, "f1_opt": f1_opt,
        "history": history,
        "best_weights": final_weights,
        "pso_accepted": pso_accepted,
        "pso_gain": pso_gain,
        "feature_names": vectorizer.get_feature_names_out(),
        "y_test": y_test, "y_pred_base": y_pred_base, "y_pred_opt": y_pred_opt,
        "train_idx": train_idx, "test_idx": test_idx,
        "vectorizer": vectorizer,
        "baseline_model": baseline_model,
        "optimized_model": optimized_model,
    }


# OUTER SPLIT: pilih skema evaluasi sesuai QUICK_TEST
if QUICK_TEST:
    print("=" * 78)
    print("MODE QUICK_TEST AKTIF -- 1x TRAIN/TEST SPLIT (80/20)")
    print("Dipakai HANYA untuk menguji apakah kode berjalan dengan cepat & benar.")
    print("Untuk hasil evaluasi akhir yang layak dilaporkan, set QUICK_TEST = False.")
    print("=" * 78)

    idx_all = np.arange(len(X_text_full))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )
    splits = [(train_idx, test_idx)]
else:
    print("=" * 78)
    print(f"EVALUASI DENGAN REPEATED STRATIFIED {N_OUTER_SPLITS}-FOLD "
          f"(x{N_OUTER_REPEATS} repeats) -- PENGGANTI SINGLE 80/20 SPLIT")
    print("=" * 78)

    rskf = RepeatedStratifiedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS,
                                    random_state=RANDOM_STATE)
    splits = list(rskf.split(X_text_full, y_full))

n_total_splits = len(splits)
fold_results = []
for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
    verbose = (fold_i == 1)
    result = run_one_fold(train_idx, test_idx, fold_label=f"Fold {fold_i}/{n_total_splits}",
                           verbose=verbose)
    fold_results.append(result)
    if not verbose:
        gate_note = "diterima" if result["pso_accepted"] else "ditolak->fallback"
        print(f"Fold {fold_i:2d}/{n_total_splits} selesai -> "
              f"Baseline F1: {result['f1_base']:.4f} | PSO F1: {result['f1_opt']:.4f} "
              f"| Gate PSO: {gate_note}")

main_fold = fold_results[0]

print("8. TESTING MODEL DENGAN 1 DATA PILIHAN")

vectorizer_main = main_fold["vectorizer"]
weights_main = main_fold["best_weights"]
baseline_model_main = main_fold["baseline_model"]
optimized_model_main = main_fold["optimized_model"]

INDEX_DATA_UJI = 0

print(f"Jumlah data yang tersedia untuk dipilih : {len(df)} (index 0 s/d {len(df) - 1})")
if not (0 <= INDEX_DATA_UJI < len(df)):
    raise ValueError(
        f"INDEX_DATA_UJI = {INDEX_DATA_UJI} di luar jangkauan. "
        f"Pilih angka antara 0 dan {len(df) - 1}."
    )

teks_asli_pilih = df["review_text"].iloc[INDEX_DATA_UJI]
teks_bersih_pilih = df["clean_text"].iloc[INDEX_DATA_UJI]
label_asli_pilih = df["label"].iloc[INDEX_DATA_UJI]

X_pilih_vec = vectorizer_main.transform([teks_bersih_pilih]).toarray()
X_pilih_weighted = X_pilih_vec * weights_main

pred_base_raw = baseline_model_main.predict(X_pilih_vec)
pred_opt_raw, proba_opt_pilih_arr, tie_pilih = hybrid_predict(
    optimized_model_main, baseline_model_main, X_pilih_weighted, X_pilih_vec
)
pred_base_pilih = le.inverse_transform(pred_base_raw)[0]
pred_opt_pilih = le.inverse_transform(pred_opt_raw)[0]
proba_opt_pilih = proba_opt_pilih_arr[0]
proba_dict_pilih = dict(zip(le.classes_, proba_opt_pilih))

print(f"\nGate PSO pada fold ini        : "
      f"{'DITERIMA' if main_fold['pso_accepted'] else 'DITOLAK (fallback ke bobot netral, PSO = Baseline)'} "
      f"(selisih fitness: {main_fold['pso_gain']:+.4f}, ambang: {PSO_MIN_IMPROVEMENT})")
print(f"Index data yang diuji        : {INDEX_DATA_UJI}")
print(f"Teks ulasan asli             : {teks_asli_pilih}")
print(f"Label asli (dataset)         : {label_asli_pilih}")
print(f"Prediksi Baseline (tanpa PSO): {pred_base_pilih} "
      f"({'BENAR' if pred_base_pilih == label_asli_pilih else 'SALAH'})")
print(f"Prediksi PSO-Optimized       : {pred_opt_pilih} "
      f"({'BENAR' if pred_opt_pilih == label_asli_pilih else 'SALAH'})"
      f"{'  [fallback ke baseline: margin tipis]' if tie_pilih[0] else ''}")
print(f"Probabilitas PSO per kelas   : "
      f"{ {k: round(v, 4) for k, v in proba_dict_pilih.items()} }")

margin = abs(proba_opt_pilih[0] - proba_opt_pilih[1])
if margin < PSO_TIE_MARGIN:
    print(f"[CATATAN] Margin probabilitas ({margin:.4f}) di bawah ambang tie-break "
          f"({PSO_TIE_MARGIN}) -> prediksi PSO OTOMATIS memakai hasil baseline "
          f"untuk sampel ini.")
elif margin < 0.05:
    print(f"[CATATAN] Selisih probabilitas antar kelas masih tipis ({margin:.4f}), "
          f"walau di atas ambang tie-break.")

acc_baseline_all = np.array([r["acc_base"] for r in fold_results])
f1_baseline_all = np.array([r["f1_base"] for r in fold_results])
acc_optimized_all = np.array([r["acc_opt"] for r in fold_results])
f1_optimized_all = np.array([r["f1_opt"] for r in fold_results])
n_pso_accepted = sum(r["pso_accepted"] for r in fold_results)

acc_baseline = acc_baseline_all.mean()
f1_baseline = f1_baseline_all.mean()
acc_optimized = acc_optimized_all.mean()
f1_optimized = f1_optimized_all.mean()

print("\n" + "=" * 78)
print("HASIL RATA-RATA SELURUH FOLD" + (" (QUICK_TEST: 1 split saja)" if QUICK_TEST else ""))
print("=" * 78)
print(f"Baseline       -> Accuracy: {acc_baseline:.4f} (+/-{acc_baseline_all.std():.4f}) "
      f"| F1-macro: {f1_baseline:.4f} (+/-{f1_baseline_all.std():.4f})")
print(f"PSO-Optimized  -> Accuracy: {acc_optimized:.4f} (+/-{acc_optimized_all.std():.4f}) "
      f"| F1-macro: {f1_optimized:.4f} (+/-{f1_optimized_all.std():.4f})")
print(f"Bobot PSO diterima pada {n_pso_accepted}/{len(fold_results)} fold "
      f"(sisanya fallback ke bobot netral karena perbaikan tidak meyakinkan)")
print(f"Ambang tie-break per-sampel (PSO_TIE_MARGIN) : {PSO_TIE_MARGIN} "
      f"-> sampel dengan margin probabilitas di bawah ini otomatis memakai "
      f"prediksi baseline, meski fold-nya lolos gate PSO")

print("\nClassification report (baseline, fold pertama):")
print(classification_report(main_fold["y_test"], main_fold["y_pred_base"],
                             target_names=le.classes_, zero_division=0))
print("Classification report (PSO-optimized, fold pertama):")
print(classification_report(main_fold["y_test"], main_fold["y_pred_opt"],
                             target_names=le.classes_, zero_division=0))

# 9. PERBANDINGAN HASIL: SEBELUM VS SESUDAH OPTIMASI
print("\n9. PERBANDINGAN HASIL AKHIR (rata-rata semua fold)")
comparison = pd.DataFrame({
    "Model": ["Logistic Regression (Baseline)", "Logistic Regression + PSO (Optimized)"],
    "Accuracy": [acc_baseline, acc_optimized],
    "F1-score (macro)": [f1_baseline, f1_optimized],
})
print(comparison.to_string(index=False))

improvement_acc = acc_optimized - acc_baseline
improvement_f1 = f1_optimized - f1_baseline
print(f"\nPeningkatan Accuracy : {improvement_acc:+.4f}")
print(f"Peningkatan F1-score : {improvement_f1:+.4f}")

saved_path = safe_to_csv(comparison, "hasil_perbandingan.csv", index=False)
print(f"\nTabel perbandingan disimpan ke '{saved_path}'")

# 10. VISUALISASI
print("10. VISUALISASI HASIL")

metrics_names = ["Accuracy", "F1-score (macro)"]
cm_baseline = confusion_matrix(main_fold["y_test"], main_fold["y_pred_base"])
cm_optimized = confusion_matrix(main_fold["y_test"], main_fold["y_pred_opt"])

fig_metrik_sebelum, ax = plt.subplots(figsize=(6, 5))
baseline_vals_only = [acc_baseline, f1_baseline]
bars = ax.bar(metrics_names, baseline_vals_only, color="#A23B72", width=0.5)
for bar, v in zip(bars, baseline_vals_only):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_title("Grafik Metrik SEBELUM Optimasi PSO (Baseline)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
saved_path = safe_savefig(fig_metrik_sebelum, "grafik_metrik_sebelum_optimasi.png", dpi=150)
print(f"Grafik metrik sebelum optimasi disimpan ke '{saved_path}'")
plt.close(fig_metrik_sebelum)

fig_metrik_sesudah, ax = plt.subplots(figsize=(6, 5))
optimized_vals_only = [acc_optimized, f1_optimized]
bars = ax.bar(metrics_names, optimized_vals_only, color="#F18F01", width=0.5)
for bar, v in zip(bars, optimized_vals_only):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_title("Grafik Metrik SESUDAH Optimasi PSO")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
saved_path = safe_savefig(fig_metrik_sesudah, "grafik_metrik_sesudah_optimasi.png", dpi=150)
print(f"Grafik metrik sesudah optimasi disimpan ke '{saved_path}'")
plt.close(fig_metrik_sesudah)

fig_matrix_sebelum, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_baseline, display_labels=le.classes_).plot(
    ax=ax, colorbar=False, cmap="Purples"
)
ax.set_title("Confusion Matrix SEBELUM Optimasi PSO (Fold Pertama)")
plt.tight_layout()
saved_path = safe_savefig(fig_matrix_sebelum, "grafik_matrix_sebelum_optimasi.png", dpi=150)
print(f"Grafik confusion matrix sebelum optimasi disimpan ke '{saved_path}'")
plt.close(fig_matrix_sebelum)

fig_matrix_sesudah, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_optimized, display_labels=le.classes_).plot(
    ax=ax, colorbar=False, cmap="Oranges"
)
ax.set_title("Confusion Matrix SESUDAH Optimasi PSO (Fold Pertama)")
plt.tight_layout()
saved_path = safe_savefig(fig_matrix_sesudah, "grafik_matrix_sesudah_optimasi.png", dpi=150)
print(f"Grafik confusion matrix sesudah optimasi disimpan ke '{saved_path}'")
plt.close(fig_matrix_sesudah)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes[0, 0].plot(range(1, len(main_fold["history"]) + 1), main_fold["history"],
                marker="o", color="#2E86AB", linewidth=2)
axes[0, 0].set_title("Kurva Konvergensi PSO (Fold Pertama)")
axes[0, 0].set_xlabel("Iterasi")
axes[0, 0].set_ylabel("F1-score macro (inner-CV)")
axes[0, 0].grid(alpha=0.3)

x_pos = np.arange(len(metrics_names))
width = 0.35
axes[0, 1].bar(x_pos - width / 2, baseline_vals_only, width, label="Sebelum Optimasi", color="#A23B72")
axes[0, 1].bar(x_pos + width / 2, optimized_vals_only, width, label="Sesudah Optimasi", color="#F18F01")
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(metrics_names)
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].set_title("Perbandingan Metrik Rata-rata (Semua Fold)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis="y")

ConfusionMatrixDisplay(cm_baseline, display_labels=le.classes_).plot(
    ax=axes[1, 0], colorbar=False, cmap="Purples"
)
axes[1, 0].set_title("Confusion Matrix - Sebelum Optimasi (Fold Pertama)")

ConfusionMatrixDisplay(cm_optimized, display_labels=le.classes_).plot(
    ax=axes[1, 1], colorbar=False, cmap="Oranges"
)
axes[1, 1].set_title("Confusion Matrix - Sesudah Optimasi (Fold Pertama)")

plt.tight_layout()
saved_path = safe_savefig(fig, "hasil_visualisasi.png", dpi=150)
print(f"Visualisasi gabungan disimpan ke '{saved_path}'")

# 11. ANALISIS BOBOT FITUR (TOP FITUR PALING BERPENGARUH SETELAH PSO)
print("11. ANALISIS FITUR PALING BERPENGARUH SETELAH OPTIMASI PSO (fold pertama)")

weight_df = pd.DataFrame({
    "term": main_fold["feature_names"],
    "bobot_pso": main_fold["best_weights"]
}).sort_values("bobot_pso", ascending=False)

print("Top 15 fitur dengan bobot PSO tertinggi (paling diskriminatif):")
print(weight_df.head(15).to_string(index=False))

print("\nTop 10 fitur dengan bobot PSO terendah (kurang relevan / diperkecil):")
print(weight_df.tail(10).to_string(index=False))

saved_path = safe_to_csv(weight_df, "bobot_fitur_pso.csv", index=False)
print(f"\nBobot fitur lengkap disimpan ke '{saved_path}'")

print("SELESAI. Semua hasil (CSV & PNG) tersimpan di folder ini.")
if QUICK_TEST:
    print("CATATAN: hasil di atas dari mode QUICK_TEST (1 split, PSO dipercepat).")
    print("Set QUICK_TEST = False untuk evaluasi penuh sebelum melaporkan hasil akhir.")