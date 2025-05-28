import os, glob, re, pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOP = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\b0x[a-fA-F0-9]+\b", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP)

def safe_strip(val):
    """Return a stripped string, or empty string if val is NaN or None."""
    if pd.isna(val):
        return ""
    return str(val).strip()

def load_and_preprocess(buglist_dir, history_dir, desc_dir, out_prefix):
    bug_info = {}
    for csv_path in glob.glob(os.path.join(buglist_dir, "bugs*.csv")):
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            bid = str(r["Bug ID"])
            bug_info[bid] = {
                "classification": r.get("Classification",""),
                "product":        r.get("Product",""),
                "component":      r.get("Component",""),
                "assignee":       safe_strip(r.get("Assignee","")),
                "status":         r.get("Status",""),
                "resolution":     r.get("Resolution",""),
                "num_comments":   int(r.get("Number of Comments", 0)),
                "summary":        r.get("Summary","")
            }
    with open(f"{out_prefix}_bug_info.pkl", "wb") as f:
        pickle.dump(bug_info, f)
    print(f"Saved bug_info: {len(bug_info)} entries")


    bug_history = {}
    for i in range(1,21):
        sub = os.path.join(history_dir, f"bugs{i}")
        if not os.path.isdir(sub): continue
        for csv_path in glob.glob(os.path.join(sub,"*.csv")):
            try:
                df = pd.read_csv(csv_path, on_bad_lines="skip")
            except:
                continue
            for _, r in df.iterrows():
                bid = str(r.get("BugID", os.path.splitext(os.path.basename(csv_path))[0]))
                rec = {
                    "when":   r.get("When",""),
                    "who":    safe_strip(r.get("Who","")),
                    "what":   r.get("What",""),
                    "added":  r.get("Added",""),
                    "removed":r.get("Removed","")
                }
                bug_history.setdefault(bid, []).append(rec)
    with open(f"{out_prefix}_bug_history.pkl", "wb") as f:
        pickle.dump(bug_history, f)
    print(f"Saved bug_history: {len(bug_history)} bug IDs")

    bug_desc = {}
    for txt in glob.glob(os.path.join(desc_dir, "*/*.txt")):
        bid = os.path.splitext(os.path.basename(txt))[0]
        with open(txt, encoding="utf8", errors="ignore") as f:
            raw = f.read().strip()
        if raw:
            bug_desc[bid] = preprocess_text(raw)
    with open(f"{out_prefix}_bug_desc.pkl", "wb") as f:
        pickle.dump(bug_desc, f)
    print(f"Saved bug_descriptions: {len(bug_desc)} entries")

if __name__ == "__main__":
    RAW = r"F:\Thesis\RAW DATA\MOZILLA"
    load_and_preprocess(
        buglist_dir = os.path.join(RAW,"buglist"),
        history_dir = os.path.join(RAW,"BugHistory"),
        desc_dir    = os.path.join(RAW,"Bugs"),
        out_prefix  = "Mozilla_preprocessed"
    )
