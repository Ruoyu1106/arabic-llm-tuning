from datasets import load_dataset
import os

def download_kids_books():
    ds = load_dataset("mohres/The_Arabic_E-Book_Corpus", split="train")
    kids = ds.filter(lambda x: "children.stories" in x["category"])
    save_dir = "../data/raw/arabic_ebooks_kids"
    os.makedirs(save_dir, exist_ok=True)
    for idx, book in enumerate(kids):
        fname = os.path.join(save_dir, f"book_{idx+1:04d}.txt")
        with open(fname, "w", encoding="utf‑8") as f:
            f.write(book["text"])
    print(f"Downloaded {len(kids)} kids books to {save_dir}")

if __name__ == "__main__":
    download_kids_books()
