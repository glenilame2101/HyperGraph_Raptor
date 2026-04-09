import argparse
import os
from pymupdf4llm import to_markdown
from tqdm import tqdm


def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to chunked markdown files.")
    parser.add_argument("--input", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", default=None, help="Output directory (default: same dir as input).")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Words per chunk (default: 1000).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found.")
        return

    print("Converting PDF to markdown...")
    md_text = to_markdown(args.input)

    chunks = chunk_text(md_text, args.chunk_size)

    out_dir = args.output or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.input))[0]

    # Merge all chunks into one final file
    final_path = os.path.join(out_dir, f"{base}.md")
    with open(final_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Writing chunks"):
            f.write(chunk + "\n\n---\n\n")

    print(f"Wrote {len(chunks)} chunks merged into {final_path}")


if __name__ == "__main__":
    main()
