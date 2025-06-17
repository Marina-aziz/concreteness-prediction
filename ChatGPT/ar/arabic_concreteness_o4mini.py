# arabic_concreteness_o4mini.py
import sys
import csv
import time
import statistics
import openai
import tqdm

API_KEY = ""  # Set your OpenAI API key here
client = openai.OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"             # Model to use for prediction
OUTCSV = "arabic_concreteness_gpt.csv"  # Output CSV file

# Arabic prompt for the LLM rating task
PROMPT = """تشير بعض الكلمات إلى أشياء أو أفعال في الواقع، والتي يمكنك اختبارها مباشرة من خلال إحدى الحواس الخمس... 
الكلمة: {word}"""

def single_call(word: str) -> int:
    """
    Query the LLM once for a concreteness rating for the given word.

    Parameters:
        word (str): Arabic word to be rated.

    Returns:
        int: The returned rating (1–5).

    Raises:
        ValueError: If the reply is not a digit between 1 and 5.
    """
    rsp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1,
        messages=[{"role": "user", "content": PROMPT.format(word=word)}]
    )

    ans = rsp.choices[0].message.content.strip()
    if ans not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Unexpected reply {ans!r}")
    return int(ans)

def three_passes(word: str) -> list[int]:
    """
    Query the model three times for the same word, retrying on API errors.

    Parameters:
        word (str): Arabic word to be rated.

    Returns:
        list[int]: List of three ratings (each 1–5).
    """
    digits = []
    for _ in range(3):
        for attempt in range(3):  # Up to 3 attempts in case of API error
            try:
                digits.append(single_call(word))
                break
            except openai.OpenAIError:
                if attempt == 2:
                    raise
                time.sleep(5)
    return digits

def main(path: str):
    """
    For each Arabic word in the input file, run three LLM passes and write results to CSV.

    Parameters:
        path (str): Path to input file (one word per line).

    Output:
        Writes CSV with columns: word, run1, run2, run3, mean.
    """
    words = [w.strip() for w in open(path, encoding="utf-8") if w.strip()]
    with open(OUTCSV, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["word", "run1", "run2", "run3", "mean"])
        for w in tqdm.tqdm(words, unit="word"):
            r1, r2, r3 = three_passes(w)
            mean = round(statistics.mean([r1, r2, r3]), 3)
            wr.writerow([w, r1, r2, r3, mean])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python arabic_concreteness_o4mini.py <ar_words.txt>")
    main(sys.argv[1])
