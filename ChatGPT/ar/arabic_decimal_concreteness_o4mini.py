import sys
import csv
import time
import statistics
import openai
import tqdm
import math
import re

API_KEY = ""  # Set your OpenAI API key here
client = openai.OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"                    # Model for prediction
OUTCSV = "arabic_concreteness_gpt.csv"   # Output CSV file

# Arabic prompt for decimal concreteness ratings
PROMPT = """تشير بعض الكلمات إلى أشياء أو أفعال في الواقع، والتي يمكنك اختبارها مباشرة من خلال إحدى الحواس الخمس... 
دائمًا فكّر في مدى محسوسية معنى الكلمة بالنسبة لك. نطلب منك استخدام مقياس تقييم مستمر من 1.00 (مجردة جدًا) إلى 5.00 (محسوسة جدًا) لتقييم هذه الكلمة. استخدم مقياسًا
مستمرًا من ‎1.00‎ إلى ‎5.00‎. أجب **فقط** بعدد عشري يحوي خانتين
عشريتين بالضبط، مثل ‎3.14‎ أو ‎2.76‎.
الكلمة: {word}"""

def single_call(word: str) -> float:
    """
    Query the LLM once for a concreteness rating (1.00–5.00, two decimals) for the given word.

    Parameters:
        word (str): Arabic word to be rated.

    Returns:
        float: The returned rating, rounded to two decimals.

    Raises:
        ValueError: If the reply is not a valid decimal in range.
    """
    rsp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=4,
        messages=[{"role": "user", "content": PROMPT.format(word=word)}]
    )

    ans = rsp.choices[0].message.content.strip().replace("٫", ".")  # Convert Arabic decimal separator

    # Must match pattern 1.00 – 5.00, exactly two decimals
    if not re.fullmatch(r"[1-5](?:\.[0-9]{2})?", ans):
        raise ValueError(f"Unexpected reply {ans!r}")

    val = float(ans)
    if not (1.0 <= val <= 5.0):
        raise ValueError(f"Out-of-range value {val}")

    return round(val, 2)

def three_passes(word: str) -> list[float]:
    """
    Query the model three times for the same word, retrying on API errors.

    Parameters:
        word (str): Arabic word to be rated.

    Returns:
        list[float]: List of three decimal ratings (each between 1.00 and 5.00).
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
