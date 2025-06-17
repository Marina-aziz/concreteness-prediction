# english_concreteness_o4mini.py
import sys
import csv
import time
import statistics
import openai
import tqdm

API_KEY = ""  # Set your OpenAI API key here
client = openai.OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"                   # Model used for prediction
OUTCSV = "english_concreteness_gpt.csv" # Output CSV file

# English prompt for the LLM rating task
PROMPT = """Some words refer to things or actions in reality, which you can experience directly through one of the five senses. We call these words concrete words.
Other words refer to meanings that cannot be experienced directly but which we know because the meanings can be defined by other words. These are abstract words.
Still, other words fall in-between the two extremes, because we can experience them to some extent and in addition, we rely on language to understand them.
We want you to indicate how concrete the meaning of a word is for you by using a 5-point rating scale going from abstract to concrete.
A concrete word comes with a higher rating and refers to something that exists in reality; you can have immediate experience of it through your senses (smelling, tasting, touching, hearing, seeing) and the actions you do. The easiest way to explain a word is by pointing to it or by demonstrating it. For example: 
•	    To explain sweet you could have someone eat sugar;
•	    To explain jump you could simply jump up and down or show people a movie clip about someone jumping up and down;
•	    To explain couch, you could point to a couch or show a picture of a couch.
An abstract word comes with a lower rating and refers to something you cannot experience directly through your senses or actions. Its meaning depends on language. The easiest way to explain it is by using other words. For example, there is no simple way to demonstrate justice; but we can explain the meaning of the word by using other words that capture parts of its meaning.
Always think of how concrete the meaning of the word is to you. So, we ask you to use a 5-point rating scale going from 1 (very abstract) to 5 (very concrete). Answer with only a digit from 1 to 5.
Word: {word}"""

def single_call(word: str) -> int:
    """
    Query the LLM once for a concreteness rating for the given word.

    Parameters:
        word (str): English word to be rated.

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
        word (str): English word to be rated.

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
    For each English word in the input file, run three LLM passes and write results to CSV.

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
        sys.exit("Usage: python english_concreteness_o4mini.py <en_words.txt>")
    main(sys.argv[1])
