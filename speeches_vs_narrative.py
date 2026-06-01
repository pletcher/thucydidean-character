# %% [python] Initialize libraries, constants, and basic functions
from pathlib import Path

from MyCapytain.resources.texts.local.capitains.cts import CapitainsCtsText
from MyCapytain.common.constants import Mimetypes
import polars as pl
import spacy
from spacy import tokens as spacy_tokens

SPACY_MODEL = "grc_proiel_trf"

nlp = spacy.load(SPACY_MODEL)

SPEECHES = pl.read_csv("./alt-thuc-speeches.tsv", separator="\t").with_columns(
    pl.col("start", "end").str.split(".").cast(pl.List(pl.UInt32)),
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("index"),
)

THUCYDIDES_PARQUET = "alt-thucydides.parquet"


def get_speech_for_ref(ref: list[int]):
    for speech in SPEECHES.iter_rows(named=True):
        ref_is_after_speech_start = True
        ref_is_before_speech_end = True

        for pair in zip(speech["start"], ref):
            if pair[0] > pair[1]:
                ref_is_after_speech_start = False

        for pair in zip(speech["end"], ref):
            if pair[0] < pair[1]:
                ref_is_before_speech_end = False

        if ref_is_after_speech_start and ref_is_before_speech_end:
            return dict(
                id=speech["index"],
                speaker=speech["speaker"],
                location=speech["location"],
            )

    return None


def thuc_to_table():
    """
    See https://mycapytain.readthedocs.io/en/latest/MyCapytain.local.html
    for a fuller example.
    """
    with open("./tei_primary_sources/tlg0003.tlg001.perseus-grc2.xml") as f:
        text = CapitainsCtsText(resource=f)

    refs = [r[0] for r in text.getReffs(level=len(text.citation))]
    passages = [
        text.getTextualNode(subreference=ref, simple=True).export(
            Mimetypes.PLAINTEXT, exclude=["tei:note"]
        )
        for ref in refs
    ]
    refs = [list(map(lambda i: int(i), r.split("."))) for r in refs]
    speeches = [get_speech_for_ref(ref) for ref in refs]
    speech_ids = [s["id"] if s is not None else None for s in speeches]
    speakers = [s["speaker"] if s is not None else None for s in speeches]
    locations = [s["location"] if s is not None else None for s in speeches]

    return pl.DataFrame(
        {
            "reference": refs,
            "passage": passages,
            "speaker": speakers,
            "speech_id": speech_ids,
            "location": locations,
        }
    )


def save_df(df: pl.DataFrame) -> pl.DataFrame:
    df = thuc_to_table().with_columns(
        pl.col("passage")
        .map_elements(lambda p: spacy_tokens.DocBin(docs=[nlp(p)]).to_bytes())
        .alias("parsed_passage")
    )

    df.write_parquet(THUCYDIDES_PARQUET)

    return df


def restore_df():
    return pl.read_parquet(THUCYDIDES_PARQUET).with_columns(
        pl.col("parsed_passage").map_elements(
            lambda p: list(spacy.tokens.DocBin().from_bytes(p).get_docs(nlp.vocab))[0],
            return_dtype=pl.Object,
        )
    )


# %%
df = restore_df()
# %%
speeches_df = df.filter(pl.col("speech_id").is_not_null())
narrative_df = df.filter(pl.col("speech_id").is_null())


# %%
import string

from collections import Counter


def calculate_ttr(passage_strs, n=4):
    n_grams = []

    for p in passage_strs:
        no_punctuation = p.replace("‘", "").translate(
            str.maketrans("", "", string.punctuation)
        )
        n_grams += [no_punctuation[i : i + n] for i in range(0, len(no_punctuation), n)]

    types = len(set(n_grams))
    tokens = len(n_grams)

    return types / tokens


def calculate_true_ttr(passage_strs, n=4):
    tokens = []

    for p in passage_strs:
        no_punctuation = p.replace("‘", "").translate(
            str.maketrans("", "", string.punctuation)
        )
        tokens += [s for s in no_punctuation.split() if s.strip() != ""]

    n_types = len(set(tokens))
    n_tokens = len(tokens)

    return n_types / n_tokens


speeches_ttr = calculate_ttr(speeches_df["passage"].explode())
narrative_ttr = calculate_ttr(narrative_df["passage"].explode())

print(f"Speeches TTR: {speeches_ttr}")
print(f"Narrative TTR: {narrative_ttr}")

speeches_true_ttr = calculate_true_ttr(speeches_df["passage"].explode())
narrative_true_ttr = calculate_true_ttr(narrative_df["passage"].explode())

print(f"Speeches true TTR (space-delimited): {speeches_true_ttr}")
print(f"Narrative true TTR (space-delimited): {narrative_true_ttr}")

# %%
def speeches_to_txt():
    import re

    for speaker, location, passages in (
        speeches_df.group_by(pl.col("speaker"), pl.col("location"))
        .agg(pl.col("passage"))
        .iter_rows()
    ):
        ttr = calculate_ttr(passages)

        print(f"{speaker} in {location} TTR: {ttr}")

        true_ttr = calculate_true_ttr(passages)

        print(f"{speaker} in {location} true TTR (space-delimited): {true_ttr}")

        filename = f"speeches/{speaker}_{location}.txt"

        with open(filename, "w") as f:
            cleaned_passages = (
                "".join(passages).replace("‘", "").replace("\n", " ").replace("  ", " ")
            )

            f.write(cleaned_passages)


# %%
def lemmatize_speeches():
    SPEECHES = Path("speeches").glob("./*.txt")

    for speech in SPEECHES:
        with open(speech) as f:
            doc = nlp(f.read())

        with open(f"{str(speech).replace(speech.suffix, ".lemmatized.txt")}", "w") as f:
            lemmata = [t.lemma_ for t in doc]

            f.write("\n".join(lemmata))


# %%


def lemmatize_narrative():
    filename = Path("narrative") / "narrative.lemmatized.txt"

    with open(filename, "w") as f:
        for row in narrative_df.iter_rows():
            text = row[1]
            doc = nlp(text)

            print(
                "\n".join([t.lemma_ for t in doc if t.lemma_.strip() != ""]),
                sep="\n",
                file=f,
            )


lemmatize_narrative()
# %%
