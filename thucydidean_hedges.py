# %% [python] Initialize libraries, constants, and basic functions

from MyCapytain.resources.texts.local.capitains.cts import CapitainsCtsText
from MyCapytain.common.constants import Mimetypes
import polars as pl
import spacy
from spacy import tokens as spacy_tokens

SPACY_MODEL = "grc_proiel_trf"

nlp = spacy.load(SPACY_MODEL)

SPEECHES = pl.read_csv("./thuc-speeches.tsv", separator="\t").with_columns(
    pl.col("start", "end").str.split(".").cast(pl.List(pl.UInt32)),
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("index"),
)

THUCYDIDES_PARQUET = "thucydides.parquet"

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
        # We initiate a Text object giving the IO instance to resource argument
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
            lambda p: list(spacy_tokens.DocBin().from_bytes(p).get_docs(nlp.vocab))[0])
    )

# %% [python]
df = restore_df()
# %%

def count_finite_potential_optatives(tokens: list[spacy_tokens.Token]) -> int:
    n_potential_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("Mood") == "Opt"
            and "ἄν" in [t.lemma_ for t in token.children]
        ):
            n_potential_optatives += 1

    return n_potential_optatives

def count_possible_participial_potential_optatives(tokens: list[spacy_tokens.Token]) -> int:
    n_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("VerbForm") == "Part"
            and "ἄν" in [t.lemma_ for t in token.children]
        ):
            n_optatives += 1

    return n_optatives


def count_possible_infinitival_potential_optatives(tokens: list[spacy_tokens.Token]) -> int:
    n_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("VerbForm") == "Inf"
            and "ἄν" in [t.lemma_ for t in token.children]
        ):
            n_optatives += 1

    return n_optatives


df = df.with_columns(
    pl.col("parsed_passage").map_elements(
        count_finite_potential_optatives,
        return_dtype=pl.Int64
    ).alias("n_pot_opt"),
    pl.col("parsed_passage").map_elements(
        count_possible_participial_potential_optatives,
        return_dtype=pl.Int64
    ).alias("n?_part_opt"),
    pl.col("parsed_passage").map_elements(
        count_possible_infinitival_potential_optatives,
        return_dtype=pl.Int64
    ).alias("n?_inf_opt"),
)

# %% [markdown]
# Count potential optatives. We can be fairly certain that finite-optatives
# with ἄν are potential, but we also need to count possible matches with
# participles and infinitives. These will require manual confirmation.


# %% [python]
df.filter(pl.col("speech_id") == 100).select(pl.col("n_pot_opt", "n?_part_opt", "n?_inf_opt", "reference", "passage")).sum()
