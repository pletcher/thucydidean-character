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
            lambda p: list(spacy.tokens.DocBin().from_bytes(p).get_docs(nlp.vocab))[0],
            return_dtype=pl.Object,
        )
    )


# %% [python]
df = restore_df()
# %%


def clause_has_an(token):
    return any(
        "ἄν" == t.text for t in token.sent if t.head == token.head or t.head == token
    )


def count_finite_potential_optatives(tokens: list[spacy_tokens.Token]) -> int:
    n_potential_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("Mood") == "Opt"
            and ("ἄν" in [t.lemma_ for t in token.children] or clause_has_an(token))
        ):
            n_potential_optatives += 1

    return n_potential_optatives


def count_possible_participial_potential_optatives(
    tokens: list[spacy_tokens.Token],
) -> int:
    n_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("VerbForm") == "Part"
            and ("ἄν" in [t.lemma_ for t in token.children] or clause_has_an(token))
        ):
            n_optatives += 1

    return n_optatives


def count_possible_infinitival_potential_optatives(
    tokens: list[spacy_tokens.Token],
) -> int:
    n_optatives = 0

    for token in tokens:
        if (
            token.pos_ == "VERB"
            and token.morph.to_dict().get("VerbForm") == "Inf"
            and ("ἄν" in [t.lemma_ for t in token.children] or clause_has_an(token))
        ):
            n_optatives += 1

    return n_optatives


df = df.with_columns(
    pl.col("parsed_passage")
    .map_elements(count_finite_potential_optatives, return_dtype=pl.Int64)
    .alias("n_pot_opt"),
    pl.col("parsed_passage")
    .map_elements(count_possible_participial_potential_optatives, return_dtype=pl.Int64)
    .alias("n?_part_opt"),
    pl.col("parsed_passage")
    .map_elements(count_possible_infinitival_potential_optatives, return_dtype=pl.Int64)
    .alias("n?_inf_opt"),
).with_columns(
    (pl.col("n_pot_opt") + pl.col("n?_part_opt") + pl.col("n?_inf_opt")).alias(
        "n_pot_opt_total"
    )
)

# %% [markdown]
# Count potential optatives. We can be fairly certain that finite-optatives
# with ἄν are potential, but we also need to count possible matches with
# participles and infinitives. These will require manual confirmation.


# %% [python]
df.select(pl.col("n_pot_opt", "n?_part_opt", "n?_inf_opt")).sum()

# %% [python]
speech_stats = (
    df.filter(pl.col("speech_id").is_not_null())
    .with_columns(
        pl.col("parsed_passage")
        .map_elements(len, return_dtype=pl.UInt32)
        .alias("n_tokens"),
        (pl.col("n_pot_opt") + pl.col("n?_part_opt") + pl.col("n?_inf_opt")).alias(
            "n_pot_opt_total"
        ),
    )
    .with_columns(
        (pl.col("n_pot_opt_total") / pl.col("n_tokens") * 1000).alias(
            "opt_per_1000_tokens"
        ),
    )
    .group_by("speech_id", "speaker", "location")
    .agg(
        pl.col("n_pot_opt_total").sum(),
        pl.col("n_tokens").sum(),
    )
    .with_columns(
        (pl.col("n_pot_opt_total") / pl.col("n_tokens") * 1000).alias(
            "opt_per_1000_tokens"
        ),
    )
)

# %%
speech_stats.sort("n_pot_opt_total", descending=True)
# %%
total_pot_optatives = speech_stats["n_pot_opt_total"].sum()
total_tokens_in_speeches = speech_stats["n_tokens"].sum()

expected_ratio = total_pot_optatives / total_tokens_in_speeches
expected_frequency_per_1000_tokens = expected_ratio * 1000

# %%
speech_stats.with_columns(
    (pl.col("n_tokens") * expected_ratio).alias("expected_pot_opt_total"),
).with_columns(
    (pl.col("n_pot_opt_total") - pl.col("expected_pot_opt_total")).alias(
        "actual - expected"
    )
).sort(
    "actual - expected"
)  # .write_csv("thucydidean_hedges.csv")
# %%
## dispersion

import math
import numpy as np


# def dispersion(grouped_df):
#     parsed_passage = grouped_df["parsed_passage"]
#     rel_freq_pot_opt = []

#     for passage in parsed_passage:
#         n_optatives = (
#             count_finite_potential_optatives(passage)
#             + count_possible_infinitival_potential_optatives(passage)
#             + count_possible_participial_potential_optatives(passage)
#         ) * 2  # multiply by two because each pot. optative is two tokens
#         passage_length = len(passage)

#         rel_freq_pot_opt.append(n_optatives / passage_length)

#     # numpy calculates the population std. dev.
#     # by default, which is what we want here
#     std_dev = np.std(rel_freq_pot_opt)
#     mean = np.mean(rel_freq_pot_opt)

#     var_coef = std_dev / mean
#     corpus_size = len(parsed_passage)

#     juilland_d = 1 - (var_coef * (1 / math.sqrt(corpus_size - 1)))


# df.filter(pl.col("speech_id").is_not_null(), pl.col("n_pot_opt_total") > 0).select(
#     ["reference", "speech_id", "speaker", "location", "parsed_passage"]
# ).group_by(["speech_id", "speaker", "location"]).map_groups(dispersion)

(
    df.filter(pl.col("speech_id").is_not_null(), pl.col("n_pot_opt_total") > 0)
    .select(["reference", "speech_id", "speaker", "location", "parsed_passage"])
    .with_columns(
        pl.col("parsed_passage")
        .map_elements(
            lambda p: (
                count_finite_potential_optatives(p)
                + count_possible_infinitival_potential_optatives(p)
                + count_possible_participial_potential_optatives(p)
            )
            * 2,
            return_dtype=pl.Int64,
        )
        .alias("n_opt"),
        pl.col("parsed_passage")
        .map_elements(len, return_dtype=pl.Int64)
        .alias("passage_length"),
    )
    .with_columns((pl.col("n_opt") / pl.col("passage_length")).alias("opt_rel_freq"))
    .group_by(["speech_id", "speaker", "location"])
    .agg(
        pl.col("opt_rel_freq")
        .std(ddof=0)
        .alias("std_dev"),  # population std dev, matching np.std default
        pl.col("opt_rel_freq").mean().alias("mean"),
        pl.col("opt_rel_freq").count().alias("corpus_size"),
    )
    .with_columns(
        (pl.col("std_dev") / pl.col("mean")).alias("var_coef"),
    )
    .with_columns(
        (1 - (pl.col("var_coef") / (pl.col("corpus_size") - 1).sqrt())).alias(
            "juilland_d"
        )
    )
).sort("juilland_d", descending=True) # .write_csv("./optative_juilland_d.csv")
