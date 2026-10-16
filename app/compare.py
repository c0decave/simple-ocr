"""Compare OCR or description results of several models: similarity, grouping and line/word diffs.

Pure functions, standard library only.
"""
import difflib
import re
import unicodedata

SIMILAR_THRESHOLD = 0.80   # word-level similarity from which two results count as "similar"
MAX_TOKENS = 3000          # comparison is capped so a huge answer cannot stall the benchmark

_TABLE_RULE = re.compile(r"^[\s|:\-]*-[\s|:\-]*$")


def normalize(text):
    """Text as compared: unicode-normalized, markdown table scaffolding and layout removed, case kept."""
    text = unicodedata.normalize("NFC", text)
    lines = [line for line in text.splitlines() if not _TABLE_RULE.match(line)]
    return re.sub(r"\s+", " ", " ".join(lines).replace("|", " ")).strip()


def similarity(a, b):
    """Word-level similarity in 0..1. 1.0 means identical after normalization."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na > nb:   # SequenceMatcher.ratio() depends on the order; a fixed order makes a ~ b equal b ~ a
        na, nb = nb, na
    ta, tb = na.split(" "), nb.split(" ")
    truncated = len(ta) > MAX_TOKENS or len(tb) > MAX_TOKENS
    ratio = difflib.SequenceMatcher(None, ta[:MAX_TOKENS], tb[:MAX_TOKENS], autojunk=False).ratio()
    return min(ratio, 0.999) if truncated else ratio


def classify(ratio):
    if ratio >= 1.0:
        return "identical"
    return "similar" if ratio >= SIMILAR_THRESHOLD else "different"


def _label(index):
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def analyze(texts):
    """texts: ordered {model: text}. Returns pairs, a similarity matrix and groups.

    A model joins the first group whose representative (its first member) it is similar to.
    """
    models = list(texts)
    n = len(models)
    values = [[1.0] * n for _ in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ratio = round(similarity(texts[models[i]], texts[models[j]]), 4)
            values[i][j] = values[j][i] = ratio
            pairs.append({"a": models[i], "b": models[j], "similarity": ratio, "class": classify(ratio)})
    groups = []   # each: {"rep": index, "members": [indexes]}
    for i in range(n):
        for group in groups:
            if values[group["rep"]][i] >= SIMILAR_THRESHOLD:
                group["members"].append(i)
                break
        else:
            groups.append({"rep": i, "members": [i]})
    return {
        "similar_threshold": SIMILAR_THRESHOLD,
        "matrix": {"models": models, "values": values},
        "tree": _tree(models, values),
        "agreement": {
            m: round(sum(values[i][j] for j in range(n) if j != i) / (n - 1), 4) if n > 1 else None
            for i, m in enumerate(models)},
        "pairs": pairs,
        "groups": [
            {
                "label": _label(k),
                "representative": models[g["rep"]],
                "models": [models[i] for i in g["members"]],
                "identical_to_representative": [
                    models[i] for i in g["members"] if i != g["rep"] and values[g["rep"]][i] >= 1.0],
            }
            for k, g in enumerate(groups)
        ],
    }


def _tree(models, values):
    """Average-linkage clustering on the distance 1 - similarity.

    Leaves are {"model": id}, inner nodes {"height": distance, "children": [left, right]}.
    Ties merge the earliest pair, so the same input always gives the same tree.
    """
    if not models:
        return None
    clusters = [({"model": m}, [i]) for i, m in enumerate(models)]
    while len(clusters) > 1:
        best = None
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                a, b = clusters[x][1], clusters[y][1]
                distance = sum(1.0 - values[i][j] for i in a for j in b) / (len(a) * len(b))
                if best is None or distance < best[0] - 1e-12:
                    best = (distance, x, y)
        distance, x, y = best
        (left, members_left), (right, members_right) = clusters[x], clusters[y]
        height = max(round(distance, 4), left.get("height", 0.0), right.get("height", 0.0))
        clusters[x] = ({"height": height, "children": [left, right]}, members_left + members_right)
        del clusters[y]
    return clusters[0][0]


def _words(line_a, line_b):
    """Token-level segments [[text, changed], ...] for both sides of a replaced line pair."""
    ta, tb = re.findall(r"\s+|\S+", line_a), re.findall(r"\s+|\S+", line_b)
    if len(ta) > MAX_TOKENS or len(tb) > MAX_TOKENS:   # the matching is quadratic: mark such lines as a whole
        return [[line_a, True]], [[line_b, True]]
    seg_a, seg_b = [], []
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        changed = tag != "equal"
        seg_a.extend([t, changed] for t in ta[a1:a2])
        seg_b.extend([t, changed] for t in tb[b1:b2])
    return seg_a, seg_b


def diff(a_text, b_text):
    """Line diff of two raw results. Replaced line pairs carry word-level change marks.

    Returns ops: {"tag": equal|replace|delete|insert, "a": [line, ...], "b": [line, ...]} where a line
    is a list of [text, changed] segments. Only the first MAX_TOKENS lines are compared; if there are more,
    a last op {"tag": "truncated", "a": [], "b": [], "lines": MAX_TOKENS} says so.
    """
    a_all = [line.rstrip() for line in a_text.splitlines()]
    b_all = [line.rstrip() for line in b_text.splitlines()]
    a_lines, b_lines = a_all[:MAX_TOKENS], b_all[:MAX_TOKENS]
    ops = []
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).get_opcodes():
        a_chunk, b_chunk = a_lines[a1:a2], b_lines[b1:b2]
        if tag == "equal":
            a_out = [[[line, False]] for line in a_chunk]
            b_out = [[[line, False]] for line in b_chunk]
        elif tag == "replace" and len(a_chunk) == len(b_chunk):
            pairs = [_words(x, y) for x, y in zip(a_chunk, b_chunk)]
            a_out, b_out = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            a_out = [[[line, True]] for line in a_chunk]
            b_out = [[[line, True]] for line in b_chunk]
        ops.append({"tag": tag, "a": a_out, "b": b_out})
    if len(a_all) > MAX_TOKENS or len(b_all) > MAX_TOKENS:
        ops.append({"tag": "truncated", "a": [], "b": [], "lines": MAX_TOKENS})
    return ops
