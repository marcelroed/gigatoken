import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import tokenizers
    from pathlib import Path
    return (Path,)


@app.cell
def _(Path):
    r50k_string = Path("/Users/marcel/data/tokenizers/r50k_base.tiktoken").read_text()
    return (r50k_string,)


@app.cell
def _(r50k_string):
    from base64 import b64decode

    data = []
    tokens = []
    for line in r50k_string.splitlines():
        t_b64, i = line.split(' ')
        t = b64decode(t_b64)
        i = int(i)
        tokens.append(t)
    return (tokens,)


@app.cell
def _(tokens):
    import polars as pl
    df = pl.DataFrame(data=[[repr(t) for t in tokens]]).with_row_index()
    df
    return


@app.cell
def _(tokens):
    tokens[1212]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
