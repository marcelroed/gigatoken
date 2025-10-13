import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from unicodedata import category
    return (category,)


@app.cell
def _():
    ord('ø')
    return


@app.cell
def _(category):
    category('Ø')
    return


@app.cell
def _(category):
    codepoints = []
    invalid = 0
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:  # skip surrogate range
            continue
        ch = chr(cp)
        if category(ch) == 'Cn':
            invalid += 1
            continue
        codepoints.append(ch)
    return codepoints, invalid


@app.cell
def _(codepoints):
    print(codepoints)
    return


@app.cell
def _(invalid):
    invalid
    return


@app.cell
def _(codepoints):
    len(codepoints)
    return


@app.cell
def _():
    from enum import Enum, auto

    class Group(Enum):
        LETTER = auto()
        NUMBER = auto()
        OTHER = auto()



    return


if __name__ == "__main__":
    app.run()
