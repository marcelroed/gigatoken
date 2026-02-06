import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # Used to read files, don't redefine often
    from xopen import xopen
    from emoji import demojize
    from pathlib import Path
    from collections import Counter
    return Counter, Path, demojize, xopen


@app.function
def cp_length_blocks(data, block_size: int = 256, threshold_length: int = 1, show_errors: bool = False):
    successes = 0
    total = 0
    for window_start in range(0, len(data), block_size):
        window = data[window_start:window_start+block_size]
        for c in window:
            if c < 128:
                clen = 1
            elif c < 0b11100000:
                clen = 2
            elif c < 0b11110000:
                clen = 3
            else:
                clen = 4
            if clen > threshold_length:
                if show_errors:
                    try:
                        print('[', window.decode(), ']', sep='')
                    except:
                        pass
                break
        else:
            successes += 1
        total += 1
    return successes / total


@app.cell
def _():
    def proportion_matches(data, fail_condition: callable, block_size: int = 64, show_errors: bool = False):
        successes = 0
        total = 0
        for window_start in range(0, len(data), block_size):
            window = data[window_start:window_start + block_size]
            for c in window:
                if fail_condition(c):
                    if show_errors:
                        try:
                            print('[', window.decode(), ']', sep='')
                        except:
                            pass
                    break
            else:
                successes += 1
            total += 1
        return successes / total

    # print(cp_length_blocks(Path("/Users/marcel/data/TinyStoriesV2-GPT4-valid.txt").read_bytes(), 2048))
    return (proportion_matches,)


@app.cell
def _(Path, demojize):
    owt = Path("/Users/marcel/data/owt_valid.txt").read_text()
    owt = demojize(owt)
    owt = owt.encode('utf-8')
    return (owt,)


@app.cell
def _(owt):
    print(cp_length_blocks(owt, 512, 3, show_errors=True))
    return


@app.cell
def _():
    # s = Path("/Users/marcel/data/owt_valid.txt").read_text()
    # for c in s:
        # b = c.encode('utf-8')
        # if len(b) == 2:
            # print(c)
    return


@app.cell
def _(demojize, xopen):
    cc_text = xopen("/Users/marcel/data/CC-MAIN-20251005114239-20251005144239-00000.warc.wet.gz", "r").read()
    cc_demojized = demojize(cc_text)
    cc_demojized = cc_demojized.encode('utf-8')
    cc = cc_text.encode('utf-8')
    return cc, cc_text


@app.cell
def _(cc):
    print(cp_length_blocks(cc, 64, 3, show_errors=False))
    return


@app.cell
def _(cc, proportion_matches):
    def longer_than_2(byte):
        return byte >= 0b11100000
    def longer_than_3(byte):
        return byte >= 0b11110000
    print('Proportion 2 or shorter characters', proportion_matches(cc, longer_than_2, 256))
    print('Proportion 3 or shorter characters', proportion_matches(cc, longer_than_3, 256))
    return


@app.cell
def _(Counter, cc_text):
    # Count each codepoint in the archive file
    def count_cps(text: str):
        counts = Counter((ord(c) for c in text))
        return counts

    cc_counts = count_cps(cc_text)
    return (cc_counts,)


@app.cell
def _(cc_counts):
    from json import dump
    with open("/Users/marcel/data/cc_codepoint_counts.json", "w") as f:
        dump(cc_counts, f)
    return


@app.cell
def _(cc_counts):
    list(sorted(cc_counts.items(), key=lambda e: e[0]))
    return


if __name__ == "__main__":
    app.run()
