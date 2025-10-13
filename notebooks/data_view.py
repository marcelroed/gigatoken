import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from pathlib import Path


    def pure_ascii_blocks(data, block_size: int = 256):
        successes = 0
        total = 0
        for window_start in range(0, len(data), block_size):
            window = data[window_start:window_start+block_size]
            for c in window:
                if c >= 128:
                    break
            else:
                successes += 1
            total += 1
        return successes / total

    print(pure_ascii_blocks(Path("/Users/marcel/data/TinyStoriesV2-GPT4-valid.txt").read_bytes(), 2048))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
