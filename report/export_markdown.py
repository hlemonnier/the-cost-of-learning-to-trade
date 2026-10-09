"""Generate the GitHub-readable main report from its exact editable TeX source."""
from pathlib import Path
import re
import pypandoc

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / 'Market_Making_Study.tex').read_text()

    def expand(match):
        path = ROOT / match.group(1)
        if path.suffix != '.tex':
            path = path.with_suffix('.tex')
        return path.read_text()

    source = re.sub(r'\\input\{([^}]+)\}', expand, source)
    if r'\input{' in source:
        raise ValueError('Unexpanded nested report input')
    text = pypandoc.convert_text(source, 'gfm+tex_math_dollars', format='latex',
                                extra_args=['--wrap=none'])
    if len(re.findall(r'^# [1-8]\. ', text, re.M)) != 8:
        raise ValueError('Missing report section')
    for path in re.findall(r'<img src="([^"]+)"', text):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    header = ('# Main report — text edition for repository review\n\n'
              'Generated from the final LaTeX source with numerical inputs expanded. '
              'The PDF supplies authoritative pagination; the source and mathematical '
              'appendix supply the notation. Figures are the saved report figures.\n\n')
    (ROOT / 'MAIN_REPORT.md').write_text(header + text)


if __name__ == '__main__':
    main()
