// Project-work report (source). Build from the repository root:
//   uv run python report/make_figures.py   # figures from the results JSON
//   typst compile report/report.typ report/report.pdf

#set document(title: "Do Active Units Reflect a Known State Space? Variational Autoencoders on Blocksworld", author: "Klein Tahiraj")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.2cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.6em, spacing: 0.85em)
#show heading.where(level: 1): set text(size: 12.5pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading: set block(above: 1.3em, below: 0.7em)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none }, inset: (x: 5pt, y: 3.2pt))
#show table: set text(size: 8.8pt)
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(start)
#set sub(size: 0.75em)
#show figure: set block(above: 1.2em, below: 1.2em)
#set figure(placement: auto)
#set figure.caption(separator: [. ])
#show figure.where(kind: table): set figure.caption(position: top)


#include "sections/00_front.typ"

#include "sections/01_introduction.typ"

#include "sections/02_background.typ"

#include "sections/03_setup.typ"

#include "sections/04_results.typ"

#include "sections/05_discussion.typ"

#include "sections/07_conclusion.typ"

#include "sections/08_references.typ"

#include "sections/09_appendix.typ"
