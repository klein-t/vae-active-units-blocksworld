#set par(justify: false, spacing: 0.35em, leading: 0.4em)
#set text(size: 7.5pt)
// Two balanced columns: the block's height is the room left on this page plus
// half of what overflows onto the next one, so both last columns end level.
#let bib = bibliography("../refs.bib", style: "ieee", title: [References])
#layout(size => context {
  let gutter = 1.4em
  let col = measure(block(width: (size.width - gutter) / 2, bib)).height
  let room = page.height - 2.2cm - here().position().y
  let h = if col <= 2 * room { col / 2 } else { room + (col - 2 * room) / 2 }
  block(height: h + 3em, breakable: true, columns(2, gutter: gutter, bib))
})
