
#align(center)[
  #text(size: 16pt, weight: "bold")[Do Active Units Reflect a Known State Space?\ Variational Autoencoders on Blocksworld]
  #v(0.6em)
  #text(size: 11.5pt)[Klein Tahiraj]
  #v(0.1em)
  #text(size: 9.5pt)[#link("mailto:klein.tahiraj@studio.unibo.it")[klein.tahiraj\@studio.unibo.it]]
  #v(0.2em)
  #text(size: 9.5pt)[Project Work in Deep Learning (3 CFU), Prof. Andrea Asperti, Università di Bologna\ November 2026]
]

#v(0.8em)
#block(inset: (x: 1.2em))[
#text(size: 9.5pt)[
*Abstract.* Asperti suggested that, for a given architecture, there seems to be an internal dimension of data, which the active units of a variational autoencoder might reflect. We test this idea on Blocksworld, whose 866 legal states are all known, by training β-VAEs on its states and on two controls, and by comparing the number of active units with dimensions we fixed before training. At β = 1, five of our six Blocksworld settings keep a median of 15 or 16 of their 32 latents active, which matches none of these dimensions. The learning rate in one network, the encoding of the states and, in a single-seed pilot, the KL warm-up move this count. In a follow-up, ten independent bits give no active unit when each is stored once and ten when each is copied. Most active units of our reference configuration detect a block on another at one end of their range and the reverse at the other. In the main sweep the β-VAEs at β = 0.25 and 1 reconstruct every state from their posterior means, but their prior samples are often illegal.
]]
