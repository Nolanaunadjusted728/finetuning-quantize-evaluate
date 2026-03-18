// template.typ — Modern technical report template for LLM Fine-Tuning Bootcamp v6
#import "@preview/meander:0.4.1"

// Color scheme
#let accent = rgb("#1a5276")
#let accent-light = rgb("#d4e6f1")
#let accent-muted = rgb("#2e86c1")
#let code-bg = rgb("#f8f8f8")
#let code-border = rgb("#e0e0e0")
#let blockquote-bg = rgb("#f0f4f8")
#let table-header-bg = rgb("#1a5276")
#let table-alt-bg = rgb("#f5f8fc")

// Blockquote function
#let blockquote(body) = {
  block(
    width: 100%,
    above: 0.8em,
    below: 0.8em,
    fill: blockquote-bg,
    stroke: (left: 3pt + accent),
    inset: (left: 14pt, right: 12pt, top: 10pt, bottom: 10pt),
    radius: (right: 4pt),
    body,
  )
}

// Caption styling for figures/equations/code
#let caption-text(body) = {
  text(size: 9.5pt, style: "italic", fill: luma(80))[#body]
}

// Two-column reflow using meander (for glossary, key takeaways, etc.)
#let two-col(body) = {
  meander.reflow({
    import meander: *
    opt.placement.spacing(below: 0.65em)
    container(width: 48%, margin: 3mm)
    container(width: 48%, margin: 3mm)
    content[#body]
  })
}

// Wrapped figure using meander — image on one side, caption below, text flows around
#let flow-figure(img-path, img-width: 40%, position: top + right, caption-text: none, body) = {
  meander.reflow({
    import meander: *
    opt.placement.spacing(below: 0.65em)
    placed(position, {
      block(inset: 4pt)[
        #image(img-path, width: img-width)
        #if caption-text != none {
          align(center)[
            #text(size: 9pt, style: "italic", fill: luma(100))[#caption-text]
          ]
        }
      ]
    })
    container()
    content[#body]
  })
}

// Project template
#let project(
  title: "",
  subtitle: none,
  author: "",
  date: none,
  body,
) = {
  set page(
    paper: "a4",
    margin: (left: 2cm, right: 2cm, top: 2.5cm, bottom: 2cm),
    header: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        align(center)[
          #text(size: 8pt, fill: luma(140))[Fine-Tune, Quantize, Evaluate: The Complete Guide]
        ]
        v(-0.3em)
        line(length: 100%, stroke: 0.3pt + luma(200))
      }
    },
    footer: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        grid(
          columns: (1fr, 1fr, 1fr),
          [],
          align(center)[#text(size: 9pt, fill: luma(140))[#counter(page).display("1")]],
          align(right)[#text(size: 8pt, fill: luma(140))[Created by Isham Rashik]],
        )
      }
    },
  )

  set text(
    font: "New Computer Modern",
    size: 10.5pt,
    lang: "en",
  )
  set par(justify: true, leading: 0.85em, spacing: 1.1em)

  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1.2em)
    text(size: 20pt, weight: "bold", fill: accent)[#it.body]
    v(0.5em)
    line(length: 100%, stroke: 1.5pt + accent)
    v(1em)
  }
  show heading.where(level: 2): it => {
    pagebreak(weak: true)
    v(1.2em)
    text(size: 15pt, weight: "bold", fill: accent)[#it.body]
    v(0.8em)
  }
  show heading.where(level: 3): it => {
    v(1.2em)
    text(size: 12.5pt, weight: "bold", fill: accent-muted)[#it.body]
    v(0.6em)
  }
  show heading.where(level: 4): it => {
    v(1em)
    text(size: 11pt, weight: "bold", fill: accent-muted.lighten(15%))[#it.body]
    v(0.5em)
  }

  // Code blocks
  show raw.where(block: true): it => {
    block(
      width: 100%,
      above: 1.25em,
      below: 1.3em,
      fill: code-bg,
      stroke: 0.5pt + code-border,
      inset: 10pt,
      radius: 4pt,
    )[
      #text(size: 8.5pt, font: "DejaVu Sans Mono")[#it]
    ]
  }

  // Code listing figures — left-aligned body, centered caption
  show figure.where(kind: raw): it => {
    align(left, it.body)
    align(center, it.caption)
  }

  // Table figures — caption on top, breakable across pages with repeating headers
  show figure.where(kind: table): set block(breakable: true)
  show figure.where(kind: table): it => {
    align(center, it.caption)
    v(0.4em)
    it.body
  }

  // Inline code
  show raw.where(block: false): it => {
    box(
      fill: code-bg,
      inset: (x: 3pt, y: 0pt),
      outset: (y: 3pt),
      radius: 2pt,
    )[
      #text(size: 9pt, font: "DejaVu Sans Mono")[#it]
    ]
  }

  // Links
  show link: it => {
    text(fill: accent-muted)[#underline(it)]
  }

  // Figure styling
  set figure(gap: 0.7em)
  show figure.caption: it => {
    text(size: 9.5pt, style: "italic")[#it]
  }

  // --- TITLE PAGE ---
  v(4cm)
  align(center)[
    #text(size: 26pt, weight: "bold", fill: accent)[#title]
  ]
  v(1.2em)
  if subtitle != none {
    align(center)[
      #text(size: 11pt, fill: luma(80))[#subtitle]
    ]
  }
  v(2em)
  if author != "" {
    align(center)[
      #text(size: 12pt, weight: "medium")[Author: #author]
    ]
  }
  v(0.5em)
  if date != none {
    align(center)[
      #text(size: 11pt, fill: luma(120))[Date: #date]
    ]
  }
  v(2cm)
  align(center)[
    #line(length: 40%, stroke: 0.5pt + luma(180))
  ]

  pagebreak()

  // --- TABLE OF CONTENTS ---
  {
    align(center)[
      #text(size: 20pt, weight: "bold", fill: accent)[Table of Contents]
    ]
    v(0.4em)
    line(length: 100%, stroke: 1.5pt + accent)
    v(0.6em)
    outline(depth: 3, indent: auto, title: none)
  }

  pagebreak()

  // --- BODY ---
  body
}
