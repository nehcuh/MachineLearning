(TeX-add-style-hook
 "matrix"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("ctexart" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("hyperref" "colorlinks" "linkcolor=black" "anchorcolor=black" "citecolor=black")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "ctexart"
    "ctexart11"
    "inputenc"
    "fontenc"
    "fixltx2e"
    "graphicx"
    "longtable"
    "float"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "marvosym"
    "wasysym"
    "amssymb"
    "booktabs"
    "hyperref"
    "listings"
    "xcolor")
   (LaTeX-add-labels
    "sec:orgfbb19ae"
    "sec:org7436bbb"
    "sec:orge71f87d"
    "sec:orge2d701c"
    "sec:orgc342fc4"
    "sec:orgf6ac5de"
    "sec:org30b810c"
    "sec:org440724b"
    "sec:orgc9531c3"
    "sec:org903d059"
    "sec:org1f0a9d2"
    "sec:org0d9b65d"
    "sec:org4b2a125"
    "sec:org79aa502"
    "sec:org0fa97ac"
    "sec:org913dc9f"
    "sec:org874a232"
    "sec:org9aacd27"
    "eq:row_partial"
    "sec:org68bbea2"
    "eq:col_partial"
    "sec:orgf0eec5c"
    "sec:org46856f0"
    "sec:orgfd265b1"
    "sec:orgf0a6f71"
    "sec:org67c1cbf"
    "sec:orgf54a920"
    "sec:orgdb21d3d"
    "sec:org8cb929b"
    "sec:orgfeadd05"
    "sec:org6bb1404"
    "sec:orgecf6e89"
    "sec:org54b4d0f"
    "sec:org220b58d"
    "sec:orgc2fa503"
    "sec:org1841c50"
    "sec:org35b6873"
    "sec:orgedf8fab"
    "sec:org62e368e"
    "sec:orga66424a"
    "sec:org063c276"
    "sec:org1a907bf"
    "sec:orgc9af2ca"
    "sec:orgbced2e5"
    "sec:org9430a65"
    "sec:orga53758e"
    "sec:orgb24a3b8"
    "sec:org095e74b"
    "sec:orgebb6055"
    "sec:org78e5c81"
    "sec:orgca6adff"
    "sec:orgcee7c58"
    "sec:org4e4905a"
    "sec:org5b787c1"
    "sec:org5d7f4ed"
    "sec:org8c71cc0"
    "sec:org8be2254"
    "eq:trace_prod"
    "sec:org4089437"
    "sec:orgd512cf7"
    "sec:org726acfa"
    "sec:orge6536ea"
    "eq:trace_two_matrix"
    "sec:orgcf8580e"
    "sec:org099d3da"
    "eq:scalar_fun_of_matrix_var"
    "sec:orgfa701ee"))
 :latex)

