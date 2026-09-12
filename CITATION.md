## CITATION
The preferred citation form is `Chen & Cunnington et al. (2026)`, and we appreciate if you take the time to define your `latex` citation alias to reflect that. For example, for most astronomy journals the following `bibtex` entry and alias will generate the correct citation:

```latex
@ARTICLE{2026arXiv260701864C,
       author = {{Chen}, Zhaoting and {Cunnington}, Steven and others},
        title = "{meer21cm: an Analysis Pipeline and Comprehensive Toolkit for HI Intensity Mapping}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = jul,
          eid = {arXiv:2607.01864},
        pages = {arXiv:2607.01864},
          doi = {10.48550/arXiv.2607.01864},
archivePrefix = {arXiv},
       eprint = {2607.01864},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260701864C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

And define the following alias in your `tex` file:
```latex
\defcitealias{2026arXiv260701864C}{Chen and Cunnington et~al.}
% Alias + year wrappers (use after \defcitealias{key}{...})
\newcommand{\citetaliasyear}[1]{\citetalias{#1}~(\citeyear{#1})}  % Alias (year)
\newcommand{\citepaliasyear}[1]{(\citetalias{#1}~\citeyear{#1})}   % (Alias year)
```

You can then use `citepaliasyear{2026arXiv260701864C}` and `citetaliasyear{2026arXiv260701864C}` accordingly.
