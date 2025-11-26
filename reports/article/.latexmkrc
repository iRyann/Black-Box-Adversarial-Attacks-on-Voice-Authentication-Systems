# --- Sorties & dossiers ---
$out_dir = 'build';            # PDF, logs finaux utiles (.log, .bbl, etc.)
$aux_dir = 'build/.aux';       # Auxiliaires (aux, fls, fdb_latexmk, etc.)

# --- Bibliographie (BibTeX) ---
$bibtex = 'bibtex %O %B';

# --- Nettoyage : étends ce que -c / -C suppriment ---
push @clean_ext, qw(synctex.gz bbl run.xml bcfl);
push @clean_full_ext, qw(
  aux fls fdb_latexmk log toc lof lot out nav snm
  idx ilg ind glg glo gls acr acn alg ist xdy
  vrb td0 td1 td2 td3 thm auxlock
  blg bcf xml
);

# --- Paquets spécifiques ---
# minted (Pygments)
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';

# --- Qualité de vie ---
$preview_continuous = 1;
$recorder = 1;
$emulate_aux_dir = 1;

