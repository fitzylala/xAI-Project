;;; export.el --- IEEE conference export for this paper -*- lexical-binding: t; -*-

(require 'ox-latex)
(require 'oc-bibtex)

(add-to-list 'org-latex-classes
             '("ieeeconf"
               "\\documentclass[conference,10pt]{IEEEtran}
[NO-DEFAULT-PACKAGES]
[NO-PACKAGES]
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{cite}
\\usepackage{amsmath,amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage[hidelinks]{hyperref}"
               ("\\section{%s}" . "\\section*{%s}")
               ("\\subsection{%s}" . "\\subsection*{%s}")
               ("\\subsubsection{%s}" . "\\subsubsection*{%s}")))

(defun paper-prepare-build (backend)
  "Prepare the build directory and bibliography for BACKEND export."
  (when (org-export-derived-backend-p backend 'latex)
    (let* ((source-dir (file-name-directory (buffer-file-name (buffer-base-buffer))))
           (build-dir (expand-file-name "build" source-dir)))
      (make-directory build-dir t)
      (copy-file (expand-file-name "references.bib" source-dir)
                 (expand-file-name "references.bib" build-dir) t))))

(defun paper-setup-latex-export ()
  "Configure PDF compilation for the current IEEE paper buffer."
  (when (equal (org-collect-keywords '("LATEX_CLASS"))
               '(("LATEX_CLASS" "ieeeconf")))
    (make-directory
     (expand-file-name "build" (file-name-directory (buffer-file-name))) t)
    (add-hook 'org-export-before-processing-hook #'paper-prepare-build nil t)
    (setq-local org-latex-remove-logfiles nil)
    (setq-local org-latex-pdf-process
                '("latexmk -cd -pdf -interaction=nonstopmode -halt-on-error %f"))))

(add-hook 'org-mode-hook #'paper-setup-latex-export)
(when (derived-mode-p 'org-mode)
  (paper-setup-latex-export))

(provide 'paper-export)
;;; export.el ends here
