;; publish.el
;; Emacs publish file for project.
;; Run the foloowing command to execute:
;; emacs --batch --no-init-file --load publish.el --funcall org-publish-all

;; Packages:
(require 'ox-publish)

;; Switches off use of time-stamps when publishing. I would prefer to publish
;; everything every time
(setq org-publish-use-timestamps-flag nil)

;; Automatically run source code blocks:
(setq org-confirm-babel-evaluate nil)

;; Handle new LaTeX commands in both pdf and html export.
(add-to-list 'org-src-lang-modes '("latex-macros" . latex))

(defvar org-babel-default-header-args:latex-macros
  '((:results . "raw")
    (:exports . "results")))

(defun prefix-all-lines (pre body)
  (with-temp-buffer
    (insert body)
    (string-insert-rectangle (point-min) (point-max) pre)
    (buffer-string)))

(defun org-babel-execute:latex-macros (body _params)
  (concat
   (prefix-all-lines "#+LATEX_HEADER: " body)
   "\n#+HTML_HEAD_EXTRA: <div style=\"display: none\"> \\(\n"
   (prefix-all-lines "#+HTML_HEAD_EXTRA: " body)
   "\n#+HTML_HEAD_EXTRA: \\)</div>\n"))

;; Define a publishing project for the site
(setq org-publish-project-alist
      '(("org-notes"
         :base-directory "."
         :base-extension "org"
         :publishing-directory "."
         :recursive t
         :publishing-function org-html-publish-to-html
         ;:auto-sitemap t
         ;:sitemap-filename "sitemap.org"
         ;:sitemap-title "Sitemap"
         ;:sitemap-sort-folders last
         :headline-levels 4)
        ("org-static"
         :base-directory "."
         :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf"
         :publishing-directory "."
         :recursive t
         :publishing-function org-publish-attachment)
        ("org-site" :components ("org-notes" "org-static"))))

(provide 'publish)
