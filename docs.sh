#!/bin/bash
set -x
set -e
DIR=/tmp/fbcunn
rm -rf $DIR && mkdir -p $DIR
CURDIR=$(pwd)
echo $CURDIR
(
    cd $DIR
    dokx-build-package-docs -o . $CURDIR
    # Fix relative links in HTML to point to the CDN
    replace "../_highlight/highlight.pack.js" "//cdnjs.cloudflare.com/ajax/libs/highlight.js/8.4/highlight.min.js"
    replace "../_highlight/styles/github.css" "//cdnjs.cloudflare.com/ajax/libs/highlight.js/8.4/styles/github.min.css"

    git init
    git checkout -b gh-pages
    git add .
    git commit -m "Documentation"
    git remote add origin git@github.com:facebook/fbcunn.git
    git push --set-upstream origin gh-pages -f
)
