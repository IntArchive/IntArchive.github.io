# Blog Documentation

Personal blog and research portfolio of **Dinh Minh Hai**, built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme. Supports LaTeX math via MathJax 3 and syntax-highlighted code blocks via Chroma.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [Local Development](#3-local-development)
4. [Configuration](#4-configuration)
5. [Writing Posts](#5-writing-posts)
6. [Math with LaTeX](#6-math-with-latex)
7. [Code Snippets](#7-code-snippets)
8. [Profile Page](#8-profile-page)
9. [Deployment to GitHub Pages](#9-deployment-to-github-pages)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Structure

```
IntArchive.github.io/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions — auto-deploy on push
├── assets/
│   └── css/
│       └── extended/
│           ├── custom.css      # Custom styles (typography, cards, code)
│           └── syntax.css      # Chroma syntax highlighting (generated)
├── content/
│   ├── archives.md             # /archives/ page
│   ├── search.md               # /search/   page
│   └── posts/                  # All blog posts go here
│       └── my-post.md
├── layouts/
│   ├── index.html              # Custom profile homepage
│   └── partials/
│       ├── math.html           # MathJax 3 script + macros
│       └── extend_head.html    # Injects math and fonts into <head>
├── static/
│   └── images/
│       └── avatar.jpg          # Your profile photo (replace this)
├── themes/
│   └── PaperMod/               # Git submodule — do not edit directly
├── hugo.yaml                   # Main site configuration
└── DOCS.md                     # This file
```

---

## 2. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Hugo | Latest **Extended** | Must be extended for SCSS processing |
| Git | Any recent | Needed for submodule management |
| Node.js | Optional | Not required unless adding npm tooling |

### Install Hugo Extended

```bash
# macOS
brew install hugo

# Windows
scoop install hugo-extended

# Linux (snap)
snap install hugo --channel=extended

# Verify — output must contain "extended"
hugo version
```

---

## 3. Local Development

### First-time setup

```bash
# 1. Clone your repository
git clone https://github.com/<username>/<username>.github.io.git
cd <username>.github.io

# 2. Fetch the PaperMod theme (submodule)
git submodule update --init --recursive

# 3. Generate syntax highlighting CSS
hugo gen chromastyles --style=dracula > assets/css/extended/syntax.css

# 4. Start the dev server
hugo server -D
```

Your blog is now running at **http://localhost:1313**.

The server hot-reloads on every file save — no need to restart.

### Common dev commands

```bash
# Serve including draft posts
hugo server -D

# Serve and allow access from other devices on your network
hugo server -D --bind 0.0.0.0

# Full re-render on every change (slower but safer for debugging layouts)
hugo server -D --disableFastRender

# Build the final static site into /public (for manual deployment)
hugo --minify
```

---

## 4. Configuration

All site-level settings live in **`hugo.yaml`** at the project root.

### Key fields

```yaml
baseURL: "https://intarchive.github.io/"   # Must match your GitHub Pages URL
title:   "Dinh Minh Hai"
theme:   "PaperMod"

params:
  math: true                  # Enable MathJax globally on every page
  ShowCodeCopyButtons: true   # Show copy button on all code blocks
  ShowToc: true               # Show table of contents on posts
```

### Change syntax highlight theme

1. Pick a theme name from https://xyproto.github.io/splash/docs/
2. Update `hugo.yaml`:
   ```yaml
   markup:
     highlight:
       style: monokai   # change this
   ```
3. Regenerate the CSS:
   ```bash
   hugo gen chromastyles --style=monokai > assets/css/extended/syntax.css
   ```

Available themes worth trying: `dracula`, `monokai`, `github-dark`, `nord`, `solarized-dark`, `one-dark`.

### Add a navigation link

```yaml
menu:
  main:
    - name: About
      url: /about/
      weight: 25    # higher weight = further right
```

---

## 5. Writing Posts

### Create a new post

```bash
hugo new content posts/my-post-title.md
```

This creates `content/posts/my-post-title.md` pre-filled with front matter.

### Front matter reference

```yaml
---
title: "Your Post Title"
date: 2025-06-01
draft: false          # set to false to publish; true = hidden unless -D flag

math: true            # enable LaTeX rendering for this post
tags: ["math", "ai"]  # shown as tags, also powers /tags/ page
description: "A short summary shown in post previews and SEO meta tags."

showToc: true         # show table of contents (default: inherits site setting)
TocOpen: false        # whether the ToC is expanded by default

cover:
  image: /images/my-cover.png   # optional cover image for the post
  alt: "Description of image"
---
```

### Drafts

Posts with `draft: true` are only visible when you run `hugo server -D`.
To publish, change `draft: false` (or delete the draft line).

### Organize posts by year (optional)

You can create subdirectories under `content/posts/`:

```
content/posts/
├── 2024/
│   └── first-post.md
└── 2025/
    └── second-post.md
```

---

## 6. Math with LaTeX

Math rendering is powered by **MathJax 3**. Enable it per-post with `math: true` in front matter, or globally via `params.math: true` in `hugo.yaml`.

### Inline math

Wrap with single dollar signs:

```markdown
The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
```

Renders as: The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

### Display math

Wrap with double dollar signs on their own lines:

```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
$$
```

### Numbered equations

Use the `equation` environment with `\label` and `\ref`:

```latex
$$
\begin{equation}
  \nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}
  \label{eq:gauss}
\end{equation}
$$

See equation \eqref{eq:gauss}.
```

### Built-in macros

The following shorthand macros are pre-configured in `layouts/partials/math.html`:

| Macro | Expands to | Example |
|---|---|---|
| `\R` | `\mathbb{R}` | $\mathbb{R}$ |
| `\N` | `\mathbb{N}` | $\mathbb{N}$ |
| `\E` | `\mathbb{E}` | $\mathbb{E}$ |
| `\P` | `\mathbb{P}` | $\mathbb{P}$ |
| `\norm{x}` | `\lVert x \rVert` | $\lVert x \rVert$ |
| `\abs{x}` | `\lvert x \rvert` | $\lvert x \rvert$ |

To add more macros, edit `layouts/partials/math.html`:

```javascript
MathJax = {
  tex: {
    macros: {
      R: '{\\mathbb{R}}',
      // add yours here:
      C: '{\\mathbb{C}}',
      indicator: ['{\\mathbf{1}_{#1}}', 1],
    }
  }
};
```

### Important: escaping in Hugo

Hugo's Markdown parser can mangle `_` and `^` inside math. This is prevented by the `passthrough` extension in `hugo.yaml`:

```yaml
markup:
  goldmark:
    extensions:
      passthrough:
        enable: true
        delimiters:
          block:
            - - '$$'
              - '$$'
          inline:
            - - '$'
              - '$'
```

Do **not** remove this section or math will break.

---

## 7. Code Snippets

Hugo uses **Chroma** for syntax highlighting — no JavaScript needed.

### Basic code block

Use triple backticks with a language identifier:

````markdown
```python
def hello(name: str) -> str:
    return f"Hello, {name}!"
```
````

### Supported languages (examples)

`python`, `r`, `julia`, `javascript`, `typescript`, `go`, `rust`, `bash`, `sql`, `latex`, `markdown`, `yaml`, `json`, `cpp`, `java` — and [180+ more](https://gohugo.io/content-management/syntax-highlighting/#list-of-chroma-highlighting-languages).

### Code block options

Add options after the language name:

```markdown
```python {linenos=true, hl_lines=[3,4], linenostart=10}
```

| Option | Effect |
|---|---|
| `linenos=true` | Show line numbers |
| `hl_lines=[2,5]` | Highlight specific lines |
| `linenostart=10` | Start line numbering at 10 |

### Copy button

The copy button is enabled globally via `hugo.yaml`:

```yaml
params:
  ShowCodeCopyButtons: true
```

No per-post configuration needed.

---

## 8. Profile Page

The homepage (`/`) is a fully custom academic profile, **not** PaperMod's default home layout. It is defined in `layouts/index.html`.

### Add your profile photo

Place a photo at `static/images/avatar.jpg`. The homepage will automatically display it. If the file is absent, it falls back to your initials **DMH**.

Recommended: square image, at least 300×300 px.

### Edit profile content

Open `layouts/index.html` and find the clearly marked sections:

- **HERO** — name, title, affiliation, bio, contact links
- **RESEARCH** — three research interest cards
- **PROJECTS** — project cards generated from your CV
- **EDUCATION** — degree entries
- **CONTACT** — email, GitHub, phone

Each section is plain HTML with comments — no Hugo templating knowledge needed to update text.

### Add a new project card

Copy an existing `.project-card` block and adjust:

```html
<div class="project-card">
  <div class="project-header">
    <p class="project-title">Your Project Title</p>
    <span class="project-badge badge-research">Research</span>
    <!-- badge types: badge-research | badge-thesis | badge-industry -->
  </div>
  <p class="project-org">Institution · Date range</p>
  <p class="project-desc">Description of the project...</p>
  <p class="project-outcome">🏆 Key outcome or result.</p>
  <div class="tag-list">
    <span class="tag">Tag1</span>
    <span class="tag">Tag2</span>
  </div>
</div>
```

---

## 9. Deployment to GitHub Pages

Deployment is fully automated via GitHub Actions. Every push to `main` triggers a build and deploy.

### First-time setup

1. Create a repository named `<username>.github.io` on GitHub.
2. Enable GitHub Pages: **Settings → Pages → Source → GitHub Actions**.
3. Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Hugo to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true        # fetches PaperMod theme

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true          # must be extended

      - name: Build
        run: hugo --minify

      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./public
      - id: deployment
        uses: actions/deploy-pages@v4
```

### Daily workflow

```bash
# Write or edit a post
hugo new content posts/new-post.md
# ... edit the file ...

# Preview locally
hugo server -D

# Publish
git add .
git commit -m "add: new post on XYZ"
git push
# GitHub Actions automatically builds and deploys
```

### Custom domain (optional)

1. Add a file `static/CNAME` containing just your domain:
   ```
   myblog.com
   ```
2. In your DNS provider, add a CNAME record pointing to `<username>.github.io`.
3. In GitHub: **Settings → Pages → Custom domain** → enter your domain and enable **Enforce HTTPS**.

---

## 10. Troubleshooting

### Blank page / no layout warnings

```
WARN found no layout file for "html" for kind "page"
```

**Cause:** PaperMod theme submodule is empty (not fetched).

**Fix:**
```bash
git submodule update --init --recursive
```

---

### Math not rendering

**Check 1** — Is `math: true` set in the post front matter or in `hugo.yaml` under `params`?

**Check 2** — Is the `passthrough` extension present in `hugo.yaml`? Without it, Hugo escapes `_` and `^` inside dollar signs.

**Check 3** — Open browser DevTools → Network tab. Confirm `tex-svg.js` loads from `cdn.jsdelivr.net` without errors.

---

### Code blocks have no color

The syntax CSS file is missing. Regenerate it:

```bash
hugo gen chromastyles --style=dracula > assets/css/extended/syntax.css
```

---

### Search returns no results

The JSON output index is required. Confirm `hugo.yaml` has:

```yaml
outputs:
  home:
    - HTML
    - RSS
    - JSON
```

And `content/search.md` exists with `layout: "search"` in its front matter.

---

### Profile photo not showing

Confirm the file is at exactly `static/images/avatar.jpg` (case-sensitive on Linux/macOS).
The fallback initials **DMH** display automatically if the file is absent.

---

### Deploy workflow fails with "theme not found"

The `actions/checkout` step must have `submodules: true`:

```yaml
- uses: actions/checkout@v4
  with:
    submodules: true   # ← this line is required
```
