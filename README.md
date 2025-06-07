# IntArchive Documentation

This repository contains the documentation and blog for IntArchive, built with Jekyll and GitHub Pages.

## 🚀 Quick Start

1. Install Ruby and Jekyll:
   ```bash
   gem install bundler jekyll
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/IntArchive/IntArchive.github.io.git
   cd IntArchive.github.io
   ```

3. Install dependencies:
   ```bash
   bundle install
   ```

4. Run the development server:
   ```bash
   bundle exec jekyll serve
   ```

5. Visit `http://localhost:4000` in your browser

## 📚 Content Structure

- `_posts/`: Blog posts and articles
- `_layouts/`: Page layout templates
- `_includes/`: Reusable components
- `assets/`: Static files (images, CSS, JS)
- `_config.yml`: Site configuration
- `index.html`: Homepage

## 🎨 Theme and Customization

The site uses a custom theme with modern design principles. To customize:

1. Modify `_config.yml` for site-wide settings
2. Edit CSS in `assets/css/`
3. Update layouts in `_layouts/`

## 📝 Writing Posts

Create new posts in `_posts/` following the format:
```
YYYY-MM-DD-title.md
```

Include front matter at the top of each post:
```yaml
---
layout: post
title: "Your Title"
date: YYYY-MM-DD
categories: [category1, category2]
---
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.