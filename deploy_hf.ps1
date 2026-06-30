# Deploy to HuggingFace Spaces (always creates clean orphan to avoid history issues)
Write-Host "Deploying to HuggingFace..." -ForegroundColor Cyan

# Save current branch
$current = git rev-parse --abbrev-ref HEAD

# Create fresh orphan branch
git checkout --orphan _hf_tmp
# Remove any binary image files HuggingFace rejects
git rm --cached static/css/background.png 2>$null
git add .gitattributes .gitignore Dockerfile README.md
git add app.py cricket_analytics_core.py gunicorn.conf.py pyproject.toml requirements.txt runtime.txt build.sh
git add data/ templates/
# Stage static but exclude PNG (HF uses Xet for images, not LFS)
git add static/css/style.css
git add static/js/

# Commit and push
git commit -m "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push hf _hf_tmp:main --force

# Return to original branch and clean up
git checkout -f $current
git branch -D _hf_tmp 2>$null

Write-Host "Done! Live at: https://goluibpr-cricket-analytics.hf.space" -ForegroundColor Green
