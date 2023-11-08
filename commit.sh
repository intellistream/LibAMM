
BRANCH=backup_11_08
git init
git checkout -b $BRANCH
git add .
git commit -F commit_info
git push origin $BRANCH
